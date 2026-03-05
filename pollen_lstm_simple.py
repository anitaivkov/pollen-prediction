""" Run example: python pollen_lstm_simple.py --pollen data_vk/pollen_vk.csv --weather data_vk/weather_vk.csv --weather2 data_vk/weather_vk.csv --target Ambrosia --output models_vk
Change --target to test other pollen types (e.g., Betula, Corylus, etc.). """

import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import joblib


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    return df


def build_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    seq_len: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    features = df[feature_cols].to_numpy(dtype=float)
    target = df[target_col].to_numpy(dtype=float)

    X_list = []
    y_list = []
    y_dates = []

    last_idx = len(df) - horizon
    for i in range(seq_len - 1, last_idx):
        x = features[i - seq_len + 1:i + 1]
        y_val = target[i + horizon]
        if np.isnan(y_val) or np.isnan(x).any():
            continue
        X_list.append(x)
        y_list.append(y_val)
        y_dates.append(df["Date"].iloc[i + horizon])

    if not X_list:
        raise ValueError("No valid sequences built. Check missing values.")

    return np.array(X_list), np.array(y_list), y_dates


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple LSTM for pollen prediction.")
    parser.add_argument("--pollen", default=Path("data_vk/pollen_vk.csv"), type=Path, help="pollen.csv")
    parser.add_argument("--weather", default=Path("data_vk/weather_vk.csv"), type=Path, help="weather.csv")
    parser.add_argument("--weather2", default=Path("data_vk/weather_vk.csv"), type=Path, help="weather2.csv")
    parser.add_argument("--target", required=True, help="Target pollen column (e.g., Ambrosia)")
    parser.add_argument("--seq-len", default=7, type=int, help="Sequence length")
    parser.add_argument("--horizon", default=1, type=int, help="Days ahead")
    # --test-days više nije potreban, automatski dijelimo 70/30
    parser.add_argument("--epochs", default=100, type=int, help="Epochs")
    parser.add_argument("--batch-size", default=365, type=int, help="Batch size")
    parser.add_argument("--output", default=Path("models_vk"), type=Path, help="Output dir")
    parser.add_argument("--log-file", default="training_log.csv", help="Epoch log filename")
    args = parser.parse_args()

    pollen = load_csv(args.pollen)
    weather = load_csv(args.weather)
    weather2 = load_csv(args.weather2)

    data = pollen.merge(weather, on="Date", how="inner")
    if args.target not in data.columns:
        raise ValueError(f"Target '{args.target}' not found. Available: {sorted(data.columns)}")

    vk_weather_cols = ["temp_max", "temp_min", "temp_mean", "precip", "pressure", "sunshine", "wind", "humid_min", "visibility"]
    lux_weather_cols = ["TempMax", "HumidMin", "VisibilityAvg"]
    if all(col in weather.columns for col in ["temp_max", "pressure", "wind"]):
        # Odaberi 3 najsličnije originalima: temp_max (TempMax), pressure (umjesto HumidMin), wind (umjesto VisibilityAvg)
        weather_cols = ["temp_max", "pressure", "wind"]
        forecast_cols = [f"f_{c}" for c in weather_cols]
        data["DayOfYear"] = data["Date"].dt.dayofyear
        for c in weather_cols:
            data[c] = pd.to_numeric(data[c], errors="coerce")
        weather2 = weather2.rename(columns={c: f"f_{c}" for c in weather2.columns if c != "Date"})
        data = data.merge(weather2, on="Date", how="inner")
        data["f_DayOfYear"] = data["Date"].dt.dayofyear
        for c in weather_cols:
            data[f"f_{c}"] = pd.to_numeric(data[f"f_{c}"], errors="coerce")
        feature_cols = ["DayOfYear"] + weather_cols + ["f_DayOfYear"] + forecast_cols
        dataset_name = "vk"
    else:
        weather_cols = lux_weather_cols
        forecast_cols = [f"f_{c}" for c in weather_cols]
        data["DayOfYear"] = data["Date"].dt.dayofyear
        for c in weather_cols:
            data[c] = pd.to_numeric(data[c], errors="coerce")
        weather2 = weather2.rename(columns={c: f"f_{c}" for c in weather2.columns if c != "Date"})
        data = data.merge(weather2, on="Date", how="inner")
        data["f_DayOfYear"] = data["Date"].dt.dayofyear
        for c in weather_cols:
            data[f"f_{c}"] = pd.to_numeric(data[f"f_{c}"], errors="coerce")
        feature_cols = ["DayOfYear"] + weather_cols + ["f_DayOfYear"] + forecast_cols
        dataset_name = "lux"

    data = data.dropna(subset=[args.target]).sort_values("Date")
    data[feature_cols] = data[feature_cols].fillna(data[feature_cols].median(numeric_only=True))
    model_cols = [args.target] + feature_cols
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(data[model_cols].to_numpy(dtype=float))
    scaled_df = pd.DataFrame(scaled_values, columns=model_cols)
    scaled_df["Date"] = data["Date"].values


    X, y, y_dates = build_sequences(scaled_df, feature_cols, args.target, args.seq_len, args.horizon)
    y_dates = pd.to_datetime(y_dates)
    # Automatski podijeli 70% train, 30% test
    n_total = len(y_dates)
    n_train = int(n_total * 0.7)
    if n_train < 1:
        n_train = 1
    train_mask = np.arange(n_total) < n_train
    test_mask = ~train_mask
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    if X_test.shape[0] == 0:
        raise ValueError("Test set is empty. Provide more data.")

    try:
        from tensorflow import keras
    except ImportError as exc:
        raise ImportError("TensorFlow is required. Install with: pip install tensorflow") from exc


    target_slug = args.target.lower()
    output_dir = args.output / target_slug
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"{dataset_name}_{target_slug}_training_log.csv"
    callbacks = [keras.callbacks.CSVLogger(str(log_path), append=False)]

    model = keras.Sequential([
        keras.layers.Input(shape=(args.seq_len, X_train.shape[-1])),
        keras.layers.LSTM(64),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")

    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=False,
        callbacks=callbacks,
        verbose=1,
    )

    preds = model.predict(X_test, verbose=0).reshape(-1)

    test_X_last = X_test[:, -1, :]
    inv_yhat = np.concatenate([preds.reshape(-1, 1), test_X_last], axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)[:, 0]

    inv_y = np.concatenate([y_test.reshape(-1, 1), test_X_last], axis=1)
    inv_y = scaler.inverse_transform(inv_y)[:, 0]

    mae = np.mean(np.abs(inv_y - inv_yhat))
    rmse = np.sqrt(np.mean((inv_y - inv_yhat) ** 2))
    r2 = r2_score(inv_y, inv_yhat)


    model_path = output_dir / f"{dataset_name}_{target_slug}_lstm.keras"
    scaler_path = output_dir / f"{dataset_name}_{target_slug}_scaler.joblib"
    model.save(model_path)
    joblib.dump({"scaler": scaler, "features": feature_cols, "seq_len": args.seq_len}, scaler_path)

    target_stats = data[args.target].describe()
    metrics_path = output_dir / f"{dataset_name}_{target_slug}_metrics.txt"
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write(f"dataset: {dataset_name}\n")
        f.write(f"target: {args.target}\n")
        f.write(f"run_date: {datetime.now().isoformat()}\n")
        f.write(f"date_range: {data['Date'].min().date()} to {data['Date'].max().date()}\n")
        f.write(f"rows: {len(data)}\n")
        f.write(f"features: {len(feature_cols)}\n")
        f.write(f"seq_len: {args.seq_len}\n")
        f.write(f"horizon: {args.horizon}\n")
        f.write(f"train_samples: {len(X_train)}\n")
        f.write(f"test_samples: {len(X_test)}\n")
        f.write(f"mae: {mae:.3f}\n")
        f.write(f"rmse: {rmse:.3f}\n")
        f.write(f"r2: {r2:.4f}\n")
        f.write(f"min: {target_stats['min']:.3f}\n")
        f.write(f"median: {target_stats['50%']:.3f}\n")
        f.write(f"max: {target_stats['max']:.3f}\n")

    print(f"MAE: {mae:.3f} | RMSE: {rmse:.3f} | R²: {r2:.4f}", flush=True)
    print(f"Saved: {model_path}", flush=True)
    print(f"Saved: {scaler_path}", flush=True)
    print(f"Saved: {log_path}", flush=True)
    print(f"Saved: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
