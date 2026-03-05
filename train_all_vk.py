"""
Batch training of LSTM pollen models for all species in pollen_vk.csv.

Three scenarios are supported (controlled with --scenario):

  baseline         – weather2 = observed same-day weather (no real forecast).
                     This is the starting baseline.
  oracle_forecast  – weather2 = data_vk/weather_forecast_oracle_vk.csv
                     That file was produced by weather_forecaster.py --mode oracle:
                     for date t it stores the ACTUAL observed weather at t+horizon
                     (upper bound – perfect forecast).
  lstm_forecast    – weather2 = data_vk/weather_forecast_lstm_vk.csv
                     Produced by weather_forecaster.py --mode lstm:
                     for date t it stores the model-predicted weather at t+horizon
                     (realistic scenario – own meteorological prognosis).

Running all three scenarios and comparing the summary CSVs shows how much
the quality of the weather forecast drives pollen prediction accuracy.

Usage:
  python train_all_vk.py --scenario baseline
  python train_all_vk.py --scenario oracle_forecast
  python train_all_vk.py --scenario lstm_forecast
  python train_all_vk.py --scenario baseline --min-nonzero 30 --epochs 50
"""

import argparse
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import joblib

WEATHER_COLS = ["temp_max", "pressure", "wind"]
SEQ_LEN_DEFAULT = 7
HORIZON_DEFAULT = 1
EPOCHS_DEFAULT = 100
BATCH_SIZE_DEFAULT = 365
MIN_NONZERO_DEFAULT = 50   # skip species with fewer non-zero training samples


# ---------------------------------------------------------------------------
# Helpers (same logic as pollen_lstm_simple.py)
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
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

    X_list, y_list, y_dates = [], [], []
    last_idx = len(df) - horizon
    for i in range(seq_len - 1, last_idx):
        x = features[i - seq_len + 1 : i + 1]
        y_val = target[i + horizon]
        if np.isnan(y_val) or np.isnan(x).any():
            continue
        X_list.append(x)
        y_list.append(y_val)
        y_dates.append(df["Date"].iloc[i + horizon])

    if not X_list:
        raise ValueError("No valid sequences – check missing values.")

    return np.array(X_list), np.array(y_list), y_dates


def build_feature_cols(
    data: pd.DataFrame,
    weather: pd.DataFrame,
    weather2_path: Path,
    forecast_cols_base: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Merge weather2 into data; return updated data and feature list."""
    weather2 = load_csv(weather2_path)
    data = data.copy()
    data["DayOfYear"] = data["Date"].dt.dayofyear
    for c in WEATHER_COLS:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    weather2 = weather2.rename(
        columns={c: f"f_{c}" for c in weather2.columns if c != "Date"}
    )
    data = data.merge(weather2, on="Date", how="inner")
    data["f_DayOfYear"] = data["Date"].dt.dayofyear
    for c in WEATHER_COLS:
        data[f"f_{c}"] = pd.to_numeric(data[f"f_{c}"], errors="coerce")

    feature_cols = (
        ["DayOfYear"]
        + WEATHER_COLS
        + ["f_DayOfYear"]
        + [f"f_{c}" for c in WEATHER_COLS]
    )
    return data, feature_cols


# ---------------------------------------------------------------------------
# Single-species training
# ---------------------------------------------------------------------------

def train_species(
    target: str,
    base_data: pd.DataFrame,
    weather2_path: Path,
    output_dir: Path,
    seq_len: int,
    horizon: int,
    epochs: int,
    batch_size: int,
    min_nonzero: int,
    scenario: str,
) -> dict:
    """Train LSTM for one pollen species; return metrics dict."""
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise ImportError("TensorFlow is required: pip install tensorflow") from exc

    target_slug = target.lower().replace(" ", "_").replace("/", "_").replace(".", "")
    species_dir = output_dir / target_slug
    species_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Prepare data
    # ------------------------------------------------------------------
    data = base_data.copy()
    if target not in data.columns:
        return {"target": target, "status": "skipped", "reason": "column_not_found"}

    data, feature_cols = build_feature_cols(data, base_data, weather2_path, WEATHER_COLS)
    data = data.dropna(subset=[target]).sort_values("Date")
    data[feature_cols] = data[feature_cols].fillna(
        data[feature_cols].median(numeric_only=True)
    )

    # ------------------------------------------------------------------
    # Scale
    # ------------------------------------------------------------------
    model_cols = [target] + feature_cols
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(data[model_cols].to_numpy(dtype=float))
    scaled_df = pd.DataFrame(scaled_values, columns=model_cols)
    scaled_df["Date"] = data["Date"].values

    # ------------------------------------------------------------------
    # Sequences + train/test split 70/30
    # ------------------------------------------------------------------
    X, y, y_dates = build_sequences(scaled_df, feature_cols, target, seq_len, horizon)
    y_dates = pd.to_datetime(y_dates)
    n_total = len(y_dates)
    n_train = max(1, int(n_total * 0.7))
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    if X_test.shape[0] == 0:
        return {"target": target, "status": "skipped", "reason": "empty_test_set"}

    # Check non-zero in training target
    nz = int(np.sum(y_train != 0))
    if nz < min_nonzero:
        return {
            "target": target,
            "status": "skipped",
            "reason": f"nonzero_train_samples={nz}<{min_nonzero}",
        }

    # ------------------------------------------------------------------
    # Build & train model
    # ------------------------------------------------------------------
    log_path = species_dir / f"vk_{target_slug}_{scenario}_training_log.csv"
    callbacks = [keras.callbacks.CSVLogger(str(log_path), append=False)]

    model = keras.Sequential([
        keras.layers.Input(shape=(seq_len, X_train.shape[-1])),
        keras.layers.LSTM(64),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=callbacks,
        verbose=0,
    )

    # ------------------------------------------------------------------
    # Evaluate on test set (inverse-transform back to original scale)
    # ------------------------------------------------------------------
    preds = model.predict(X_test, verbose=0).reshape(-1)
    test_X_last = X_test[:, -1, :]

    inv_yhat = scaler.inverse_transform(
        np.concatenate([preds.reshape(-1, 1), test_X_last], axis=1)
    )[:, 0]
    inv_y = scaler.inverse_transform(
        np.concatenate([y_test.reshape(-1, 1), test_X_last], axis=1)
    )[:, 0]

    mae = float(np.mean(np.abs(inv_y - inv_yhat)))
    rmse = float(np.sqrt(np.mean((inv_y - inv_yhat) ** 2)))
    r2 = float(r2_score(inv_y, inv_yhat))

    # ------------------------------------------------------------------
    # Save model, scaler, metrics
    # ------------------------------------------------------------------
    model_path = species_dir / f"vk_{target_slug}_{scenario}_lstm.keras"
    scaler_path = species_dir / f"vk_{target_slug}_{scenario}_scaler.joblib"
    model.save(model_path)
    joblib.dump(
        {"scaler": scaler, "features": feature_cols, "seq_len": seq_len},
        scaler_path,
    )

    target_stats = data[target].describe()
    metrics_path = species_dir / f"vk_{target_slug}_{scenario}_metrics.txt"
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write(f"dataset: vk\n")
        f.write(f"target: {target}\n")
        f.write(f"scenario: {scenario}\n")
        f.write(f"run_date: {datetime.now().isoformat()}\n")
        f.write(f"date_range: {data['Date'].min().date()} to {data['Date'].max().date()}\n")
        f.write(f"rows: {len(data)}\n")
        f.write(f"features: {len(feature_cols)}\n")
        f.write(f"seq_len: {seq_len}\n")
        f.write(f"horizon: {horizon}\n")
        f.write(f"train_samples: {n_train}\n")
        f.write(f"test_samples: {len(X_test)}\n")
        f.write(f"mae: {mae:.4f}\n")
        f.write(f"rmse: {rmse:.4f}\n")
        f.write(f"r2: {r2:.4f}\n")
        f.write(f"min: {target_stats['min']:.3f}\n")
        f.write(f"median: {target_stats['50%']:.3f}\n")
        f.write(f"max: {target_stats['max']:.3f}\n")

    return {
        "target": target,
        "status": "ok",
        "scenario": scenario,
        "train_samples": n_train,
        "test_samples": len(X_test),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "median": float(target_stats["50%"]),
        "max": float(target_stats["max"]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch LSTM training for all VK pollen species."
    )
    parser.add_argument(
        "--pollen", default=Path("data_vk/pollen_vk.csv"), type=Path
    )
    parser.add_argument(
        "--weather", default=Path("data_vk/weather_vk.csv"), type=Path
    )
    parser.add_argument(
        "--weather2", default=None, type=Path,
        help="Forecast weather file. Defaults to --weather (baseline).",
    )
    parser.add_argument(
        "--scenario",
        default="baseline",
        choices=["baseline", "oracle_forecast", "lstm_forecast"],
        help=(
            "baseline: weather2=weather (no real forecast). "
            "oracle_forecast: actual future weather (upper bound). "
            "lstm_forecast: LSTM-predicted future weather."
        ),
    )
    parser.add_argument("--output", default=Path("models_vk"), type=Path)
    parser.add_argument("--seq-len", default=SEQ_LEN_DEFAULT, type=int)
    parser.add_argument("--horizon", default=HORIZON_DEFAULT, type=int)
    parser.add_argument("--epochs", default=EPOCHS_DEFAULT, type=int)
    parser.add_argument("--batch-size", default=BATCH_SIZE_DEFAULT, type=int)
    parser.add_argument(
        "--min-nonzero", default=MIN_NONZERO_DEFAULT, type=int,
        help="Skip species with fewer non-zero training samples than this.",
    )
    parser.add_argument(
        "--targets", nargs="*", default=None,
        help="Train only these targets (default: all pollen columns).",
    )
    args = parser.parse_args()

    # Resolve weather2
    if args.weather2 is None:
        if args.scenario == "oracle_forecast":
            args.weather2 = Path("data_vk/weather_forecast_oracle_vk.csv")
        elif args.scenario == "lstm_forecast":
            args.weather2 = Path("data_vk/weather_forecast_lstm_vk.csv")
        else:
            args.weather2 = args.weather  # baseline

    if not args.weather2.exists():
        raise FileNotFoundError(
            f"weather2 file not found: {args.weather2}\n"
            "Run weather_forecaster.py first to generate forecast files."
        )

    # Load data
    pollen = load_csv(args.pollen)
    weather = load_csv(args.weather)
    base_data = pollen.merge(weather, on="Date", how="inner")

    for c in WEATHER_COLS:
        if c not in base_data.columns:
            raise ValueError(f"Expected weather column '{c}' not found.")

    # Determine target species
    pollen_cols = [c for c in pollen.columns if c != "Date"]
    targets = args.targets if args.targets else pollen_cols

    print(
        f"\n=== train_all_vk | scenario={args.scenario} | "
        f"{len(targets)} species | epochs={args.epochs} ===\n",
        flush=True,
    )

    results = []
    for idx, target in enumerate(targets, 1):
        print(f"[{idx}/{len(targets)}] {target} ...", end=" ", flush=True)
        try:
            row = train_species(
                target=target,
                base_data=base_data,
                weather2_path=args.weather2,
                output_dir=args.output,
                seq_len=args.seq_len,
                horizon=args.horizon,
                epochs=args.epochs,
                batch_size=args.batch_size,
                min_nonzero=args.min_nonzero,
                scenario=args.scenario,
            )
            if row["status"] == "ok":
                print(
                    f"MAE={row['mae']:.3f}  RMSE={row['rmse']:.3f}  R²={row['r2']:.4f}",
                    flush=True,
                )
            else:
                print(f"SKIPPED ({row.get('reason', '')})", flush=True)
        except Exception as exc:
            print(f"ERROR: {exc}", flush=True)
            traceback.print_exc()
            row = {"target": target, "status": "error", "reason": str(exc)}
        results.append(row)

    # Save summary CSV
    summary_df = pd.DataFrame(results)
    summary_path = args.output / f"summary_{args.scenario}.csv"
    args.output.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)

    ok = summary_df[summary_df["status"] == "ok"]
    skipped = summary_df[summary_df["status"] != "ok"]
    print(f"\n=== Done: {len(ok)} trained, {len(skipped)} skipped/errors ===")
    if not ok.empty:
        print(f"Median R²  : {ok['r2'].median():.4f}")
        print(f"Median RMSE: {ok['rmse'].median():.4f}")
    print(f"Summary saved: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
