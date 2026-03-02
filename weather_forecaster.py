"""
Weather forecast file generator for pollen prediction comparison.

Two modes (--mode):

  oracle  – For each date t, stores the ACTUAL observed weather at t+horizon.
             This is the "perfect forecast" upper bound: if the pollen model had
             access to exactly correct future weather, how good would it be?
             Output: data_vk/weather_forecast_oracle_vk.csv

  lstm    – Trains an LSTM on the weather time series (first 70 % of dates)
             and uses it to predict weather at t+horizon for EVERY date.
             Output: data_vk/weather_forecast_lstm_vk.csv

Both output files have the same columns as weather_vk.csv.
Crucially, for row with Date=t, the weather values represent the forecast for
t+horizon, NOT for t.  When pollen_lstm_simple.py / train_all_vk.py merges
this file on Date=t it therefore feeds "tomorrow's weather" as a feature,
which is the intended design.

Usage:
  python weather_forecaster.py --mode oracle
  python weather_forecaster.py --mode lstm
  python weather_forecaster.py --mode lstm --epochs 200 --seq-len 14
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

WEATHER_COLS_VK = ["temp_max", "temp_min", "temp_mean", "precip",
                   "pressure", "sunshine", "wind", "humid_min", "visibility"]
SEQ_LEN_DEFAULT  = 7
HORIZON_DEFAULT  = 1
EPOCHS_DEFAULT   = 100
BATCH_SIZE_DEFAULT = 365


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_weather(path: Path, weather_cols: List[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    for c in weather_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Fill missing with column median to keep sequences continuous
    df[weather_cols] = df[weather_cols].fillna(df[weather_cols].median(numeric_only=True))
    return df


def build_sequences_weather(
    values: np.ndarray,
    seq_len: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) where:
      X[i] = values[i : i+seq_len]           (seq_len days of all weather vars)
      y[i] = values[i + seq_len - 1 + horizon] (target: weather horizon days after
                                                 the last element of the window)
    """
    X_list, y_list = [], []
    last_start = len(values) - seq_len - horizon + 1
    for i in range(last_start):
        X_list.append(values[i : i + seq_len])
        y_list.append(values[i + seq_len - 1 + horizon])
    return np.array(X_list), np.array(y_list)


# ---------------------------------------------------------------------------
# Oracle mode
# ---------------------------------------------------------------------------

def generate_oracle(
    df: pd.DataFrame,
    weather_cols: List[str],
    horizon: int,
    output_path: Path,
) -> None:
    """
    For each date t, store the ACTUAL weather at t+horizon.
    Row at position i gets values from position i+horizon.
    The last `horizon` rows are dropped (they have no future target).
    """
    out = df[["Date"] + weather_cols].copy()
    # shift values backward: row i gets values from row i+horizon
    out[weather_cols] = out[weather_cols].shift(-horizon)
    out = out.iloc[: len(out) - horizon].reset_index(drop=True)
    out.to_csv(output_path, index=False)
    print(f"Oracle forecast saved: {output_path}  ({len(out)} rows)")


# ---------------------------------------------------------------------------
# LSTM mode
# ---------------------------------------------------------------------------

def generate_lstm_forecast(
    df: pd.DataFrame,
    weather_cols: List[str],
    horizon: int,
    seq_len: int,
    epochs: int,
    batch_size: int,
    output_path: Path,
    model_dir: Path,
) -> None:
    """
    Train a multi-output LSTM on the first 70 % of data, then generate
    predictions for ALL dates using the trained model.

    The output file has the same format as the oracle file:
      row at Date=t  →  predicted weather at t+horizon.
    """
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise ImportError("TensorFlow is required: pip install tensorflow") from exc

    n = len(df)
    n_train = int(n * 0.7)

    # Scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    values_scaled = scaler.fit_transform(df[weather_cols].to_numpy(dtype=float))  # (n, n_vars)

    # Build sequences from the full dataset (for inference later)
    X_all, _ = build_sequences_weather(values_scaled, seq_len, horizon)
    # X_all[i] corresponds to predicting weather at index: i + seq_len - 1 + horizon
    # i.e., for input window ending at row (i + seq_len - 1), predict row (i + seq_len - 1 + horizon)

    # Training sequences: only those whose TARGET falls within the training set
    max_train_i = n_train - seq_len - horizon  # last i s.t. target index < n_train
    if max_train_i < 1:
        raise ValueError("Not enough training data for weather LSTM.")

    _, y_train_scaled = build_sequences_weather(values_scaled[:n_train + horizon], seq_len, horizon)
    X_train = X_all[:max_train_i + 1]
    y_train = y_train_scaled[:max_train_i + 1]

    n_vars = len(weather_cols)

    model = keras.Sequential([
        keras.layers.Input(shape=(seq_len, n_vars)),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.LSTM(32),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(n_vars),   # multi-output: all weather vars at t+horizon
    ])
    model.compile(optimizer="adam", loss="mse")

    log_path = model_dir / "weather_forecaster_training_log.csv"
    model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=[keras.callbacks.CSVLogger(str(log_path), append=False)],
        verbose=1,
    )

    # Save model + scaler
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / "weather_lstm_forecaster.keras")
    joblib.dump(
        {"scaler": scaler, "weather_cols": weather_cols, "seq_len": seq_len, "horizon": horizon},
        model_dir / "weather_lstm_forecaster.joblib",
    )

    # Predict for all windows
    preds_scaled = model.predict(X_all, verbose=0)            # (n_windows, n_vars)
    preds = scaler.inverse_transform(preds_scaled)             # back to original scale

    # Map predictions back to dates:
    # X_all[i] uses rows i..i+seq_len-1 →  predicts row i+seq_len-1+horizon
    # But in our output file we want: Date[t] → forecast for t+horizon
    # i.e., row at original index (i+seq_len-1) gets the prediction for (i+seq_len-1+horizon)
    # So "anchor date" index = i + seq_len - 1
    anchor_indices = np.arange(seq_len - 1, seq_len - 1 + len(X_all))   # indices in df
    # We only keep anchor_indices where t+horizon < n (valid target)
    valid_mask = (anchor_indices + horizon) < n
    anchor_indices = anchor_indices[valid_mask]
    preds = preds[valid_mask]

    out_dates = df["Date"].iloc[anchor_indices].values
    out_df = pd.DataFrame(preds, columns=weather_cols)
    out_df.insert(0, "Date", out_dates)

    out_df.to_csv(output_path, index=False)
    print(f"LSTM forecast saved:   {output_path}  ({len(out_df)} rows)")
    print(f"Model saved:           {model_dir / 'weather_lstm_forecaster.keras'}")
    print(f"Training log:          {log_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate weather forecast files for the pollen pipeline."
    )
    parser.add_argument(
        "--weather", default=Path("data_vk/weather_vk.csv"), type=Path
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["oracle", "lstm"],
        help=(
            "oracle: use actual future weather (upper bound). "
            "lstm:   use LSTM-predicted future weather (realistic)."
        ),
    )
    parser.add_argument("--horizon",    default=HORIZON_DEFAULT,    type=int)
    parser.add_argument("--seq-len",    default=SEQ_LEN_DEFAULT,    type=int)
    parser.add_argument("--epochs",     default=EPOCHS_DEFAULT,     type=int)
    parser.add_argument("--batch-size", default=BATCH_SIZE_DEFAULT, type=int)
    parser.add_argument(
        "--output-dir", default=Path("data_vk"), type=Path,
        help="Directory where forecast CSV files are saved."
    )
    parser.add_argument(
        "--model-dir", default=Path("models_vk/_weather_forecaster"), type=Path,
        help="Directory where weather LSTM model is saved (lstm mode only)."
    )
    args = parser.parse_args()

    # Detect available weather columns
    raw = pd.read_csv(args.weather, nrows=0)
    weather_cols = [c for c in WEATHER_COLS_VK if c in raw.columns]
    if not weather_cols:
        raise ValueError(f"No known weather columns found in {args.weather}.")
    print(f"Weather columns used: {weather_cols}")

    df = load_weather(args.weather, weather_cols)
    print(f"Loaded {len(df)} rows  ({df['Date'].min().date()} – {df['Date'].max().date()})")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "oracle":
        out_path = args.output_dir / "weather_forecast_oracle_vk.csv"
        generate_oracle(df, weather_cols, args.horizon, out_path)

    elif args.mode == "lstm":
        out_path = args.output_dir / "weather_forecast_lstm_vk.csv"
        generate_lstm_forecast(
            df=df,
            weather_cols=weather_cols,
            horizon=args.horizon,
            seq_len=args.seq_len,
            epochs=args.epochs,
            batch_size=args.batch_size,
            output_path=out_path,
            model_dir=args.model_dir,
        )


if __name__ == "__main__":
    main()
