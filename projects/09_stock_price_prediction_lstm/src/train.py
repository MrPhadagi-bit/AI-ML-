from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "stock_lstm.keras"


def generate_demo_data(points: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=points)
    trend = np.linspace(100, 140, points)
    seasonality = 4 * np.sin(np.arange(points) / 6)
    noise = rng.normal(0, 0.8, size=points)
    close_prices = trend + seasonality + noise
    return pd.DataFrame({"Date": dates, "Close": close_prices.round(2)})


def load_dataset(data_path: Path | None) -> pd.DataFrame:
    if data_path and data_path.exists():
        dataset = pd.read_csv(data_path)
    else:
        dataset = generate_demo_data()

    if "Close" not in dataset.columns:
        raise ValueError("Expected a 'Close' column in the stock price dataset.")
    return dataset.sort_values(by=dataset.columns[0]).reset_index(drop=True)


def create_sequences(values: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    features, targets = [], []
    for index in range(len(values) - window_size):
        features.append(values[index : index + window_size])
        targets.append(values[index + window_size])
    return np.array(features), np.array(targets)


def build_model(window_size: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(window_size, 1)),
            tf.keras.layers.LSTM(64, return_sequences=False),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an LSTM stock price forecaster.")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    dataset = load_dataset(args.data_path)
    scaler = MinMaxScaler()
    scaled_close = scaler.fit_transform(dataset[["Close"]].values)
    x, y = create_sequences(scaled_close, args.window_size)

    if len(x) == 0:
        raise ValueError("Not enough rows to build sequences. Add more price history or lower --window-size.")

    split_index = int(len(x) * 0.8)
    x_train, x_test = x[:split_index], x[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    model = build_model(args.window_size)
    model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.1,
        verbose=1,
    )

    predictions = model.predict(x_test, verbose=0)
    predicted_prices = scaler.inverse_transform(predictions)
    actual_prices = scaler.inverse_transform(y_test)
    rmse = float(np.sqrt(np.mean((predicted_prices - actual_prices) ** 2)))

    print(f"Test RMSE: {rmse:.4f}")
    print("Sample predictions:")
    for predicted, actual in zip(predicted_prices[:5].flatten(), actual_prices[:5].flatten()):
        print(f"  predicted={predicted:.2f}, actual={actual:.2f}")

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_path)
    print(f"Model saved to: {args.model_path}")


if __name__ == "__main__":
    main()

