from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE_PATH = PROJECT_ROOT / "artifacts" / "stock_forecaster.joblib"


def generate_demo_data(points: int = 220) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=points)
    trend = np.linspace(100, 145, points)
    seasonality = 3.5 * np.sin(np.arange(points) / 7)
    noise = rng.normal(0, 0.7, size=points)
    close_prices = trend + seasonality + noise
    return pd.DataFrame({"Date": dates, "Close": close_prices.round(2)})


def load_dataset(data_path: Path | None) -> pd.DataFrame:
    if data_path and data_path.exists():
        dataset = pd.read_csv(data_path)
    else:
        dataset = generate_demo_data()
    return dataset.sort_values(by=dataset.columns[0]).reset_index(drop=True)


def inverse_transform_column(scaler, values: np.ndarray) -> np.ndarray:
    return scaler.inverse_transform(values.reshape(-1, 1)).flatten()


def forecast_future(model: object, recent_window: np.ndarray, steps: int) -> np.ndarray:
    window = recent_window.astype(np.float64).copy()
    predictions: list[float] = []
    for _ in range(steps):
        next_value = float(model.predict(window.reshape(1, -1))[0])
        predictions.append(next_value)
        window = np.concatenate([window[1:], np.array([next_value])])
    return np.array(predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Forecast future stock prices with a trained model bundle.")
    parser.add_argument("--bundle-path", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    bundle = joblib.load(args.bundle_path)
    dataset = load_dataset(args.data_path)
    scaled_values = bundle["scaler"].transform(dataset[["Close"]].values).flatten()
    recent_window = scaled_values[-bundle["window_size"] :]
    future_scaled = forecast_future(bundle["model"], recent_window, args.steps)
    future_prices = inverse_transform_column(bundle["scaler"], future_scaled)
    future_dates = pd.date_range(
        start=pd.to_datetime(dataset.iloc[-1, 0]) + pd.Timedelta(days=1),
        periods=args.steps,
    )

    print(f"Model: {bundle['best_model_name']}")
    for date, price in zip(future_dates, future_prices):
        print(f"{date.date()}: {price:.2f}")


if __name__ == "__main__":
    main()
