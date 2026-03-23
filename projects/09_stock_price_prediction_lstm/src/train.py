from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_BUNDLE_PATH = ARTIFACTS_DIR / "stock_forecaster.joblib"
DEFAULT_REPORT_PATH = ARTIFACTS_DIR / "stock_forecaster_report.json"
DEFAULT_FORECAST_PATH = ARTIFACTS_DIR / "next_5_day_forecast.csv"


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
    if "Close" not in dataset.columns:
        raise ValueError("Expected a 'Close' column in the dataset.")
    return dataset.sort_values(by=dataset.columns[0]).reset_index(drop=True)


def create_sequences(values: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    x_values, y_values = [], []
    for index in range(len(values) - window_size):
        x_values.append(values[index : index + window_size].flatten())
        y_values.append(values[index + window_size][0])
    return np.array(x_values), np.array(y_values)


def split_time_series(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, ...]:
    train_end = int(len(x) * 0.7)
    val_end = int(len(x) * 0.85)
    return (
        x[:train_end],
        x[train_end:val_end],
        x[val_end:],
        y[:train_end],
        y[train_end:val_end],
        y[val_end:],
    )


def build_candidates() -> dict[str, object]:
    return {
        "ridge": Ridge(alpha=1.0),
        "mlp_regressor": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1500, random_state=42),
    }


def inverse_transform_column(scaler: MinMaxScaler, values: np.ndarray) -> np.ndarray:
    return scaler.inverse_transform(values.reshape(-1, 1)).flatten()


def forecast_future(model: object, recent_window: np.ndarray, steps: int) -> np.ndarray:
    window = recent_window.astype(np.float64).copy()
    predictions: list[float] = []
    for _ in range(steps):
        next_value = float(model.predict(window.reshape(1, -1))[0])
        predictions.append(next_value)
        window = np.concatenate([window[1:], np.array([next_value])])
    return np.array(predictions)


def train(
    data_path: Path | None,
    window_size: int,
    forecast_steps: int,
    bundle_path: Path,
    report_path: Path,
    forecast_path: Path,
) -> dict[str, object]:
    dataset = load_dataset(data_path)
    scaler = MinMaxScaler()
    scaled_close = scaler.fit_transform(dataset[["Close"]].values)
    x, y = create_sequences(scaled_close, window_size)
    x_train, x_val, x_test, y_train, y_val, y_test = split_time_series(x, y)

    comparison: dict[str, float] = {}
    best_name = ""
    best_rmse = float("inf")
    best_model = None

    for name, candidate in build_candidates().items():
        candidate.fit(x_train, y_train)
        val_predictions = candidate.predict(x_val)
        rmse = float(mean_squared_error(y_val, val_predictions, squared=False))
        comparison[name] = rmse
        if rmse < best_rmse:
            best_rmse = rmse
            best_name = name
            best_model = candidate

    if best_model is None:
        raise RuntimeError("No sequence models were available.")

    best_model.fit(np.vstack([x_train, x_val]), np.concatenate([y_train, y_val]))
    test_predictions = best_model.predict(x_test)
    actual_prices = inverse_transform_column(scaler, y_test)
    predicted_prices = inverse_transform_column(scaler, test_predictions)

    recent_window = scaled_close[-window_size:].flatten()
    future_scaled = forecast_future(best_model, recent_window, forecast_steps)
    future_prices = inverse_transform_column(scaler, future_scaled)
    future_dates = pd.date_range(
        start=pd.to_datetime(dataset.iloc[-1, 0]) + pd.Timedelta(days=1),
        periods=forecast_steps,
    )
    forecast_frame = pd.DataFrame({"Date": future_dates, "ForecastClose": future_prices.round(2)})
    forecast_path.parent.mkdir(parents=True, exist_ok=True)
    forecast_frame.to_csv(forecast_path, index=False)

    report_forecast = forecast_frame.assign(Date=forecast_frame["Date"].dt.strftime("%Y-%m-%d"))
    report = {
        "backend": "sequence_regression_baseline",
        "best_model": best_name,
        "validation_rmse_by_model": comparison,
        "test_rmse": float(mean_squared_error(actual_prices, predicted_prices, squared=False)),
        "test_mae": float(mean_absolute_error(actual_prices, predicted_prices)),
        "forecast_steps": forecast_steps,
        "recent_forecast": report_forecast.to_dict(orient="records"),
    }

    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best_model,
            "scaler": scaler,
            "window_size": window_size,
            "best_model_name": best_name,
        },
        bundle_path,
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a stock forecasting project with time-windowed features.")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument("--forecast-steps", type=int, default=5)
    parser.add_argument("--bundle-path", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--forecast-path", type=Path, default=DEFAULT_FORECAST_PATH)
    args = parser.parse_args()

    report = train(
        args.data_path,
        args.window_size,
        args.forecast_steps,
        args.bundle_path,
        args.report_path,
        args.forecast_path,
    )
    print(json.dumps(report, indent=2))
    print(f"Bundle saved to: {args.bundle_path}")
    print(f"Report saved to: {args.report_path}")
    print(f"Forecast saved to: {args.forecast_path}")


if __name__ == "__main__":
    main()
