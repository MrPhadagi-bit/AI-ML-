from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "sample_data" / "housing_prices.csv"
DEFAULT_BUNDLE_PATH = PROJECT_ROOT / "artifacts" / "housing_bundle.joblib"
TARGET_COLUMN = "median_house_value"


def build_feature_frame(args: argparse.Namespace, feature_names: list[str]) -> pd.DataFrame:
    if args.sample_index is not None:
        dataset = pd.read_csv(args.data_path)
        row = dataset.drop(columns=[TARGET_COLUMN]).iloc[[args.sample_index]]
        return row[feature_names]

    values = {
        "median_income": args.median_income,
        "house_age": args.house_age,
        "avg_rooms": args.avg_rooms,
        "avg_bedrooms": args.avg_bedrooms,
        "population": args.population,
        "avg_occupancy": args.avg_occupancy,
        "latitude": args.latitude,
        "longitude": args.longitude,
    }
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise ValueError(
            "Provide --sample-index or all feature arguments. Missing: "
            + ", ".join(missing)
        )
    return pd.DataFrame([values])[feature_names]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict house prices with a trained model.")
    parser.add_argument("--bundle-path", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--median-income", type=float, default=None)
    parser.add_argument("--house-age", type=float, default=None)
    parser.add_argument("--avg-rooms", type=float, default=None)
    parser.add_argument("--avg-bedrooms", type=float, default=None)
    parser.add_argument("--population", type=float, default=None)
    parser.add_argument("--avg-occupancy", type=float, default=None)
    parser.add_argument("--latitude", type=float, default=None)
    parser.add_argument("--longitude", type=float, default=None)
    args = parser.parse_args()

    bundle = joblib.load(args.bundle_path)
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    features = build_feature_frame(args, feature_names)
    prediction = float(model.predict(features)[0])

    print(f"Model: {bundle['best_model_name']}")
    print("Features:")
    print(features.to_string(index=False))
    print(f"Predicted median house value: {prediction:.2f}")


if __name__ == "__main__":
    main()
