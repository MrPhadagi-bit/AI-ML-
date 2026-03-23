from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "sample_data" / "housing_prices.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "housing_model.joblib"
TARGET_COLUMN = "median_house_value"


def build_pipeline(feature_names: list[str]) -> Pipeline:
    numeric_features = list(feature_names)
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        ]
    )

    model = RandomForestRegressor(
        n_estimators=250,
        random_state=42,
        min_samples_leaf=1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def load_dataset(data_path: Path) -> pd.DataFrame:
    dataset = pd.read_csv(data_path)
    if TARGET_COLUMN not in dataset.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in {data_path}")
    return dataset


def evaluate_model(pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = pipeline.predict(x_test)
    return {
        "rmse": float(mean_squared_error(y_test, predictions, squared=False)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }


def train(data_path: Path, model_path: Path) -> dict[str, float]:
    dataset = load_dataset(data_path)
    x = dataset.drop(columns=[TARGET_COLUMN])
    y = dataset[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    pipeline = build_pipeline(list(x.columns))
    pipeline.fit(x_train, y_train)

    metrics = evaluate_model(pipeline, x_test, y_test)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    metrics_path = model_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a housing price regression model.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    metrics = train(args.data_path, args.model_path)
    print("Training complete.")
    print(json.dumps(metrics, indent=2))
    print(f"Model saved to: {args.model_path}")


if __name__ == "__main__":
    main()

