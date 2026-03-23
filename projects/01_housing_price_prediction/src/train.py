from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "sample_data" / "housing_prices.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_BUNDLE_PATH = ARTIFACTS_DIR / "housing_bundle.joblib"
DEFAULT_METRICS_PATH = ARTIFACTS_DIR / "training_metrics.json"
TARGET_COLUMN = "median_house_value"


def build_preprocessor(feature_names: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_names,
            )
        ]
    )


def build_candidates(feature_names: list[str]) -> dict[str, Pipeline]:
    preprocessor = build_preprocessor(feature_names)
    return {
        "linear_regression": Pipeline(
            steps=[("preprocessor", preprocessor), ("model", LinearRegression())]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", GradientBoostingRegressor(random_state=42)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=120,
                        random_state=42,
                        min_samples_leaf=1,
                    ),
                ),
            ]
        ),
    }


def load_dataset(data_path: Path) -> pd.DataFrame:
    dataset = pd.read_csv(data_path)
    if TARGET_COLUMN not in dataset.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in {data_path}")
    return dataset


def evaluate_regressor(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = model.predict(x_test)
    return {
        "rmse": float(mean_squared_error(y_test, predictions, squared=False)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }


def extract_feature_scores(model: Pipeline, feature_names: list[str]) -> dict[str, float]:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        scores = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        scores = estimator.coef_
    else:
        return {}
    return {
        feature_name: float(score)
        for feature_name, score in sorted(
            zip(feature_names, scores),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
    }


def train(data_path: Path, bundle_path: Path, metrics_path: Path) -> dict[str, object]:
    dataset = load_dataset(data_path)
    x = dataset.drop(columns=[TARGET_COLUMN])
    y = dataset[TARGET_COLUMN]
    feature_names = list(x.columns)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    comparison: dict[str, float] = {}
    best_name = ""
    best_cv_rmse = float("inf")
    best_model: Pipeline | None = None

    for name, candidate in build_candidates(feature_names).items():
        cv_scores = cross_val_score(
            candidate,
            x_train,
            y_train,
            cv=3,
            scoring="neg_root_mean_squared_error",
        )
        mean_cv_rmse = float(-cv_scores.mean())
        comparison[name] = mean_cv_rmse
        if mean_cv_rmse < best_cv_rmse:
            best_name = name
            best_cv_rmse = mean_cv_rmse
            best_model = candidate

    if best_model is None:
        raise RuntimeError("No model candidates were available for training.")

    best_model.fit(x_train, y_train)
    test_metrics = evaluate_regressor(best_model, x_test, y_test)
    feature_scores = extract_feature_scores(best_model, feature_names)

    bundle = {
        "model": best_model,
        "feature_names": feature_names,
        "target_column": TARGET_COLUMN,
        "best_model_name": best_name,
    }
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, bundle_path)

    metrics = {
        "best_model": best_name,
        "cv_rmse_by_model": comparison,
        "test_metrics": test_metrics,
        "feature_scores": feature_scores,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a housing price regression workflow.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--bundle-path", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    args = parser.parse_args()

    metrics = train(args.data_path, args.bundle_path, args.metrics_path)
    print("Training complete.")
    print(json.dumps(metrics, indent=2))
    print(f"Bundle saved to: {args.bundle_path}")
    print(f"Metrics saved to: {args.metrics_path}")


if __name__ == "__main__":
    main()
