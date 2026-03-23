from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "breast_cancer_model.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a breast cancer classifier.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    dataset = load_breast_cancer(as_frame=True)
    x = dataset.data
    y = dataset.target

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(classification_report(y_test, predictions, target_names=dataset.target_names, zero_division=0))

    model = pipeline.named_steps["model"]
    coefficients = list(zip(dataset.feature_names, model.coef_[0]))
    strongest_features = sorted(coefficients, key=lambda item: abs(item[1]), reverse=True)[:10]
    print("Top coefficients:")
    for feature_name, weight in strongest_features:
        print(f"  {feature_name}: {weight:.4f}")

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.model_path)
    print(f"Model saved to: {args.model_path}")


if __name__ == "__main__":
    main()
