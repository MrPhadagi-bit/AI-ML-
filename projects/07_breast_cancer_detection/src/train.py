from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_BUNDLE_PATH = ARTIFACTS_DIR / "breast_cancer_bundle.joblib"
DEFAULT_REPORT_PATH = ARTIFACTS_DIR / "breast_cancer_report.json"


def build_candidates() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=3000, random_state=42)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", RandomForestClassifier(n_estimators=300, random_state=42)),
            ]
        ),
    }


def extract_feature_scores(model: Pipeline, feature_names: list[str]) -> dict[str, float]:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "coef_"):
        values = estimator.coef_[0]
    elif hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    else:
        return {}
    return {
        feature_name: float(score)
        for feature_name, score in sorted(
            zip(feature_names, values),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
    }


def train(bundle_path: Path, report_path: Path) -> dict[str, object]:
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

    comparison: dict[str, float] = {}
    best_name = ""
    best_score = float("-inf")
    best_pipeline: Pipeline | None = None

    for name, candidate in build_candidates().items():
        cv_scores = cross_val_score(candidate, x_train, y_train, cv=5, scoring="accuracy")
        mean_accuracy = float(cv_scores.mean())
        comparison[name] = mean_accuracy
        if mean_accuracy > best_score:
            best_score = mean_accuracy
            best_name = name
            best_pipeline = candidate

    if best_pipeline is None:
        raise RuntimeError("No breast cancer classifiers were available.")

    best_pipeline.fit(x_train, y_train)
    predictions = best_pipeline.predict(x_test)
    probabilities = best_pipeline.predict_proba(x_test)[:, 1]

    report = {
        "best_model": best_name,
        "cv_accuracy_by_model": comparison,
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=list(dataset.target_names),
            zero_division=0,
            output_dict=True,
        ),
        "feature_scores": extract_feature_scores(best_pipeline, list(dataset.feature_names)),
    }

    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best_pipeline,
            "feature_names": list(dataset.feature_names),
            "target_names": list(dataset.target_names),
            "best_model_name": best_name,
        },
        bundle_path,
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a breast cancer classification workflow.")
    parser.add_argument("--bundle-path", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    report = train(args.bundle_path, args.report_path)
    print(json.dumps(report, indent=2))
    print(f"Bundle saved to: {args.bundle_path}")
    print(f"Report saved to: {args.report_path}")


if __name__ == "__main__":
    main()
