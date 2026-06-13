from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier

from digit_utils import load_digit_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_BUNDLE_PATH = ARTIFACTS_DIR / "digit_classifier.joblib"
DEFAULT_REPORT_PATH = ARTIFACTS_DIR / "digit_classifier_report.json"


def build_candidates() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(max_iter=3000, random_state=42),
        "mlp_classifier": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=300,
            early_stopping=True,
            n_iter_no_change=12,
            random_state=42,
        ),
    }


def train(bundle_path: Path, report_path: Path) -> dict[str, object]:
    images, labels = load_digit_dataset()
    features = images.reshape(images.shape[0], -1)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    comparison: dict[str, float] = {}
    best_name = ""
    best_score = float("-inf")
    best_model = None

    for name, candidate in build_candidates().items():
        cv_scores = cross_val_score(candidate, x_train, y_train, cv=3, scoring="accuracy")
        mean_accuracy = float(cv_scores.mean())
        comparison[name] = mean_accuracy
        if mean_accuracy > best_score:
            best_score = mean_accuracy
            best_name = name
            best_model = candidate

    if best_model is None:
        raise RuntimeError("No digit recognition models were available.")

    best_model.fit(x_train, y_train)
    predictions = best_model.predict(x_test)

    report = {
        "best_model": best_name,
        "cv_accuracy_by_model": comparison,
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "classification_report": classification_report(
            y_test,
            predictions,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }

    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best_model,
            "best_model_name": best_name,
            "label_names": [str(index) for index in range(10)],
        },
        bundle_path,
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a handwritten digit recognizer.")
    parser.add_argument("--bundle-path", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    report = train(args.bundle_path, args.report_path)
    print(json.dumps(report, indent=2))
    print(f"Bundle saved to: {args.bundle_path}")
    print(f"Report saved to: {args.report_path}")


if __name__ == "__main__":
    main()
