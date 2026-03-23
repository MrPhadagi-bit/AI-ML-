from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "sample_data" / "fake_news.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_BUNDLE_PATH = ARTIFACTS_DIR / "fake_news_bundle.joblib"
DEFAULT_REPORT_PATH = ARTIFACTS_DIR / "fake_news_report.json"


def build_candidates() -> dict[str, Pipeline]:
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("vectorizer", vectorizer),
                ("model", LogisticRegression(max_iter=2000, random_state=42)),
            ]
        ),
        "complement_nb": Pipeline(
            steps=[("vectorizer", vectorizer), ("model", ComplementNB(alpha=0.5))]
        ),
    }


def load_dataset(data_path: Path) -> tuple[pd.Series, pd.Series]:
    dataset = pd.read_csv(data_path)
    required_columns = {"title", "text", "label"}
    if not required_columns.issubset(dataset.columns):
        raise ValueError(f"Expected columns {required_columns} in {data_path}")
    combined_text = dataset["title"].fillna("") + " " + dataset["text"].fillna("")
    return combined_text, dataset["label"]


def train(data_path: Path, bundle_path: Path, report_path: Path) -> dict[str, object]:
    x, y = load_dataset(data_path)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    comparison: dict[str, float] = {}
    best_name = ""
    best_score = float("-inf")
    best_pipeline: Pipeline | None = None

    for name, candidate in build_candidates().items():
        cv_scores = cross_val_score(candidate, x_train, y_train, cv=3, scoring="accuracy")
        mean_accuracy = float(cv_scores.mean())
        comparison[name] = mean_accuracy
        if mean_accuracy > best_score:
            best_score = mean_accuracy
            best_name = name
            best_pipeline = candidate

    if best_pipeline is None:
        raise RuntimeError("No fake news classifiers were available.")

    best_pipeline.fit(x_train, y_train)
    predictions = best_pipeline.predict(x_test)
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
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=["fake", "real"]).tolist(),
    }

    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best_pipeline,
            "best_model_name": best_name,
            "labels": ["fake", "real"],
        },
        bundle_path,
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a fake news detection workflow.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--bundle-path", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    report = train(args.data_path, args.bundle_path, args.report_path)
    print(json.dumps(report, indent=2))
    print(f"Bundle saved to: {args.bundle_path}")
    print(f"Report saved to: {args.report_path}")


if __name__ == "__main__":
    main()
