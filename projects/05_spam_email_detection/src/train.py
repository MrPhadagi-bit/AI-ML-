from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "sample_data" / "spam_emails.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "spam_classifier.joblib"


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
            ("model", MultinomialNB()),
        ]
    )


def load_dataset(data_path: Path) -> tuple[pd.Series, pd.Series]:
    dataset = pd.read_csv(data_path)
    required_columns = {"text", "label"}
    if not required_columns.issubset(dataset.columns):
        raise ValueError(f"Expected columns {required_columns} in {data_path}")
    return dataset["text"], dataset["label"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a spam email detector.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--text", type=str, default=None, help="Optional email text to score.")
    args = parser.parse_args()

    x, y = load_dataset(args.data_path)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(classification_report(y_test, predictions, zero_division=0))

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.model_path)
    print(f"Model saved to: {args.model_path}")

    if args.text:
        prediction = pipeline.predict([args.text])[0]
        probabilities = pipeline.predict_proba([args.text])[0]
        print(f"Prediction for custom text: {prediction}")
        print(f"Class probabilities: {probabilities}")


if __name__ == "__main__":
    main()

