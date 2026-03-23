from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "sample_data" / "movie_reviews.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "sentiment_model.joblib"


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=1,
                ),
            ),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def load_dataset(data_path: Path) -> tuple[pd.Series, pd.Series]:
    dataset = pd.read_csv(data_path)
    required_columns = {"review", "sentiment"}
    if not required_columns.issubset(dataset.columns):
        raise ValueError(f"Expected columns {required_columns} in {data_path}")
    return dataset["review"], dataset["sentiment"]


def train(data_path: Path, model_path: Path, review_text: str | None) -> None:
    x, y = load_dataset(data_path)
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

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Model saved to: {model_path}")

    if review_text:
        predicted_label = pipeline.predict([review_text])[0]
        predicted_probabilities = pipeline.predict_proba([review_text])[0]
        print(f"Prediction for custom text: {predicted_label}")
        print(f"Class probabilities: {predicted_probabilities}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a movie review sentiment model.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--text", type=str, default=None, help="Optional review text to score.")
    args = parser.parse_args()

    train(args.data_path, args.model_path, args.text)


if __name__ == "__main__":
    main()

