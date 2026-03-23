from __future__ import annotations

import argparse
from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE_PATH = PROJECT_ROOT / "artifacts" / "fake_news_bundle.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict whether an article is fake or real.")
    parser.add_argument("--bundle-path", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--headline", required=True)
    parser.add_argument("--body", default="")
    args = parser.parse_args()

    bundle = joblib.load(args.bundle_path)
    model = bundle["model"]
    article = f"{args.headline} {args.body}".strip()
    prediction = model.predict([article])[0]
    print(f"Model: {bundle['best_model_name']}")
    print(f"Prediction: {prediction}")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([article])[0]
        labels = model.named_steps["model"].classes_
        for label, probability in zip(labels, probabilities):
            print(f"{label}: {probability:.4f}")


if __name__ == "__main__":
    main()
