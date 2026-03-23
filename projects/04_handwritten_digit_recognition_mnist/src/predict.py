from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from digit_utils import image_path_to_digit_vector, load_digit_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE_PATH = PROJECT_ROOT / "artifacts" / "digit_classifier.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict handwritten digits from an image or sample index.")
    parser.add_argument("--bundle-path", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--image-path", type=Path, default=None)
    args = parser.parse_args()

    if args.sample_index is None and args.image_path is None:
        raise ValueError("Provide --sample-index or --image-path.")

    bundle = joblib.load(args.bundle_path)
    model = bundle["model"]

    if args.image_path is not None:
        features = image_path_to_digit_vector(args.image_path)
    else:
        images, _ = load_digit_dataset()
        features = images[args.sample_index].reshape(1, -1)

    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    print(f"Model: {bundle['best_model_name']}")
    print(f"Prediction: {prediction}")
    for label_name, probability in zip(bundle["label_names"], probabilities):
        print(f"{label_name}: {probability:.4f}")


if __name__ == "__main__":
    main()
