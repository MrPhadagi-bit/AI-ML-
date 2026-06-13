from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from image_utils import image_file_to_array


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE_PATH = PROJECT_ROOT / "artifacts" / "shape_classifier.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict the class of a shape image.")
    parser.add_argument("--bundle-path", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--image-path", type=Path, required=True)
    args = parser.parse_args()

    bundle = joblib.load(args.bundle_path)
    model = bundle["model"]
    image_size = bundle["image_size"]
    label_names = bundle["label_names"]

    image = image_file_to_array(args.image_path, image_size)
    prediction_index = int(model.predict(image.reshape(1, -1))[0])
    probabilities = model.predict_proba(image.reshape(1, -1))[0]

    print(f"Prediction: {label_names[prediction_index]}")
    for label_name, probability in zip(label_names, probabilities):
        print(f"{label_name}: {probability:.4f}")


if __name__ == "__main__":
    main()
