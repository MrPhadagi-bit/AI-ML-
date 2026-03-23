from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from image_utils import LABEL_NAMES, generate_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_BUNDLE_PATH = ARTIFACTS_DIR / "shape_classifier.joblib"
DEFAULT_REPORT_PATH = ARTIFACTS_DIR / "shape_classifier_report.json"


def flatten_images(images: np.ndarray) -> np.ndarray:
    return images.reshape(images.shape[0], -1)


def train(samples_per_class: int, image_size: int, bundle_path: Path, report_path: Path) -> dict[str, object]:
    images, labels = generate_dataset(samples_per_class=samples_per_class, image_size=image_size, seed=42)
    x_train, x_test, y_train, y_test = train_test_split(
        flatten_images(images),
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=12,
        random_state=42,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    report = {
        "backend": "sklearn_mlp",
        "label_names": LABEL_NAMES,
        "samples_per_class": samples_per_class,
        "image_size": image_size,
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=LABEL_NAMES,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }

    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "backend": "sklearn_mlp",
            "image_size": image_size,
            "label_names": LABEL_NAMES,
        },
        bundle_path,
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an image classifier on synthetic shape images.")
    parser.add_argument("--samples-per-class", type=int, default=120)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--bundle-path", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    report = train(args.samples_per_class, args.image_size, args.bundle_path, args.report_path)
    print(json.dumps(report, indent=2))
    print(f"Bundle saved to: {args.bundle_path}")
    print(f"Report saved to: {args.report_path}")


if __name__ == "__main__":
    main()
