from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE_PATH = PROJECT_ROOT / "artifacts" / "breast_cancer_bundle.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict breast cancer class for a sample row.")
    parser.add_argument("--bundle-path", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--sample-index", type=int, default=0)
    args = parser.parse_args()

    bundle = joblib.load(args.bundle_path)
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data.iloc[[args.sample_index]][bundle["feature_names"]]
    target_names = bundle["target_names"]
    model = bundle["model"]

    prediction_index = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    print(f"Model: {bundle['best_model_name']}")
    print(f"Sample index: {args.sample_index}")
    print(f"Prediction: {target_names[prediction_index]}")
    for label, probability in zip(target_names, probabilities):
        print(f"{label}: {probability:.4f}")


if __name__ == "__main__":
    main()
