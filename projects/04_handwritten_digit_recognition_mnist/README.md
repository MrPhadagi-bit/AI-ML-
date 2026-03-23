# Handwritten Digit Recognition with MNIST

This project has been upgraded into a real handwritten-digit recognition workflow using a built-in digit dataset, a trainable classifier, and a separate prediction path for sample indices or image files.

## What is included

- A training script with model selection
- A prediction script for exported digit images
- A demo image exporter for quick manual tests

## Quick start

```bash
pip install -r requirements.txt
python src/train.py
python src/export_demo_images.py
python src/predict.py --image-path sample_data/demo_digits/3/digit_1.png
```

## Outputs

- `artifacts/digit_classifier.joblib`
- `artifacts/digit_classifier_report.json`
