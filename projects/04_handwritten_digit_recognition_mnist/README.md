# 04. Handwritten Digit Recognition MNIST

This project recognizes handwritten digits from image data. It uses a classic digit-recognition workflow: train a classifier, export demo digit images, and predict either a built-in sample index or a supplied image file.

## Objective

Classify handwritten digit images into one of ten classes: `0` through `9`.

## Workflow

1. Load digit image data.
2. Train candidate classifiers.
3. Save the best model bundle.
4. Export demo digit images for testing.
5. Predict a digit from a sample index or image path.

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python src/train.py
```

## Export Demo Images

```bash
python src/export_demo_images.py
```

## Predict

Predict from a sample index:

```bash
python src/predict.py --sample-index 0
```

Predict from an image:

```bash
python src/predict.py --image-path sample_data/demo_digits/digit_0_0.png
```

## Outputs

| File | Description |
|------|-------------|
| `artifacts/digit_classifier.joblib` | Saved digit classifier bundle. |
| `artifacts/digit_classifier_report.json` | Model metrics and evaluation summary. |
| `sample_data/demo_digits/` | Exported digit images for demos. |

## Example Outcome

The prediction script prints the model, predicted digit, and class probabilities. See [OUTCOME.md](OUTCOME.md).

## Improvement Plan

- Add a CNN-based model for stronger image performance.
- Add a confusion matrix by digit class.
- Add a notebook showing misclassified examples.
- Add a simple drawing canvas for live predictions.
