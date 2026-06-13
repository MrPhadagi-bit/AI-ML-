# 03. Image Classification CNN

This project trains an image classifier on generated shape images. It demonstrates the computer-vision workflow of creating demo images, converting pixels into features, training a classifier, and predicting the class of a new image.

## Objective

Classify simple PNG images into shape categories such as circles, rectangles, and triangles.

## Workflow

1. Generate synthetic training images in memory.
2. Train an image classification model.
3. Save the model bundle and training report.
4. Generate demo images for prediction tests.
5. Predict the class of a selected image file.

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python src/train.py
```

Optional training controls:

```bash
python src/train.py --samples-per-class 200 --image-size 32
```

## Generate Demo Images

```bash
python src/generate_demo_images.py
```

## Predict

```bash
python src/predict.py --image-path sample_data/demo_images/circle_0.png
```

## Outputs

| File | Description |
|------|-------------|
| `artifacts/shape_classifier.joblib` | Saved image classifier bundle. |
| `artifacts/shape_classifier_report.json` | Training and evaluation report. |
| `sample_data/demo_images/` | Generated images for inference examples. |

## Example Outcome

The prediction script prints the predicted shape label and class probabilities. See [OUTCOME.md](OUTCOME.md).

## Improvement Plan

- Add a deeper CNN implementation.
- Add augmentation such as rotation, brightness, and noise.
- Save a confusion matrix image.
- Add support for user-uploaded images.
