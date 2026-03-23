# Image Classification with CNN

This project is now a real local image-classification workflow built around synthetic shape images. It trains a usable classifier end to end with the libraries already available on this machine, while still leaving room to swap in a deeper CNN backend later if you want.

## What is included

- A dataset generator for labeled shape images
- A training script that fits a real classifier and saves a model bundle
- A prediction script for PNG images
- A demo image generator for quick inference tests

## Quick start

```bash
pip install -r requirements.txt
python src/train.py
python src/generate_demo_images.py
python src/predict.py --image-path sample_data/demo_images/circle/circle_1.png
```

## Outputs

- `artifacts/shape_classifier.joblib`
- `artifacts/shape_classifier_report.json`
