# 07. Breast Cancer Detection

This project classifies breast cancer records as benign or malignant using a supervised healthcare machine-learning workflow. It is intended for learning and portfolio demonstration, not medical diagnosis.

## Objective

Train a classifier that predicts whether a tumor record is benign or malignant from numeric diagnostic measurements.

## Workflow

1. Load the breast cancer dataset.
2. Split features and target labels.
3. Train candidate classifiers.
4. Save the best model bundle.
5. Predict the class of a sample record.

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python src/train.py
```

## Predict

```bash
python src/predict.py --sample-index 0
```

## Outputs

| File | Description |
|------|-------------|
| `artifacts/breast_cancer_bundle.joblib` | Saved classifier bundle. |
| `artifacts/breast_cancer_report.json` | Model metrics and evaluation report. |

## Example Outcome

The prediction script prints the selected model, sample index, predicted class, and class probabilities. See [OUTCOME.md](OUTCOME.md).

## Important Note

This project is educational. It should not be used to make real medical decisions without clinical validation, governance, and expert review.

## Improvement Plan

- Add stronger validation and calibration.
- Add sensitivity and specificity reporting.
- Add feature importance explanations.
- Add a clinical-risk disclaimer to any user-facing interface.
