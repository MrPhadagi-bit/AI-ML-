# 01. Housing Price Prediction

This project predicts median house value from structured housing features. It demonstrates a complete regression workflow: loading tabular data, training multiple models, selecting the best model, saving a reusable bundle, and running predictions on sample or custom feature values.

## Objective

Build a regression model that estimates `median_house_value` from neighborhood and property-level features such as income, rooms, population, occupancy, latitude, longitude, and house age.

## Workflow

1. Load the housing dataset from `sample_data/housing_prices.csv`.
2. Split features from the target column.
3. Train candidate regression models.
4. Compare metrics and save the best pipeline.
5. Use `predict.py` to score a sample row or custom values.

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python src/train.py
```

Use another CSV with the same schema:

```bash
python src/train.py --data-path path/to/housing.csv
```

## Predict

Predict from a sample row:

```bash
python src/predict.py --sample-index 0
```

Predict from custom values:

```bash
python src/predict.py --median-income 5.2 --house-age 18 --avg-rooms 6.4 --avg-bedrooms 1.1 --population 920 --avg-occupancy 2.7 --latitude 34.05 --longitude -118.25
```

## Expected Columns

| Column | Meaning |
|--------|---------|
| `median_income` | Median income for the area. |
| `house_age` | Median age of homes. |
| `avg_rooms` | Average rooms per household. |
| `avg_bedrooms` | Average bedrooms per household. |
| `population` | Area population. |
| `avg_occupancy` | Average household occupancy. |
| `latitude` | Location latitude. |
| `longitude` | Location longitude. |
| `median_house_value` | Target value to predict. |

## Outputs

| File | Description |
|------|-------------|
| `artifacts/housing_bundle.joblib` | Saved model bundle used by `predict.py`. |
| `artifacts/training_metrics.json` | Regression metrics and model comparison details. |

## Example Outcome

The prediction script prints the chosen model, the input feature row, and the predicted median house value. See [OUTCOME.md](OUTCOME.md) for the visual outcome page.

## Improvement Plan

- Add cross-validation for more stable model comparison.
- Add feature importance charts for explainability.
- Add residual plots to inspect where predictions are weak.
- Expand the sample dataset with more realistic regional variation.
