# Housing Price Prediction

Predict housing prices from structured tabular features using a regression model.

## What is included

- A starter regression training pipeline
- A small sample dataset for local testing
- Model saving to an `artifacts/` directory

## Project structure

```text
01_housing_price_prediction/
|-- README.md
|-- requirements.txt
|-- sample_data/
|   |-- housing_prices.csv
|-- src/
|   |-- train.py
```

## Quick start

```bash
pip install -r requirements.txt
python src/train.py
```

To use your own dataset:

```bash
python src/train.py --data-path path/to/housing.csv
```

Expected columns:

- `median_income`
- `house_age`
- `avg_rooms`
- `avg_bedrooms`
- `population`
- `avg_occupancy`
- `latitude`
- `longitude`
- `median_house_value`

## Output

The training script prints regression metrics and saves a trained pipeline to `artifacts/housing_model.joblib`.

