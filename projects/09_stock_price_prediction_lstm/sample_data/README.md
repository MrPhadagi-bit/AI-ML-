# Stock Forecasting Sample Data

This folder is reserved for custom stock price CSV files used by the stock forecasting project. The training script can also generate demo data automatically when no custom file is supplied.

## Required CSV Format

| Column | Required | Description |
|--------|----------|-------------|
| `Date` | Yes | Trading date in a format pandas can parse, such as `2026-01-15`. |
| `Close` | Yes | Closing price for the stock on that date. |

## Example

```csv
Date,Close
2026-01-01,100.25
2026-01-02,101.40
2026-01-03,100.95
```

## How To Use A Custom File

Place your CSV in this folder, then run:

```bash
python src/train.py --data-path sample_data/your_stock_data.csv
python src/forecast.py --data-path sample_data/your_stock_data.csv
```

## Expected Outcome

The training script builds a forecasting bundle from the closing-price sequence and writes:

- `artifacts/stock_forecaster.joblib`
- `artifacts/stock_forecaster_report.json`
- `artifacts/next_5_day_forecast.csv`

## Data Quality Plan

Before using real market data, check for missing dates, duplicate rows, non-numeric closing prices, and large gaps in the time series. Clean data makes the forecast easier to interpret.
