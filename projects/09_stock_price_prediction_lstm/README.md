# 09. Stock Price Prediction LSTM

This project forecasts future stock closing prices from recent time-series history. The folder name references LSTM-style forecasting, while the implementation focuses on a lightweight sequence-window forecasting workflow that can run locally.

## Objective

Use recent closing prices to predict the next stock closing prices.

## Workflow

1. Load stock price data or generate demo time-series data.
2. Build rolling windows from recent closing prices.
3. Train forecasting models.
4. Save the best forecaster bundle and report.
5. Forecast the next set of closing prices.

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python src/train.py
```

Custom data:

```bash
python src/train.py --data-path path/to/stock_prices.csv --window-size 12 --forecast-steps 5
```

## Forecast

```bash
python src/forecast.py --steps 5
```

## Expected Columns

| Column | Meaning |
|--------|---------|
| `Date` | Trading date. |
| `Close` | Closing price for that date. |

## Outputs

| File | Description |
|------|-------------|
| `artifacts/stock_forecaster.joblib` | Saved forecasting bundle. |
| `artifacts/stock_forecaster_report.json` | Training and evaluation report. |
| `artifacts/next_5_day_forecast.csv` | Forecasted future closing prices. |

## Example Outcome

The forecast script prints the selected model and predicted closing price for each future date. See [OUTCOME.md](OUTCOME.md).

## Improvement Plan

- Add real market data ingestion.
- Add LSTM or GRU deep-learning models.
- Add forecast interval estimates.
- Add plots comparing historical and predicted prices.
