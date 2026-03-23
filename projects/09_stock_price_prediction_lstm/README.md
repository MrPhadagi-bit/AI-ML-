# Stock Price Prediction with LSTM

Forecast stock closing prices with a simple LSTM model using either a custom CSV or a generated demo time series.

## What is included

- A TensorFlow LSTM training script
- Synthetic demo data generation when no CSV is supplied
- Model saving to `artifacts/stock_lstm.keras`

## Quick start

```bash
pip install -r requirements.txt
python src/train.py --epochs 5
```

To use your own stock data:

```bash
python src/train.py --data-path path/to/stock_prices.csv
```

Expected CSV columns:

- `Date`
- `Close`

If no CSV is provided, the script trains on a deterministic synthetic price series for demonstration purposes.

