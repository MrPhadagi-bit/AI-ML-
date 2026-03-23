# Stock Price Prediction with LSTM

This project now includes a real time-series forecasting workflow with training, validation, bundle saving, and multi-step forecasting. The current local backend uses sequence-based regression so the project runs on this machine today, while keeping the door open for a future LSTM-specific backend.

## What is included

- Deterministic demo stock-price generation
- A training script with model selection over time-window features
- A separate forecasting script for future dates
- Saved reports and forecast CSV outputs

## Quick start

```bash
pip install -r requirements.txt
python src/train.py
python src/forecast.py
```

To use your own stock data:

```bash
python src/train.py --data-path path/to/stock_prices.csv
```
