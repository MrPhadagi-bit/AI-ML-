# Movie Review Sentiment Analysis

Classify movie reviews as positive or negative using a TF-IDF plus logistic regression pipeline.

## What is included

- A small sample dataset for quick local training
- A starter NLP classification pipeline
- Optional prediction for custom text input

## Quick start

```bash
pip install -r requirements.txt
python src/train.py
python src/predict.py --text "This movie was surprisingly good and well acted."
```

## Outputs

- `artifacts/sentiment_bundle.joblib`
- `artifacts/sentiment_report.json`
