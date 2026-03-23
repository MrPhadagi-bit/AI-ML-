# Fake News Detection

Classify text as likely fake or real news using TF-IDF features and logistic regression.

## What is included

- A small sample fake-news dataset
- A starter training script
- Optional prediction for custom headline and body text

## Quick start

```bash
pip install -r requirements.txt
python src/train.py
python src/predict.py --headline "Scientists discover water on Mars" --body "Researchers published new rover findings today."
```

## Outputs

- `artifacts/fake_news_bundle.joblib`
- `artifacts/fake_news_report.json`
