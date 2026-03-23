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
```

To score a custom article:

```bash
python src/train.py --headline "Scientists discover water on Mars" --body "Researchers published new rover findings today."
```

The script saves the trained model pipeline to `artifacts/fake_news_model.joblib`.

