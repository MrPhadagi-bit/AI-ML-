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
```

To score your own review text:

```bash
python src/train.py --text "This movie was surprisingly good and well acted."
```

The script saves the trained pipeline to `artifacts/sentiment_model.joblib`.

