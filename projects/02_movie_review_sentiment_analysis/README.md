# 02. Movie Review Sentiment Analysis

This project classifies movie reviews as positive or negative using a TF-IDF text pipeline and a supervised classifier. It is a practical introduction to NLP preprocessing, text vectorization, model training, evaluation, and inference.

## Objective

Train a model that reads review text and predicts the sentiment label.

## Workflow

1. Load labeled reviews from `sample_data/movie_reviews.csv`.
2. Convert raw text into TF-IDF features.
3. Train and compare classification models.
4. Save the best model bundle.
5. Predict sentiment for new review text.

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python src/train.py
```

Use another labeled review dataset:

```bash
python src/train.py --data-path path/to/movie_reviews.csv
```

## Predict

```bash
python src/predict.py --text "This movie was surprisingly good and well acted."
```

## Inputs

The training CSV should contain review text and sentiment labels. The prediction script accepts a single text string through `--text`.

## Outputs

| File | Description |
|------|-------------|
| `artifacts/sentiment_bundle.joblib` | Saved text vectorizer and classifier bundle. |
| `artifacts/sentiment_report.json` | Evaluation report with model results. |

## Example Outcome

The prediction script prints the selected model, the sentiment label, and probability scores when available. See [OUTCOME.md](OUTCOME.md).

## Improvement Plan

- Add more balanced movie-review examples.
- Add confusion matrix visualization.
- Compare TF-IDF models with transformer embeddings.
- Add a small web form for interactive review scoring.
