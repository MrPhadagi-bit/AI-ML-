# 06. Fake News Detection

This project predicts whether a news article is fake or real using supervised NLP. It combines article headline and body text, converts the text into machine-learning features, trains a classifier, and scores new articles.

## Objective

Build a text classifier that labels news content as `fake` or `real`.

## Workflow

1. Load labeled articles from `sample_data/fake_news.csv`.
2. Combine headline and body fields into model input text.
3. Transform text with TF-IDF features.
4. Train and evaluate classification models.
5. Save a reusable bundle for prediction.

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python src/train.py
```

Use custom data:

```bash
python src/train.py --data-path path/to/fake_news.csv
```

## Predict

```bash
python src/predict.py --headline "Scientists confirm new climate milestone" --body "Researchers published verified findings after a multi-year study."
```

## Outputs

| File | Description |
|------|-------------|
| `artifacts/fake_news_bundle.joblib` | Saved vectorizer and classifier bundle. |
| `artifacts/fake_news_report.json` | Evaluation report and model metrics. |

## Example Outcome

The prediction script prints the selected model, fake or real prediction, and class probabilities when available. See [OUTCOME.md](OUTCOME.md).

## Improvement Plan

- Add larger and more diverse news datasets.
- Add source credibility and publication-date features.
- Add explainability with top weighted terms.
- Add careful validation to reduce bias and overfitting.
