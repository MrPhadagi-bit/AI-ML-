# Projects Directory

This folder contains the 10 complete AI and machine learning projects in the portfolio. Each project is independent, so you can open one folder, install its dependencies, run the scripts, and inspect its generated outputs.

## Project Map

| # | Folder | Skill Practiced | Outcome File |
|---|--------|-----------------|--------------|
| 01 | `01_housing_price_prediction` | Regression and tabular ML | [OUTCOME.md](01_housing_price_prediction/OUTCOME.md) |
| 02 | `02_movie_review_sentiment_analysis` | Sentiment classification | [OUTCOME.md](02_movie_review_sentiment_analysis/OUTCOME.md) |
| 03 | `03_image_classification_cnn` | Image classification | [OUTCOME.md](03_image_classification_cnn/OUTCOME.md) |
| 04 | `04_handwritten_digit_recognition_mnist` | Digit recognition | [OUTCOME.md](04_handwritten_digit_recognition_mnist/OUTCOME.md) |
| 05 | `05_spam_email_detection` | Email text classification | [OUTCOME.md](05_spam_email_detection/OUTCOME.md) |
| 06 | `06_fake_news_detection` | News reliability classification | [OUTCOME.md](06_fake_news_detection/OUTCOME.md) |
| 07 | `07_breast_cancer_detection` | Healthcare classification | [OUTCOME.md](07_breast_cancer_detection/OUTCOME.md) |
| 08 | `08_text_summarization_nlp` | Extractive summarization | [OUTCOME.md](08_text_summarization_nlp/OUTCOME.md) |
| 09 | `09_stock_price_prediction_lstm` | Time-series forecasting | [OUTCOME.md](09_stock_price_prediction_lstm/OUTCOME.md) |
| 10 | `10_object_detection_yolo` | Object detection | [OUTCOME.md](10_object_detection_yolo/OUTCOME.md) |

## Recommended Run Order

1. Run the projects in numeric order if you want a guided learning path.
2. Run Projects 02, 05, 06, and 08 together if you want to focus on NLP.
3. Run Projects 03, 04, and 10 together if you want to focus on computer vision.
4. Run Project 09 after the tabular projects, because forecasting uses a different train/test style.

## Common Workflow

```bash
cd projects/<project-folder>
pip install -r requirements.txt
python src/train.py
```

Some projects use a different second command:

- `predict.py` for classification and regression projects.
- `summarize.py` for text summarization.
- `forecast.py` for stock forecasting.
- `detect.py` for object detection.

## What To Review

After running a project, check:

- `README.md` for the full explanation.
- `OUTCOME.md` for a visual result summary.
- `artifacts/` for generated models, reports, summaries, forecasts, or annotated images.
