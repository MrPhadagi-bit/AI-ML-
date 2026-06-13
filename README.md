# AI and Machine Learning Portfolio

This repository is a complete 10-project AI and machine learning portfolio. It covers structured data, natural language processing, computer vision, healthcare classification, time-series forecasting, and object detection.

Each project is designed as a small, runnable workflow with source code, sample data or generated demo data, training scripts, prediction scripts, documented outputs, and an outcome page with visual examples.

## Project Index

| # | Project | Area | Main Outcome |
|---|---------|------|--------------|
| 01 | [Housing Price Prediction](projects/01_housing_price_prediction/) | Regression | Estimates median house value from housing features. |
| 02 | [Movie Review Sentiment Analysis](projects/02_movie_review_sentiment_analysis/) | NLP | Classifies reviews as positive or negative. |
| 03 | [Image Classification CNN](projects/03_image_classification_cnn/) | Computer Vision | Classifies generated shape images. |
| 04 | [Handwritten Digit Recognition MNIST](projects/04_handwritten_digit_recognition_mnist/) | Computer Vision | Recognizes digit images from 0 to 9. |
| 05 | [Spam Email Detection](projects/05_spam_email_detection/) | NLP | Detects spam and ham email text. |
| 06 | [Fake News Detection](projects/06_fake_news_detection/) | NLP | Predicts whether a news article is fake or real. |
| 07 | [Breast Cancer Detection](projects/07_breast_cancer_detection/) | Healthcare ML | Classifies tumor records as benign or malignant. |
| 08 | [Text Summarization NLP](projects/08_text_summarization_nlp/) | NLP | Produces extractive article summaries. |
| 09 | [Stock Price Prediction LSTM](projects/09_stock_price_prediction_lstm/) | Time Series | Forecasts future closing prices from recent history. |
| 10 | [Object Detection YOLO Workflow](projects/10_object_detection_yolo/) | Computer Vision | Detects objects and saves annotated images. |

## Quick Start

Run a project from its own directory:

```bash
cd projects/01_housing_price_prediction
pip install -r requirements.txt
python src/train.py
python src/predict.py --sample-index 0
```

Most projects follow the same pattern:

1. Install dependencies from `requirements.txt`.
2. Run a training or processing script from `src/`.
3. Run a prediction, forecast, summary, or detection script.
4. Review the generated files in the local `artifacts/` folder.
5. Open `OUTCOME.md` to understand the expected result.

## Repository Layout

```text
AI-ML--main/
|-- README.md
|-- docs/
|   |-- PROJECT_CATALOG.md
|   |-- REPOSITORY_GUIDE.md
|-- projects/
|   |-- README.md
|   |-- 01_housing_price_prediction/
|   |-- 02_movie_review_sentiment_analysis/
|   |-- 03_image_classification_cnn/
|   |-- 04_handwritten_digit_recognition_mnist/
|   |-- 05_spam_email_detection/
|   |-- 06_fake_news_detection/
|   |-- 07_breast_cancer_detection/
|   |-- 08_text_summarization_nlp/
|   |-- 09_stock_price_prediction_lstm/
|   |-- 10_object_detection_yolo/
```

## Documentation Standard

Every numbered project includes:

- A detailed `README.md` with setup, commands, inputs, outputs, examples, and next-step plans.
- An `OUTCOME.md` file that explains the final result and includes a visual preview.
- Source code under `src/`.
- A clear artifact list so generated models, reports, predictions, and annotated files are easy to find.

## Portfolio Usage Plan

Use this repository as a learning path or presentation portfolio:

1. Start with Project 01 to understand supervised regression.
2. Move through Projects 02, 05, 06, and 08 for NLP workflows.
3. Use Projects 03, 04, and 10 for image-based tasks.
4. Review Project 07 for a healthcare classification example.
5. Finish with Project 09 for time-series forecasting.

## Notes

Generated `artifacts/` folders are created when scripts run. They are intentionally local outputs so the repository stays lightweight while still documenting exactly what each project produces.
