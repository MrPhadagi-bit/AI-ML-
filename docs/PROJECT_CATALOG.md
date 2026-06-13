# Project Catalog

This catalog explains what each project does, the skill it demonstrates, and the outcome produced after running it.

## 1. Housing Price Prediction

- Area: Regression
- Goal: Predict median house value from structured housing features.
- Skills: Feature preparation, regression modeling, model evaluation, saved inference bundles.
- Outcome: A trained model predicts a numeric housing value for a sample row or custom feature input.
- Output files: `artifacts/housing_bundle.joblib`, `artifacts/training_metrics.json`

## 2. Movie Review Sentiment Analysis

- Area: Natural language processing
- Goal: Classify reviews as positive or negative.
- Skills: Text cleaning, TF-IDF vectorization, logistic regression, classification metrics.
- Outcome: A trained classifier predicts the sentiment of custom review text.
- Output files: `artifacts/sentiment_bundle.joblib`, `artifacts/sentiment_report.json`

## 3. Image Classification

- Area: Computer vision
- Goal: Classify generated shape images.
- Skills: Image generation, image feature extraction, supervised classification, probability outputs.
- Outcome: A trained classifier predicts whether an input image is a circle, square, triangle, or another supported demo class.
- Output files: `artifacts/shape_classifier.joblib`, `artifacts/shape_classifier_report.json`

## 4. Handwritten Digit Recognition

- Area: Computer vision
- Goal: Recognize handwritten digits from 0 to 9.
- Skills: Image normalization, dataset splitting, classifier selection, class probability reporting.
- Outcome: A trained model predicts the digit shown in a sample index or exported image.
- Output files: `artifacts/digit_classifier.joblib`, `artifacts/digit_classifier_report.json`

## 5. Spam Email Detection

- Area: Natural language processing
- Goal: Detect whether email content is spam or ham.
- Skills: Text preprocessing, TF-IDF features, Naive Bayes classification, inference on custom text.
- Outcome: A trained model labels email text as spam or ham.
- Output files: `artifacts/spam_bundle.joblib`, `artifacts/spam_report.json`

## 6. Fake News Detection

- Area: Natural language processing
- Goal: Classify news text as fake or real.
- Skills: Headline and body feature preparation, vectorization, binary classification, probability reporting.
- Outcome: A trained model predicts whether a supplied article is fake or real.
- Output files: `artifacts/fake_news_bundle.joblib`, `artifacts/fake_news_report.json`

## 7. Breast Cancer Detection

- Area: Healthcare machine learning
- Goal: Classify diagnostic records as benign or malignant.
- Skills: Medical-style tabular classification, model evaluation, ROC AUC, feature importance.
- Outcome: A trained classifier predicts the diagnosis class for a sample record.
- Output files: `artifacts/breast_cancer_bundle.joblib`, `artifacts/breast_cancer_report.json`

## 8. Text Summarization

- Area: Natural language processing
- Goal: Generate shorter summaries from longer text.
- Skills: Sentence scoring, extractive summarization, lightweight evaluation.
- Outcome: The script prints a concise summary for sample or custom text.
- Output files: `artifacts/summarization_evaluation.json` when evaluation is enabled

## 9. Stock Price Forecasting

- Area: Time series forecasting
- Goal: Predict future closing prices from recent price history.
- Skills: Sequence windows, scaling, validation, iterative forecasting.
- Outcome: A trained forecaster predicts the next stock closing prices by date.
- Output files: `artifacts/stock_forecaster.joblib`, `artifacts/stock_forecaster_report.json`, `artifacts/next_5_day_forecast.csv`

## 10. Object Detection Workflow

- Area: Computer vision
- Goal: Detect and localize objects in demo images.
- Skills: Demo scene generation, connected-component detection, bounding boxes, JSON reporting.
- Outcome: The detector writes annotated images and detection reports with labels, confidence values, and bounding boxes.
- Output files: annotated images and `*_detections.json` files in `artifacts/predictions/`

## Learning Path

Start with tabular and text projects, then move into vision and forecasting:

1. Housing Price Prediction
2. Movie Review Sentiment Analysis
3. Spam Email Detection
4. Fake News Detection
5. Breast Cancer Detection
6. Text Summarization
7. Handwritten Digit Recognition
8. Image Classification
9. Stock Price Forecasting
10. Object Detection Workflow
