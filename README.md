# AI / ML Project Portfolio

This repository is a hands-on portfolio of artificial intelligence and machine learning projects. It brings together practical examples across regression, natural language processing, computer vision, healthcare analytics, and time-series forecasting.

Each project is organized in its own folder with a dedicated README, dependencies, source code, and local testing resources where needed.

## What You Will Find Here

- End-to-end machine learning workflows
- Clean project folders with focused documentation
- Training, prediction, evaluation, and forecasting scripts
- Sample datasets or demo-data generators for quick testing
- Projects that demonstrate both classic ML and deep learning concepts

## Skills Covered

| Area | Example Projects |
| --- | --- |
| Regression | Housing price prediction |
| NLP classification | Sentiment analysis, spam detection, fake news detection |
| Computer vision | Image classification, digit recognition, object detection workflow |
| Healthcare ML | Breast cancer detection |
| NLP summarization | Text summarization workflow |
| Time series | Stock price forecasting |

## Projects

| # | Project | Main Topic | Open |
| --- | --- | --- | --- |
| 1 | Housing Price Prediction | Regression | [View project](projects/01_housing_price_prediction/README.md) |
| 2 | Movie Review Sentiment Analysis | NLP Classification | [View project](projects/02_movie_review_sentiment_analysis/README.md) |
| 3 | Image Classification with CNN Concepts | Computer Vision | [View project](projects/03_image_classification_cnn/README.md) |
| 4 | Handwritten Digit Recognition | Computer Vision | [View project](projects/04_handwritten_digit_recognition_mnist/README.md) |
| 5 | Spam Email Detection | NLP Classification | [View project](projects/05_spam_email_detection/README.md) |
| 6 | Fake News Detection | NLP Classification | [View project](projects/06_fake_news_detection/README.md) |
| 7 | Breast Cancer Detection | Healthcare ML | [View project](projects/07_breast_cancer_detection/README.md) |
| 8 | Text Summarization with NLP | NLP Summarization | [View project](projects/08_text_summarization_nlp/README.md) |
| 9 | Stock Price Forecasting | Time Series | [View project](projects/09_stock_price_prediction_lstm/README.md) |
| 10 | Object Detection Workflow | Computer Vision | [View project](projects/10_object_detection_yolo/README.md) |

## Quick Start

Clone the repository:

```bash
git clone https://github.com/MrPhadagi-bit/AI-ML-.git
cd AI-ML-
```

Choose a project:

```bash
cd projects/01_housing_price_prediction
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Run the commands shown in that project's README. Example:

```bash
python src/train.py
python src/predict.py --sample-index 0
```

## Repository Layout

```text
AI-ML-/
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

## Documentation

- [Project Catalog](docs/PROJECT_CATALOG.md) - detailed descriptions, objectives, skills, deliverables, and references for each project.
- [Repository Guide](docs/REPOSITORY_GUIDE.md) - structure, naming conventions, documentation standards, and development workflow.
- [Projects Directory](projects/README.md) - quick links to all project folders.

## Notes

- Some projects use sample datasets so they can run locally without large downloads.
- Some deep learning and time-series projects include demo-data generation utilities.
- Results may vary depending on installed libraries, environment, and compute resources.
- External links in the documentation are used as learning references; this repository is structured as original portfolio work.
