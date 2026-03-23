# AI and Machine Learning Projects

This repository is a curated collection of practical machine learning projects built across regression, natural language processing, computer vision, healthcare analytics, and time-series forecasting. It is structured as a portfolio-style workspace where each project lives in its own folder with clear documentation, runnable scripts, and project-specific requirements.

## What This Repository Includes

- 10 machine learning projects in one organized repository
- Project-level `README.md` files with usage instructions
- Runnable training, prediction, evaluation, and forecasting scripts
- Sample data or demo generators for quick local testing
- Supporting documentation for repository structure and project planning

## Projects

| # | Project | Area | Status | Folder |
| --- | --- | --- | --- | --- |
| 1 | Housing Price Prediction | Regression | Implemented | [projects/01_housing_price_prediction](projects/01_housing_price_prediction/README.md) |
| 2 | Movie Review Sentiment Analysis | NLP Classification | Implemented | [projects/02_movie_review_sentiment_analysis](projects/02_movie_review_sentiment_analysis/README.md) |
| 3 | Image Classification with CNN Concepts | Computer Vision | Implemented | [projects/03_image_classification_cnn](projects/03_image_classification_cnn/README.md) |
| 4 | Handwritten Digit Recognition | Computer Vision | Implemented | [projects/04_handwritten_digit_recognition_mnist](projects/04_handwritten_digit_recognition_mnist/README.md) |
| 5 | Spam Email Detection | NLP Classification | Implemented | [projects/05_spam_email_detection](projects/05_spam_email_detection/README.md) |
| 6 | Fake News Detection | NLP Classification | Implemented | [projects/06_fake_news_detection](projects/06_fake_news_detection/README.md) |
| 7 | Breast Cancer Detection | Healthcare ML | Implemented | [projects/07_breast_cancer_detection](projects/07_breast_cancer_detection/README.md) |
| 8 | Text Summarization with NLP | NLP Summarization | Implemented | [projects/08_text_summarization_nlp](projects/08_text_summarization_nlp/README.md) |
| 9 | Stock Price Forecasting | Time Series | Implemented | [projects/09_stock_price_prediction_lstm](projects/09_stock_price_prediction_lstm/README.md) |
| 10 | Object Detection Workflow | Computer Vision | Implemented | [projects/10_object_detection_yolo](projects/10_object_detection_yolo/README.md) |

## Quick Start

1. Clone the repository:

```bash
git clone https://github.com/MrPhadagi-bit/AI-ML-.git
cd AI-ML-
```

2. Open the project you want to run:

```bash
cd projects/01_housing_price_prediction
```

3. Install the dependencies for that project:

```bash
pip install -r requirements.txt
```

4. Run the project script shown in that project's README.

Example:

```bash
python src/train.py
python src/predict.py --sample-index 0
```

## Repository Structure

```text
AI-ML-/
|-- README.md
|-- .gitignore
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

- [Project Catalog](docs/PROJECT_CATALOG.md)
- [Repository Guide](docs/REPOSITORY_GUIDE.md)
- [Projects Directory](projects/README.md)

## Reference Sources

The original project ideas were inspired by public repositories and notebooks. Those links are preserved in the documentation and were used as reference points for topic selection and structure, while the implementations in this repository are organized as original project work.

## Notes

- Some projects use local sample datasets for demonstration.
- Some computer vision and time-series projects include demo-data generators so they can be tested quickly.
- Project-level results can vary depending on the available environment and installed libraries.
