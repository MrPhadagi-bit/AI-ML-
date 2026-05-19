# AI and Machine Learning Projects

A portfolio-style collection of practical AI and machine learning projects covering regression, natural language processing, computer vision, healthcare analytics, and time-series forecasting.

The repository is organized so each project can stand on its own: every project folder includes its own README, dependencies, source code, and any sample data or demo utilities needed for quick local testing.

## Highlights

- 10 implemented machine learning projects in one structured workspace
- Project-level setup and usage instructions
- Training, prediction, evaluation, and forecasting scripts
- Sample datasets or demo-data generators where useful
- Supporting documentation for project planning and repository structure

## Project Index

| # | Project | Focus Area | Folder |
| --- | --- | --- | --- |
| 1 | Housing Price Prediction | Regression | [projects/01_housing_price_prediction](projects/01_housing_price_prediction/README.md) |
| 2 | Movie Review Sentiment Analysis | NLP Classification | [projects/02_movie_review_sentiment_analysis](projects/02_movie_review_sentiment_analysis/README.md) |
| 3 | Image Classification with CNN Concepts | Computer Vision | [projects/03_image_classification_cnn](projects/03_image_classification_cnn/README.md) |
| 4 | Handwritten Digit Recognition | Computer Vision | [projects/04_handwritten_digit_recognition_mnist](projects/04_handwritten_digit_recognition_mnist/README.md) |
| 5 | Spam Email Detection | NLP Classification | [projects/05_spam_email_detection](projects/05_spam_email_detection/README.md) |
| 6 | Fake News Detection | NLP Classification | [projects/06_fake_news_detection](projects/06_fake_news_detection/README.md) |
| 7 | Breast Cancer Detection | Healthcare ML | [projects/07_breast_cancer_detection](projects/07_breast_cancer_detection/README.md) |
| 8 | Text Summarization with NLP | NLP Summarization | [projects/08_text_summarization_nlp](projects/08_text_summarization_nlp/README.md) |
| 9 | Stock Price Forecasting | Time Series | [projects/09_stock_price_prediction_lstm](projects/09_stock_price_prediction_lstm/README.md) |
| 10 | Object Detection Workflow | Computer Vision | [projects/10_object_detection_yolo](projects/10_object_detection_yolo/README.md) |

## Quick Start

Clone the repository:

```bash
git clone https://github.com/MrPhadagi-bit/AI-ML-.git
cd AI-ML-
```

Open the project you want to run:

```bash
cd projects/01_housing_price_prediction
```

Install that project's dependencies:

```bash
pip install -r requirements.txt
```

Run the commands listed in the selected project's README. For example:

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

- [Project Catalog](docs/PROJECT_CATALOG.md): expanded descriptions, objectives, skills, and reference sources for each project.
- [Repository Guide](docs/REPOSITORY_GUIDE.md): folder layout, naming conventions, documentation standards, and development workflow.
- [Projects Directory](projects/README.md): quick links to all project folders.

## Notes

- Some projects use local sample datasets for demonstration.
- Some computer vision and time-series projects include demo-data generators for faster testing.
- Results may vary depending on the environment, installed libraries, and available compute resources.
- External repositories and notebooks listed in the documentation are used as reference material for topic selection and learning context; the implementations here are organized as original project work.
