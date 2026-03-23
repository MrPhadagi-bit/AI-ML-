# Project Catalog

This document expands on the machine learning roadmap for this repository. It outlines what each project is about, the main skills involved, and the expected outcome of each build.

## 1. Predicting Housing Prices

- Category: Regression
- Objective: Predict housing prices based on structured housing features.
- Skills: Exploratory data analysis, feature engineering, regression modeling, evaluation.
- Typical Tools: Python, Pandas, NumPy, scikit-learn, Matplotlib.
- Expected Deliverables: Notebook or training script, preprocessing pipeline, trained regression model, evaluation report.
- Reference: [ageron/handson-ml2](https://github.com/ageron/handson-ml2/blob/master/02_end_to_end_machine_learning_project.ipynb)

## 2. Sentiment Analysis on Movie Reviews

- Category: NLP Classification
- Objective: Classify movie reviews as positive or negative.
- Skills: Text cleaning, tokenization, vectorization, supervised learning for NLP.
- Typical Tools: Python, scikit-learn, NLTK or spaCy, Pandas.
- Expected Deliverables: Text preprocessing pipeline, sentiment classifier, model evaluation, sample predictions.
- Reference: [dishijn2/Sentiment-analysis-on-movie-reveiws](https://github.com/dishijn2/Sentiment-analysis-on-movie-reveiws)

## 3. Image Classification with CNNs

- Category: Computer Vision / Deep Learning
- Objective: Classify images using a convolutional neural network.
- Skills: Image preprocessing, CNN architecture design, training and validation workflows.
- Typical Tools: Python, TensorFlow, Keras, NumPy, Matplotlib.
- Expected Deliverables: Training notebook or script, saved model, training curves, prediction examples.
- Reference: [debdattasarkar/Image-Classification-with-CNN-in-TensorFlow](https://github.com/debdattasarkar/Image-Classification-with-CNN-in-TensorFlow)

## 4. Handwritten Digit Recognition with MNIST

- Category: Deep Learning / Computer Vision
- Objective: Recognize handwritten digits using the MNIST dataset.
- Skills: Neural networks, image normalization, model training, classification metrics.
- Typical Tools: Python, TensorFlow or Keras, NumPy, Matplotlib.
- Expected Deliverables: Digit classifier, model evaluation, confusion matrix, example predictions.
- Reference: [shubham99bisht/Handwritten-digit-recognition-MNIST](https://github.com/shubham99bisht/Handwritten-digit-recognition-MNIST)

## 5. Spam Email Detection

- Category: NLP Classification
- Objective: Detect whether email text is spam or legitimate.
- Skills: Text preprocessing, feature extraction, binary classification, model comparison.
- Typical Tools: Python, scikit-learn, NLTK, Pandas.
- Expected Deliverables: Spam detection pipeline, trained model, evaluation metrics, inference examples.
- Reference: [omaarelsherif/Email-Spam-Detection-Using-NLP](https://github.com/omaarelsherif/Email-Spam-Detection-Using-NLP)

## 6. Fake News Detection

- Category: NLP Classification
- Objective: Classify news articles or claims as fake or real.
- Skills: Dataset preparation, TF-IDF or similar features, classification models, evaluation.
- Typical Tools: Python, scikit-learn, Pandas, NLP preprocessing libraries.
- Expected Deliverables: Data pipeline, fake news classifier, validation results, sample predictions.
- Reference: [nishitpatel01/Fake_News_Detection](https://github.com/nishitpatel01/Fake_News_Detection)

## 7. Breast Cancer Detection

- Category: Healthcare ML / Classification
- Objective: Predict whether tumors are benign or malignant using diagnostic features.
- Skills: Classification modeling, feature analysis, interpretability, evaluation on medical-style tabular data.
- Typical Tools: Python, scikit-learn, Pandas, Seaborn, Matplotlib.
- Expected Deliverables: Clean training pipeline, classification report, confusion matrix, feature insights.
- Reference: [0205Rahul/Breast-Cancer-prediction-for-Wisconsin-diagnostic-data-set](https://github.com/0205Rahul/Breast-Cancer-prediction-for-Wisconsin-diagnostic-data-set)

## 8. Text Summarization with NLP

- Category: NLP / Summarization
- Objective: Generate shorter summaries from long-form text.
- Skills: Text preprocessing, extractive or abstractive summarization, evaluation of summary quality.
- Typical Tools: Python, transformers or classic NLP libraries, Pandas.
- Expected Deliverables: Summarization pipeline, example input-output pairs, notes on model quality and limitations.
- Reference: [praj2408/Text-Summarizer-Project](https://github.com/praj2408/Text-Summarizer-Project)

## 9. Stock Price Prediction with LSTM

- Category: Time Series / Deep Learning
- Objective: Forecast stock prices from historical time series data using LSTMs.
- Skills: Sequence preparation, scaling, recurrent neural networks, time series evaluation.
- Typical Tools: Python, TensorFlow or Keras, Pandas, NumPy, yfinance or CSV data sources.
- Expected Deliverables: LSTM training workflow, forecast plots, comparison between predicted and actual trends.
- Reference: [ashendrasharma/Stock-Price-Prediction-Using-LSTM](https://github.com/ashendrasharma/Stock-Price-Prediction-Using-LSTM)

## 10. Object Detection with YOLO

- Category: Computer Vision / Object Detection
- Objective: Detect and localize objects in images or video using YOLO.
- Skills: Detection pipelines, bounding boxes, confidence thresholds, inference workflows.
- Typical Tools: Python, OpenCV, PyTorch or YOLO tooling, Jupyter notebooks.
- Expected Deliverables: Detection notebook or script, sample output images, model setup instructions, inference examples.
- Reference: [Garima13a/YOLO-Object-Detection](https://github.com/Garima13a/YOLO-Object-Detection)

## Suggested Build Order

If this repository is developed as a learning path, the following sequence gives a smooth progression from fundamentals to more advanced topics:

1. Predicting Housing Prices
2. Sentiment Analysis on Movie Reviews
3. Spam Email Detection
4. Breast Cancer Detection
5. Handwritten Digit Recognition with MNIST
6. Image Classification with CNNs
7. Fake News Detection
8. Text Summarization with NLP
9. Stock Price Prediction with LSTM
10. Object Detection with YOLO

## Common Deliverables Across Projects

To keep the repository consistent, each project should ideally include:

- `README.md` with project-specific overview
- `requirements.txt` or environment setup notes
- `notebooks/` for exploration
- `src/` for reusable code
- `reports/` or `results/` for outputs and evaluation
- Clear dataset and model notes

