# Spam Email Detection

Detect whether an email is spam or ham using TF-IDF features and a Multinomial Naive Bayes classifier.

## What is included

- A sample email dataset
- A starter training and evaluation script
- Optional scoring for custom email text

## Quick start

```bash
pip install -r requirements.txt
python src/train.py
```

To predict a custom message:

```bash
python src/train.py --text "Congratulations, you won a free prize. Click now."
```

The trained pipeline is saved to `artifacts/spam_classifier.joblib`.

