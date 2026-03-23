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
python src/predict.py --text "Congratulations, you won a free prize. Click now."
```

## Outputs

- `artifacts/spam_bundle.joblib`
- `artifacts/spam_report.json`
