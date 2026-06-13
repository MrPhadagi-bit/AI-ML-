# 05. Spam Email Detection

This project detects whether an email message is spam or ham. It uses a text-classification workflow with TF-IDF features and supervised learning.

## Objective

Classify email text into `spam` or `ham` so suspicious messages can be flagged.

## Workflow

1. Load labeled email examples from `sample_data/spam_emails.csv`.
2. Convert message text into TF-IDF vectors.
3. Train candidate classifiers.
4. Save the best text classification bundle.
5. Predict spam status for new email text.

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python src/train.py
```

Use another CSV:

```bash
python src/train.py --data-path path/to/spam_emails.csv
```

## Predict

```bash
python src/predict.py --text "Congratulations, you won a prize. Click now to claim."
```

## Outputs

| File | Description |
|------|-------------|
| `artifacts/spam_bundle.joblib` | Saved vectorizer and classifier bundle. |
| `artifacts/spam_report.json` | Training report and classification metrics. |

## Example Outcome

The prediction script prints the selected model, final spam or ham prediction, and class probabilities when available. See [OUTCOME.md](OUTCOME.md).

## Improvement Plan

- Add more realistic spam and ham examples.
- Add sender, subject, and URL-count features.
- Add precision-recall reporting for spam detection.
- Add threshold tuning to reduce false positives.
