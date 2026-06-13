# 08. Text Summarization NLP

This project generates short extractive summaries from article text. It demonstrates an NLP workflow that scores and selects important sentences rather than generating new text from scratch.

## Objective

Reduce long article text into a shorter summary while preserving the most important information.

## Workflow

1. Load article examples from `sample_data/articles.csv`.
2. Select a row or accept custom text.
3. Score sentences using extractive summarization logic.
4. Return the highest-value sentences as the summary.
5. Optionally evaluate generated summaries against references.

## Setup

```bash
pip install -r requirements.txt
```

## Summarize A Sample Article

```bash
python src/summarize.py --row-index 0 --sentences 2
```

## Summarize Custom Text

```bash
python src/summarize.py --text "Long article text goes here." --sentences 2
```

## Evaluate

```bash
python src/summarize.py --evaluate
```

## Outputs

| File | Description |
|------|-------------|
| `artifacts/summarization_eval.json` | Optional evaluation report when `--evaluate` is used. |

## Example Outcome

The script prints the article title, generated summary, reference summary when available, and overlap metrics. See [OUTCOME.md](OUTCOME.md).

## Improvement Plan

- Add abstractive summarization with transformer models.
- Add ROUGE-style evaluation.
- Add support for URL or PDF article input.
- Add summary length controls for bullet, short, and detailed modes.
