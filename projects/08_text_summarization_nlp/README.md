# Text Summarization with NLP

Generate extractive summaries from long-form text using a lightweight frequency-based summarization approach.

## What is included

- A sample article dataset
- A summarization script that works without downloading large models
- Support for summarizing text from the sample CSV or custom input

## Quick start

```bash
pip install -r requirements.txt
python src/summarize.py
```

To summarize your own text:

```bash
python src/summarize.py --text "Your long article goes here." --sentences 2
```

To summarize a different row from the sample dataset:

```bash
python src/summarize.py --row-index 1
```

