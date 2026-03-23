from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "sample_data" / "articles.csv"
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "will",
    "with",
}


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def summarize(text: str, sentence_count: int) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return ""
    if len(sentences) <= sentence_count:
        return " ".join(sentences)

    tokens = [token for token in tokenize(text) if token not in STOPWORDS]
    token_frequencies = Counter(tokens)
    sentence_scores: list[tuple[int, float, str]] = []

    for index, sentence in enumerate(sentences):
        sentence_tokens = [token for token in tokenize(sentence) if token not in STOPWORDS]
        if not sentence_tokens:
            sentence_scores.append((index, 0.0, sentence))
            continue
        score = sum(token_frequencies[token] for token in sentence_tokens) / len(sentence_tokens)
        sentence_scores.append((index, score, sentence))

    top_sentences = sorted(sentence_scores, key=lambda item: item[1], reverse=True)[:sentence_count]
    ordered_summary = [sentence for _, _, sentence in sorted(top_sentences, key=lambda item: item[0])]
    return " ".join(ordered_summary)


def get_default_article(row_index: int, data_path: Path) -> tuple[str, str]:
    dataset = pd.read_csv(data_path)
    if not {"title", "article"}.issubset(dataset.columns):
        raise ValueError(f"Expected columns {{'title', 'article'}} in {data_path}")
    row = dataset.iloc[row_index]
    return str(row["title"]), str(row["article"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an extractive text summary.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--text", type=str, default=None, help="Optional custom text to summarize.")
    parser.add_argument("--sentences", type=int, default=2)
    args = parser.parse_args()

    if args.text:
        title = "Custom Input"
        article = args.text
    else:
        title, article = get_default_article(args.row_index, args.data_path)

    summary = summarize(article, max(1, args.sentences))
    print(f"Title: {title}")
    print("\nSummary:")
    print(summary)


if __name__ == "__main__":
    main()

