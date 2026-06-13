from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "sample_data" / "articles.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_EVAL_PATH = ARTIFACTS_DIR / "summarization_evaluation.json"
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


def score_summary(reference: str, candidate: str) -> dict[str, float]:
    reference_tokens = [token for token in tokenize(reference) if token not in STOPWORDS]
    candidate_tokens = [token for token in tokenize(candidate) if token not in STOPWORDS]
    if not reference_tokens or not candidate_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    reference_counter = Counter(reference_tokens)
    candidate_counter = Counter(candidate_tokens)
    overlap = sum(min(candidate_counter[token], reference_counter[token]) for token in candidate_counter)
    precision = overlap / len(candidate_tokens)
    recall = overlap / len(reference_tokens)
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def load_dataset(data_path: Path) -> pd.DataFrame:
    dataset = pd.read_csv(data_path)
    required_columns = {"title", "article"}
    if not required_columns.issubset(dataset.columns):
        raise ValueError(f"Expected columns {required_columns} in {data_path}")
    return dataset


def evaluate_dataset(data_path: Path, sentence_count: int, eval_path: Path) -> dict[str, object]:
    dataset = load_dataset(data_path)
    if "summary" not in dataset.columns:
        raise ValueError("Expected a 'summary' column for evaluation.")

    rows: list[dict[str, object]] = []
    for _, row in dataset.iterrows():
        generated = summarize(str(row["article"]), sentence_count)
        metrics = score_summary(str(row["summary"]), generated)
        rows.append(
            {
                "title": str(row["title"]),
                "generated_summary": generated,
                "reference_summary": str(row["summary"]),
                "scores": metrics,
            }
        )

    average = {
        "precision": sum(row["scores"]["precision"] for row in rows) / len(rows),
        "recall": sum(row["scores"]["recall"] for row in rows) / len(rows),
        "f1": sum(row["scores"]["f1"] for row in rows) / len(rows),
    }
    report = {"rows": rows, "average_scores": average}
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate or evaluate extractive text summaries.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--text", type=str, default=None, help="Optional custom text to summarize.")
    parser.add_argument("--sentences", type=int, default=2)
    parser.add_argument("--evaluate", action="store_true", help="Evaluate against reference summaries.")
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_EVAL_PATH)
    args = parser.parse_args()

    if args.evaluate:
        report = evaluate_dataset(args.data_path, max(1, args.sentences), args.eval_path)
        print(json.dumps(report, indent=2))
        print(f"Evaluation saved to: {args.eval_path}")
        return

    dataset = load_dataset(args.data_path)
    if args.text:
        title = "Custom Input"
        article = args.text
        reference = None
    else:
        row = dataset.iloc[args.row_index]
        title = str(row["title"])
        article = str(row["article"])
        reference = str(row["summary"]) if "summary" in row else None

    summary = summarize(article, max(1, args.sentences))
    print(f"Title: {title}")
    print("\nSummary:")
    print(summary)
    if reference:
        print("\nReference summary:")
        print(reference)
        print("\nOverlap metrics:")
        print(json.dumps(score_summary(reference, summary), indent=2))


if __name__ == "__main__":
    main()
