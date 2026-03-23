from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "predictions"


def normalize_source(raw_source: str) -> str | int:
    return int(raw_source) if raw_source.isdigit() else raw_source


def validate_source(source: str | int) -> None:
    if isinstance(source, int):
        return

    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source path does not exist: {source_path}")

    if source_path.is_dir():
        media_files = list(source_path.glob("*.*"))
        if not media_files:
            raise FileNotFoundError(
                f"No media files were found in {source_path}. Add images or videos before running detection."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO object detection.")
    parser.add_argument("--source", required=True, help="Image path, video path, directory, or webcam index.")
    parser.add_argument("--weights", default="yolov8n.pt", help="YOLO model weights to use.")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    source = normalize_source(args.source)
    validate_source(source)

    model = YOLO(args.weights)
    results = model.predict(
        source=source,
        conf=args.confidence,
        save=True,
        project=str(args.output_dir.parent),
        name=args.output_dir.name,
        exist_ok=True,
    )

    print(f"Processed {len(results)} result items.")
    print(f"Outputs saved under: {args.output_dir}")


if __name__ == "__main__":
    main()

