from __future__ import annotations

import argparse
from pathlib import Path

from digit_utils import export_demo_images


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "sample_data" / "demo_digits"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export demo digit images for prediction tests.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples-per-digit", type=int, default=2)
    args = parser.parse_args()

    export_demo_images(args.output_dir, args.samples_per_digit)
    print(f"Demo images saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
