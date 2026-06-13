from __future__ import annotations

import argparse
from pathlib import Path

from image_utils import save_demo_images


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "sample_data" / "demo_images"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo shape images for prediction tests.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--copies-per-label", type=int, default=3)
    args = parser.parse_args()

    save_demo_images(args.output_dir, args.image_size, seed=7, copies_per_label=args.copies_per_label)
    print(f"Demo images saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
