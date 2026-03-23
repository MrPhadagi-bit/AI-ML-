from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "sample_data"


def create_scene_one(path: Path) -> None:
    image = Image.new("RGB", (256, 256), (246, 246, 244))
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 40, 110, 120), fill=(220, 70, 70))
    draw.ellipse((150, 30, 230, 110), fill=(60, 120, 220))
    draw.polygon([(60, 190), (20, 240), (100, 240)], fill=(60, 170, 90))
    image.save(path)


def create_scene_two(path: Path) -> None:
    image = Image.new("RGB", (256, 256), (246, 246, 244))
    draw = ImageDraw.Draw(image)
    draw.rectangle((25, 150, 95, 220), fill=(155, 90, 210))
    draw.ellipse((110, 120, 200, 210), fill=(240, 180, 60))
    draw.polygon([(185, 35), (145, 105), (225, 105)], fill=(220, 70, 70))
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo images for the object detection project.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    create_scene_one(args.output_dir / "demo_scene_1.png")
    create_scene_two(args.output_dir / "demo_scene_2.png")
    print(f"Demo scenes saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
