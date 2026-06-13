from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "sample_data" / "demo_scene_1.png"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "predictions"
BACKGROUND_COLOR = np.array([246, 246, 244], dtype=np.int16)
COLOR_MAP = {
    "red": np.array([220, 70, 70]),
    "blue": np.array([60, 120, 220]),
    "green": np.array([60, 170, 90]),
    "yellow": np.array([240, 180, 60]),
    "purple": np.array([155, 90, 210]),
}


def get_source_paths(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        files = [path for path in source.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        if not files:
            raise FileNotFoundError(f"No image files found in {source}")
        return sorted(files)
    raise FileNotFoundError(f"Source path does not exist: {source}")


def closest_color_name(color: np.ndarray) -> str:
    best_name = "object"
    best_distance = float("inf")
    for name, reference in COLOR_MAP.items():
        distance = float(np.linalg.norm(color - reference))
        if distance < best_distance:
            best_distance = distance
            best_name = name
    return best_name


def infer_shape(component_mask: np.ndarray) -> str:
    fill_ratio = float(component_mask.mean())
    if fill_ratio >= 0.72:
        return "square"
    if fill_ratio >= 0.52:
        return "circle"
    return "triangle"


def fallback_detect_image(image_path: Path, output_dir: Path, threshold: int, min_area: int) -> dict[str, object]:
    image = Image.open(image_path).convert("RGB")
    array = np.asarray(image, dtype=np.int16)
    difference = np.linalg.norm(array - BACKGROUND_COLOR, axis=2)
    mask = difference > threshold

    labeled, _ = ndimage.label(mask)
    slices = ndimage.find_objects(labeled)
    detections: list[dict[str, object]] = []
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    for label_index, current_slice in enumerate(slices, start=1):
        if current_slice is None:
            continue
        component_mask = labeled[current_slice] == label_index
        area = int(component_mask.sum())
        if area < min_area:
            continue

        y_slice, x_slice = current_slice
        x0, x1 = x_slice.start, x_slice.stop
        y0, y1 = y_slice.start, y_slice.stop
        component_pixels = array[current_slice][component_mask]
        average_color = component_pixels.mean(axis=0)
        label = f"{closest_color_name(average_color)} {infer_shape(component_mask)}"
        detection = {
            "label": label,
            "bbox": [int(x0), int(y0), int(x1), int(y1)],
            "area": area,
        }
        detections.append(detection)

        draw.rectangle((x0, y0, x1, y1), outline=(20, 20, 20), width=2)
        draw.text((x0 + 2, max(0, y0 - 12)), label, fill=(20, 20, 20))

    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = output_dir / f"{image_path.stem}_annotated.png"
    report_path = output_dir / f"{image_path.stem}_detections.json"
    annotated.save(annotated_path)
    report = {"source": str(image_path), "backend": "fallback_connected_components", "detections": detections}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"annotated_path": str(annotated_path), "report_path": str(report_path), "detections": detections}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local object detection.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold", type=int, default=35)
    parser.add_argument("--min-area", type=int, default=60)
    args = parser.parse_args()

    source_paths = get_source_paths(args.source)
    results = []
    for image_path in source_paths:
        result = fallback_detect_image(image_path, args.output_dir, args.threshold, args.min_area)
        results.append(result)
        print(f"Processed: {image_path}")
        print(f"Annotated image: {result['annotated_path']}")
        print(f"Report: {result['report_path']}")
        print(f"Detections: {len(result['detections'])}")

    print(f"Processed {len(results)} image(s).")


if __name__ == "__main__":
    main()
