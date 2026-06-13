from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.datasets import load_digits


def load_digit_dataset() -> tuple[np.ndarray, np.ndarray]:
    dataset = load_digits()
    return dataset.images.astype(np.float32) / 16.0, dataset.target.astype(np.int64)


def image_path_to_digit_vector(image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("L").resize((8, 8))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array.reshape(1, -1)


def export_demo_images(output_dir: Path, samples_per_digit: int = 2) -> None:
    images, labels = load_digit_dataset()
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {digit: 0 for digit in range(10)}

    for image, label in zip(images, labels):
        if counts[int(label)] >= samples_per_digit:
            continue
        digit_dir = output_dir / str(int(label))
        digit_dir.mkdir(parents=True, exist_ok=True)
        image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(image_uint8, mode="L").save(digit_dir / f"digit_{counts[int(label)] + 1}.png")
        counts[int(label)] += 1
        if all(count >= samples_per_digit for count in counts.values()):
            break
