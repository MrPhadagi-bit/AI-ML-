from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


LABEL_NAMES = ["circle", "square", "triangle"]
PALETTE = [
    (220, 60, 60),
    (50, 130, 220),
    (245, 170, 50),
    (60, 170, 90),
    (150, 90, 210),
]


def render_shape_image(label_name: str, image_size: int, rng: random.Random) -> Image.Image:
    image = Image.new("RGB", (image_size, image_size), (250, 250, 248))
    draw = ImageDraw.Draw(image)
    color = PALETTE[rng.randrange(len(PALETTE))]
    margin = max(6, image_size // 6)
    left = rng.randint(margin // 2, margin)
    top = rng.randint(margin // 2, margin)
    right = rng.randint(image_size - margin, image_size - margin // 2)
    bottom = rng.randint(image_size - margin, image_size - margin // 2)

    if label_name == "square":
        draw.rectangle((left, top, right, bottom), fill=color)
    elif label_name == "circle":
        draw.ellipse((left, top, right, bottom), fill=color)
    elif label_name == "triangle":
        apex = (rng.randint(image_size // 3, 2 * image_size // 3), top)
        left_point = (left, bottom)
        right_point = (right, bottom)
        draw.polygon([apex, left_point, right_point], fill=color)
    else:
        raise ValueError(f"Unsupported label name: {label_name}")

    for _ in range(image_size // 3):
        x = rng.randrange(image_size)
        y = rng.randrange(image_size)
        image.putpixel((x, y), tuple(max(0, channel - rng.randint(0, 20)) for channel in image.getpixel((x, y))))
    return image


def generate_dataset(samples_per_class: int, image_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    images: list[np.ndarray] = []
    labels: list[int] = []

    for label_index, label_name in enumerate(LABEL_NAMES):
        for _ in range(samples_per_class):
            image = render_shape_image(label_name, image_size, rng)
            images.append(np.asarray(image, dtype=np.float32) / 255.0)
            labels.append(label_index)

    return np.stack(images), np.array(labels, dtype=np.int64)


def image_file_to_array(image_path: Path, image_size: int) -> np.ndarray:
    image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    return np.asarray(image, dtype=np.float32) / 255.0


def save_demo_images(output_dir: Path, image_size: int, seed: int, copies_per_label: int = 3) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    for label_name in LABEL_NAMES:
        label_dir = output_dir / label_name
        label_dir.mkdir(parents=True, exist_ok=True)
        for index in range(copies_per_label):
            image = render_shape_image(label_name, image_size, rng)
            image.save(label_dir / f"{label_name}_{index + 1}.png")
