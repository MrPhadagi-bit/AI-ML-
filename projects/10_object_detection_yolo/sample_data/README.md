# Object Detection Sample Data

This folder stores demo scenes and custom images for the object detection workflow. The included demo images show simple colored objects that the local detector can identify and annotate.

## Generate Demo Images

Run this command from the project folder:

```bash
python src/generate_demo_data.py
```

It creates:

- `sample_data/demo_scene_1.png`
- `sample_data/demo_scene_2.png`

## Add Custom Images

You can place `.png`, `.jpg`, or `.jpeg` files in this folder and run detection against the whole folder:

```bash
python src/detect.py --source sample_data
```

Or run detection against one image:

```bash
python src/detect.py --source sample_data/demo_scene_1.png
```

## Detection Settings

| Option | Purpose |
|--------|---------|
| `--threshold` | Controls how strongly a region must differ from the background. |
| `--min-area` | Ignores tiny regions below the selected pixel area. |
| `--output-dir` | Changes where annotated images and reports are saved. |

## Expected Outcome

The detector reads images from this folder and writes annotated images plus reports to:

- `artifacts/predictions/`

## Data Plan

For stronger demos, add images with clear lighting, visible object boundaries, and minimal background clutter. For a true YOLO implementation, this folder can later be expanded into `images/train`, `images/val`, and labeled annotation files.
