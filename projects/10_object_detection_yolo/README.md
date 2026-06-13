# 10. Object Detection YOLO Workflow

This project demonstrates an object-detection workflow using generated demo scenes. It detects colored objects, saves annotated images, and writes detection reports. The project name references YOLO-style object detection, while the local implementation is lightweight and runnable without downloading large model weights.

## Objective

Detect visible objects in images, draw bounding boxes, and save structured detection outputs.

## Workflow

1. Generate demo image scenes with simple objects.
2. Run the detection script on one image or a folder of images.
3. Identify object regions using local image processing.
4. Save annotated images with bounding boxes.
5. Save a report for each processed image.

## Setup

```bash
pip install -r requirements.txt
```

## Generate Demo Data

```bash
python src/generate_demo_data.py
```

## Detect Objects

Run detection on the default sample-data folder:

```bash
python src/detect.py
```

Run detection on a custom source:

```bash
python src/detect.py --source path/to/image_or_folder --threshold 35 --min-area 60
```

## Outputs

| File or Folder | Description |
|----------------|-------------|
| `sample_data/demo_scene_1.png` | Generated demo scene. |
| `sample_data/demo_scene_2.png` | Generated demo scene. |
| `artifacts/predictions/` | Annotated images and detection reports. |

## Example Outcome

The detection script prints each processed image path, annotated output path, report path, and number of detections. See [OUTCOME.md](OUTCOME.md).

## Improvement Plan

- Add a true YOLO model integration.
- Add class labels for detected object types.
- Add confidence scores from a learned detector.
- Add batch evaluation against labeled bounding boxes.
