# Object Detection with YOLO

Run object detection on images or videos with the Ultralytics YOLO interface.

## What is included

- A starter detection script
- Support for image, video, folder, or webcam sources
- Output saving to `artifacts/predictions`

## Quick start

```bash
pip install -r requirements.txt
python src/detect.py --source path/to/image.jpg
```

To use a folder of images:

```bash
python src/detect.py --source sample_data
```

To use a webcam:

```bash
python src/detect.py --source 0
```

Notes:

- The first run downloads the YOLO weights automatically.
- Place any test images you want to keep in `sample_data/`.

