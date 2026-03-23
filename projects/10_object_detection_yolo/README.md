# Object Detection with YOLO

This project has been upgraded into a runnable local object-detection workflow with demo-scene generation, bounding-box output, and saved JSON reports. It uses a lightweight connected-components detector so the project works on this machine now, and it can be extended to a YOLO backend later if you want.

## What is included

- A demo scene generator
- A detection script that writes annotated images and JSON detections
- Batch processing for a single image or a folder of images

## Quick start

```bash
pip install -r requirements.txt
python src/generate_demo_data.py
python src/detect.py
```

Outputs are written under `artifacts/predictions`.
