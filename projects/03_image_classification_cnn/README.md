# Image Classification with CNN

Train a convolutional neural network for image classification using the CIFAR-10 dataset from TensorFlow.

## What is included

- A starter TensorFlow CNN training script
- Configurable training epochs and dataset size limit
- Model saving to `artifacts/cnn_image_classifier.keras`

## Quick start

```bash
pip install -r requirements.txt
python src/train.py --epochs 3
```

To speed up experimentation during development:

```bash
python src/train.py --epochs 1 --limit 5000
```

Note: TensorFlow downloads CIFAR-10 automatically the first time you run the script.

