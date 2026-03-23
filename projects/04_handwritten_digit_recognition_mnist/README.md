# Handwritten Digit Recognition with MNIST

Recognize handwritten digits using a small convolutional neural network trained on MNIST.

## What is included

- A TensorFlow training script for MNIST
- Configurable epochs and optional dataset limit
- Model saving to `artifacts/mnist_digit_classifier.keras`

## Quick start

```bash
pip install -r requirements.txt
python src/train.py --epochs 3
```

For a quick smoke test:

```bash
python src/train.py --epochs 1 --limit 10000
```

TensorFlow downloads MNIST automatically when you run the project for the first time.

