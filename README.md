# DL-Assignment-2


# CNN from Scratch: Image Classification on iNaturalist

This project implements a Convolutional Neural Network (CNN) from scratch in PyTorch to classify images from the iNaturalist 12K dataset. The model is trained with different architectural configurations using Weights & Biases hyperparameter sweeps.

---

## Project Features

- Modular CNN architecture
- Image preprocessing and augmentation
- Train/validation/test split with class balancing
- Dynamic model creation with custom filters and activations
- W&B integration for experiment tracking and sweeps
- Evaluation and visualization of best model predictions

---

## Installation

Install the required dependencies:

```bash
pip install torch torchvision wandb matplotlib
```

---

## Dataset Structure

```
inaturalist_12K/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
└── val/
    ├── class1/
    ├── class2/
    └── ...
```

---

## Model Architecture

The CNN model is defined dynamically using lists of filter sizes, kernel sizes, dropout, and batchnorm flags. The number of dense layers is limited to one before the output.

```python
class CNN(nn.Module):
    def __init__(self, num_classes, conv_filters=[...], kernel_sizes=[...], ...):
        ...
```

---

## Data Preprocessing

- Resize to 128x128
- Normalize to [-1, 1]
- Optionally apply augmentations: horizontal flip, rotation, color jitter

```python
transform = transforms.Compose([...])
augmented_transform = transforms.Compose([...])
```

---

## Training

Model is trained using a standard PyTorch loop. W&B is used to log losses and accuracy.

```python
def train(config):
    ...
```

---

## Hyperparameter Sweep (W&B)

Sweep configuration includes:

- Learning rate
- CNN filter layout
- Activation function
- Batch size
- Dropout rate
- BatchNorm usage
- Data augmentation

```python
sweep_config = { 'method': 'bayes', 'parameters': {...} }
```

Run sweep with:

```python
wandb.agent(sweep_id, function=train, count=10)
```

---

## Best Model Evaluation

Evaluates the top-performing model (based on validation accuracy) on the test dataset and logs final accuracy.

```python
trained_model = train(best_config)
```

---

## Visualization

Creates a 10x3 image grid of test predictions, highlighting correct predictions in green and incorrect ones in red.

```python
plt.subplots(10, 3)
...
wandb.log({"Sample Predictions Grid": wandb.Image(fig)})
```

---

## Results

All training and validation metrics, model configurations, and prediction grids are tracked and visualized on [Weights & Biases](https://wandb.ai).

---

## Author Notes

- Designed for experimentation and reproducibility
- Easily extendable to other datasets or deeper CNNs
- W&B project name: `inaturalist-hyperparam-tuning`
