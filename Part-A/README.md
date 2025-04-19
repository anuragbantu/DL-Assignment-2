# CNN From Scratch for Image Classification

This project implements a customizable Convolutional Neural Network (CNN) using PyTorch to classify images from the iNaturalist 12K dataset. The model supports configurable depth, filters, activation functions, dropout, batch normalization, and data augmentation. It also integrates Weights & Biases (W&B) for hyperparameter tuning and logging.

---

## Setup

1. Clone the repository or run the notebook in a Colab/Kaggle environment.
2. Install dependencies:
```bash
pip install torch torchvision wandb matplotlib
```
3. Make sure the dataset is available at:
```
/kaggle/input/inaturalist/inaturalist_12K/train
/kaggle/input/inaturalist/inaturalist_12K/val
```

---

## Data Preprocessing

- **transform**: Standard preprocessing for images (resize to 128×128, normalize to [-1,1]).
- **augmented_transform**: Adds random flips, rotation, and color jitter to improve generalization.

---

## Dataset Loading

Uses `ImageFolder` to load image datasets from directory structure. A validation split of 80:20 is applied to the training set. Data is loaded via PyTorch's `DataLoader`.

---

## CNN Model Definition

```python
class CNN(nn.Module)
```
Defines a customizable CNN:
- `conv_filters`: List of output channels for convolutional layers.
- `kernel_sizes`: List of kernel sizes.
- `activation_fn`: Activation function (e.g., ReLU, GELU).
- `fc_units`: Number of units in the single dense layer.
- `dropout`: Dropout rate.
- `use_batchnorm`: Whether to apply Batch Normalization after each convolution.

Output: logits of shape `(batch_size, num_classes)`.

---

## Activation Map

Defines a mapping for string-based activation configuration:

```python
activation_map = {
    'relu': F.relu,
    'gelu': F.gelu,
    'silu': F.silu,
    'mish': F.mish
}
```

---

## Training Function

```python
def train(config=None)
```
This function performs training for one experiment run, using parameters passed from a W&B sweep. It:
- Applies augmentation if enabled.
- Splits training data into training/validation.
- Trains the CNN model.
- Evaluates performance on validation data each epoch.
- Saves the best-performing model (based on validation accuracy) as `best_model.pth`.
- Logs all metrics to W&B.

---

## Hyperparameter Sweep

Uses `wandb.sweep()` with the following parameters:
- `lr`: Learning rate
- `epochs`: Number of training epochs
- `conv_filters`: Configurations for CNN depth and width
- `activation_fn`: Choice of activation function
- `fc_units`: Dense layer size
- `batch_size`: Size of mini-batches
- `use_batchnorm`: Use of BatchNorm layers
- `use_augmentation`: Use of image augmentation
- `dropout`: Dropout rate

To run the sweep:
```python
wandb.agent(sweep_id, function=train, count=10)
```

---

## Best Model Evaluation

```python
def train(config)
```
Same as above, but used to retrain using the best config found by the sweep. After training, it evaluates the model on the test set and prints test accuracy.

---

## Final Test Evaluation

Evaluates the trained model on the test data and computes final test accuracy:

```python
trained_model.eval()
...
print(f"Test Accuracy: {test_acc:.2f}%")
```

---

## Visualization of Predictions

Generates a 10×3 grid of sample predictions from the test set. Correct predictions are marked in green, and incorrect ones in red. Logged to W&B.

```python
plt.subplots(10, 3)
...
wandb.log({"Sample Predictions Grid": wandb.Image(fig)})
```

---

## Notes

- This project supports fast experimentation with different CNN configurations.
- All experiments are tracked using W&B under the project name `inaturalist-hyperparam-tuning`.
