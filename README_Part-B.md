
# Fine-Tuning EfficientNet on iNaturalist Dataset

This project demonstrates how to fine-tune a pretrained EfficientNet model on the iNaturalist 12K dataset using PyTorch. It includes steps for data preparation, model customization, training, validation, and evaluation.

---

## Setup Instructions

1. **Environment**: Google Colab (recommended) or local machine with CUDA GPU.
2. **Install Dependencies**:
```bash
pip install torch torchvision matplotlib
```
3. **Dataset**: Place `nature_12K.zip` in your Google Drive under `/MyDrive/`. The dataset should have the following structure after extraction:
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

## Step-by-Step Code Breakdown

### 1. Imports and Setup
Mounts Google Drive and loads required libraries for model training and image transformation.

### 2. Load Pretrained EfficientNet
```python
from torchvision.models import efficientnet_v2_s
model = efficientnet_v2_s(pretrained=True)
```
Loads an EfficientNet-V2 small model pretrained on ImageNet.

### 3. Data Preprocessing
```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```
Standard normalization and resizing to 224×224 for compatibility with EfficientNet.

### 4. Dataset Loading and Splitting
```python
ImageFolder(root=..., transform=...)
```
Loads train and validation images. Performs an 80:20 split using `random_split`.

### 5. Model Modification
```python
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
```
Replaces the classifier to match the number of classes in the dataset.

### 6. Freezing Layers
```python
for name, child in model.features.named_children():
    if int(name) < 4:
        for param in child.parameters():
            param.requires_grad = False
```
Freezes the first 4 blocks of the EfficientNet model to retain pre-trained features.

### 7. Loss Function and Optimizer
```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
```
Only updates parameters that are unfrozen. Uses Adam optimizer and cross-entropy loss.

---

## Training and Validation

### Loop Overview
```python
for epoch in range(epochs):
    ...
```

**Key Steps**:
- Sets the model to training mode.
- Iterates through `train_loader` to update model weights.
- Computes training accuracy and loss.
- Sets the model to evaluation mode.
- Computes validation accuracy without updating weights.

### Functionality
- Tracks metrics across epochs.
- Prints loss and accuracy after each epoch.
- Saves and compares initial/final weights of frozen blocks to verify freezing.

---

## Evaluation on Test Set
```python
model.eval()
with torch.no_grad():
    ...
```
Predicts labels on the held-out test set and computes overall test accuracy.

---

## Performance Summary

| Scenario                                 | Test Accuracy |
|------------------------------------------|---------------|
| Without freezing                         | 84.85%        |
| Freezing all layers except classifier    | 70.10%        |
| Freezing only first 4 layers             | 85.45%        |
| Pretrained model (no fine-tuning)        | 13.00%        |

---

## Conclusion

- Fine-tuning pretrained models improves performance dramatically.
- Freezing early layers while fine-tuning later layers retains learned features while adapting to new data.
- This project provides a simple but effective template for transfer learning in PyTorch.


