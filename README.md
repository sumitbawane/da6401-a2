# DA6401 Assignment 2: Deep Learning Framework for Image Classification

This repository contains the implementation of an image classification framework for the iNaturalist dataset. The project is divided into two main parts:
1. **Part A:** Training a custom convolutional neural network (CNN) with hyperparameter optimization using Weight & Biases (W&B).
2. **Part B:** Fine-tuning a pre-trained EfficientNetV2 model.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Dependencies](#dependencies)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
  - [Part A](#part-a)
  - [Part B](#part-b)
  - [Jupyter Notebook](#jupyter-notebook)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Results](#results)
---

## Project Overview

This project is designed to classify images from the iNaturalist dataset. It leverages PyTorch for deep learning, supports mixed precision training, and integrates W&B for experiment tracking and hyperparameter optimization.

- **Part A:** Implements a custom CNN architecture with configurable hyperparameters (e.g., kernel size, activation functions, dropout rates) and performs hyperparameter optimization using W&B sweeps.
- **Part B:** Fine-tunes a pre-trained EfficientNetV2 model to adapt it to the iNaturalist dataset.

---

## Features

- Custom CNN implementation with configurable layers and parameters.
- Fine-tuning of pre-trained models using transfer learning.
- Support for data augmentation during training.
- Mixed precision training for computational efficiency.
- Integration with W&B for logging metrics and hyperparameter sweeps.
- Automatic saving of the best model during training.
- Visualization of test predictions in a grid format.

---

## Directory Structure

```
da6401-a2/
├── PartA/
│   ├── best_config.py    # Script to train the CNN using the best hyperparameters
│   ├── cnn.py            # Definition of the custom CNN model
│   ├── data_utils.py     # Data loading and transformation utilities
│   ├── evaluate.py       # Evaluate the trained model and visualize results
│   ├── main.py           # Entry point for W&B sweeps
│   ├── model_utils.py    # Utility functions for training and evaluation
│   ├── train.py          # Training script for custom CNN
├── PartB/
│   ├── finetune.py       # Script for fine-tuning EfficientNetV2
├── da6401-a2.ipynb       # Jupyter Notebook demonstrating the workflow
└── inaturalist_12K/      # Dataset directory (not included in the repo)
```

---

## Dependencies

- Python 3.8 or later
- PyTorch
- Torchvision
- W&B
- Jupyter Notebook
- Other dependencies listed in `requirements.txt`



---

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/sumitbawane/da6401-a2.git
   cd da6401-a2
   ```


2. Download the iNaturalist dataset and place it in the `inaturalist_12K` directory.

3. Log in to Weights & Biases:
   ```bash
   wandb login
   ```

---

## Usage

### Part A

#### Training the Custom CNN
1. Configure the hyperparameters in the sweep configuration (located in `PartA/main.py`).
2. Run the sweep:
   ```bash
   python PartA/main.py
   ```
3. Monitor the training progress on the W&B dashboard.

#### Training with Best Configuration
1. The `PartA/best_config.py` script trains the model using the best hyperparameters obtained from the sweep.
2. Run the script:
   ```bash
   python PartA/best_config.py
   ```
3. The model with the best validation accuracy will be saved as `best_model.pth`.

#### Evaluating the Model
1. Use the `PartA/evaluate.py` script to evaluate the trained model on the test set and visualize predictions.
2. Run the script:
   ```bash
   python PartA/evaluate.py
   ```
3. This will save a 10×3 prediction grid as `prediction_grid.png` and print the test accuracy.

### Part B

#### Fine-tuning EfficientNetV2
1. Modify the data path in `PartB/finetune.py` to point to the dataset.
2. Run the fine-tuning script:
   ```bash
   python PartB/finetune.py
   ```
3. The model with the best validation accuracy will be saved as `best_efficientnetv2.pth`.

### Jupyter Notebook

#### Interactive Training and Testing
1. Open the notebook:
   ```bash
   jupyter notebook da6401-a2.ipynb
   ```
2. The notebook includes:
   - GPU availability check and configuration.
   - Data preprocessing and loading.
   - Training with the best hyperparameter configuration from sweeps.
   - Fine-tuning EfficientNetV2.
   - Testing the model with saved weights.
   - Visualization of results and metrics.

---

## Hyperparameter Optimization

The project uses W&B sweeps for hyperparameter optimization. The following parameters are optimized in Part A:
- Filter strategy (`same`, `doubling`, `halving`)
- Base filter size
- Kernel sizes
- Activation function (`ReLU`, `GELU`, `SiLU`, etc.)
- Use of batch normalization
- Dropout rate
- Fully connected layer size
- Learning rate
- Batch size
- Data augmentation
- Number of epochs

### Example Sweep Configuration
```python
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val_acc', 'goal': 'maximize'},
    'parameters': {
        'filter_strategy': {'values': ['same', 'doubling', 'halving']},
        'base_filter': {'values': [32, 64]},
        'kernel_sizes': {'values': [[3, 3, 5, 5, 7], [5, 5, 7, 7, 3]]},
        'activation': {'values': ['ReLU', 'GELU']},
        'batch_norm': {'values': [True, False]},
        'dropout': {'values': [0.0, 0.2]},
        'fc_size': {'values': [256, 512]},
        'learning_rate': {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-3},
        'batch_size': {'values': [32, 64]},
        'data_augmentation': {'values': [True, False]},
        'epochs': {'values': [10, 15, 20]}
    }
}
```

---

## Results

### Part A
- **Best Validation Accuracy:** ~43% Achieved using hyperparameter optimization with W&B sweeps.
- **Test Accuracy:** ~43% Evaluated using the `evaluate.py` script.

### Part B
- **Fine-tuned EfficientNetV2:** >75% Achieved significant improvement in classification accuracy compared to the base CNN.

