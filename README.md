# Image Classification Model

This repository contains three implementations of a Convolutional Neural Network (CNN) for image classification, each building upon the previous version with progressive improvements.

Each model is independent from the others. Thus, while it builds on the previous model, each file includes the relevant code from the previous model, and therefore does not need to import from the other files.

***NB***: Improvements to these models are likely to be made in due course - see 'Development History' section below

## Models overview

### 1. Baseline Model

A simple CNN architecture implemented with TensorFlow for CIFAR-100 classification:
- Input layer handling 32x32x3 images
- Two convolutional blocks:
    - First block: Conv2D(16, 7x7) + ReLU + MaxPooling(2x2)
    - Second block: Conv2D(32, 5x5) + ReLU + MaxPooling(2x2)
- Flatten layer
- Dense layer with 128 units and ReLU activation
- Output layer with 100 units (for CIFAR-100 classes)
- Adam optimizer with default learning rate
- Training for 10 epochs with batch size of 32

This model is in the file `baseline_model.py`.

### 2. Enhanced Model

Significant improvements over the baseline through architectural changes and regularisation:
- Data augmentation:
    - Horizontal random flip
    - Random contrast adjustment (0.1)
- Input normalisation layer
- Enhanced convolutional architecture:
    - Initial Conv2D(24, 3x3) with BatchNorm and LayerNorm
    - Two Inception modules with varying filter sizes
    - Additional Conv2D(64, 4x4) layers
- Inception modules featuring:
    - 1x1 convolutions
    - 3x3 convolutions with dimensionality reduction
    - 5x5 convolutions with dimensionality reduction
    - Max pooling followed by 1x1 convolutions
- Batch and Layer normalisation throughout the network
- Same training configuration (10 epochs, batch size 32)

This model is in the file titled `enhanced_model.py`.

### 3. Model Improvements
Final optimisations focusing on training dynamics and GPU utilization:

- GPU memory growth configuration
- Learning rate scheduling:
    - Initial learning rate: 1e-3
    - Exponential decay schedule
    - Decay steps: 100
    - Decay rate: 0.96
    - Staircase implementation
- Uses legacy Adam optimizer for better compatibility
- Maintains the enhanced architecture from Model 2
- Type hints and better error handling for GPU configuration

This model is in the file titled `additional_model_improvements.py`.

## Requirements

```python
tensorflow>=2.0.0
numpy<2.0

## Development History

This project evolved from a single Jupyter notebook (.ipynb) file, which was refactored into separate modules for better maintainability and reuse. Each model represents a distinct development phase in improving the classification performance.

__NB__: This project was originally created as part of a University assignment. Therefore, some aspects of each model were tailored towards meeting the assignment criteria and compute availability. Accuracy of the models may therefore not be as high as it could be. While some refactoring has occurred, these models are likely to be further refined in due course to improve their utility.