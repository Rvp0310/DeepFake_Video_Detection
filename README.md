# Deepfake Detection (Spatial + Frequency)

## Project Overview

Lightweight deepfake detection system using spatial (RGB frames) and frequency (FFT) features.
The pipeline covers preprocessing, training, evaluation, and a demo inference interface.

---

## Directory Structure

```
project/
│
├── Notebook_01.ipynb   (Preprocessing)
├── Notebook_02.ipynb   (Training)
├── Notebook_03.ipynb   (Evaluation)
├── Notebook_04.ipynb   (Interface)
│
├── preprocessed/     (not in repository due to size limitation, contains saved tensor to load)
│   ├── train/
│   ├── val/
│   └── test/
│
├── Model3.pth    (model currently used)
└── README.md      (this file)
```

---

## Workflow

Videos
-> Frame Sampling
-> Face Detection
-> Spatial + Frequency Extraction
-> Save as tensors
-> Train Model
-> Evaluate
-> Demo Inference

---

## Notebook 1: Preprocessing

### Purpose

* Sample a fixed number of frames per video
* Extract faces from frames
* Generate spatial (RGB) and frequency (FFT) representations
* Store processed samples as tensor files

### Key Steps

* Face detection and tracking
* Handling missing or insufficient frames
* Frequency transformation using FFT
* Normalization and tensor formatting
* Saving data in train/val/test splits

---

## Notebook 2: Training

### Purpose

* Load preprocessed data
* Train the model using augmentation for spatial and frequency features
* Monitor performance using validation set
* Early Stopping based on Validation Data Loss
* Save the best performing model

### Components

* Custom dataset loader for tensor files
* Dual-branch model (spatial + frequency)
* Loss function: Binary Cross Entropy with logits
* Optimizer: AdamW
* Early stopping to prevent overfitting

---

## Notebook 3: Evaluation

### Purpose

* Evaluate trained model on unseen test dataset
* Compute performance metrics

### Metrics Used

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

### Notes

* Default classification threshold: 0.5
* Adjusted threshold (e.g., 0.7) used for cross-dataset evaluation

---

## Notebook 4: Demo

### Purpose

* Provide a simple interface for real-time prediction
* Upload a video and classify it as real or fake

### Process

* Video upload
* Face extraction
* Feature generation (spatial + frequency)
* Model inference
* Output probability and classification

---

## Generalization

* Model performs well on trained datasets
* On unseen datasets, performance is maintained with threshold calibration
* Demonstrates transferable feature learning across domains

---

## Conclusion

The model effectively combines spatial and frequency information for deepfake detection.
It achieves strong performance on known datasets and shows reasonable generalization to unseen data with minor calibration.

This framework can be extended further with additional modalities such as audio for improved robustness.
