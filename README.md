# CMPT_419_ML_Project

# The Help for Helplessness 

This project classifies levels of helplessness in video data using multiple deep learning models, including a 2D CNN + LSTM, a 3D CNN, and a pre-trained SwinTransformer3D. It includes full preprocessing, training pipelines, and a live webcam-based GUI for real-time inference.

---

## Repository Structure

```
CMPT_419_ML_Project/
│
├── classifier_model/                     # All models and dataset scripts
│   ├── cnn_2d_model/
│   │   ├── CNN_LSTM_dataset.py          # Dataset loader for grayscale CNN-LSTM
│   │   ├── CNN_LSTM_model.py            # 2D CNN + LSTM model definition
│   │   ├── CNN_LSTM_training.ipynb      # Notebook for training grayscale CNN-LSTM
│   │   └── grayscale_cnn_lstm.pth       # Trained grayscale CNN-LSTM weights
│   │
│   ├── cnn_3d_model/
│   │   ├── model.py                     # 3D CNN model
│   │   ├── training.ipynb               # Notebook for training 3D CNN model
│   │   └── model_weights.pth            # Trained 3D CNN weights
│   │
│   ├── pre_trained_transformer_model/
│   │   ├── model.py                     # SwinTransformer3D model (uses Torchvision's weights)
│   │   ├── pretrained.ipynb             # Notebook for training SwinTransformer3D model
│   │   └── model_weights.pth            # Trained Swin Transformer weights
│   │
│   ├── dataset.py                       # Dataset loader for 3D model
│   ├── find_mean_std.py                # Utility to compute mean and std of dataset used for 3D model
│   ├── cross_validation.ipynb          # Notebook for running cross-validation on our models
│   └── training.ipynb                  # Skeleton notebook for training a model
│
├── data/                                # Training/validation frame data (processed)
│   ├── train/
│   └── val/
│
├── processed_frames/                    # Output from ETL.py, organized by class
│
├── extreme-helpless/                    # Raw video data
├── little_helplessness/
├── no-helpless/
│
├── .gitignore                           # Ignore weights, logs, OS files
├── .gitattributes                       # Git LFS tracked files
├── .DS_Store                            # macOS system file (safe to delete)
│
├── main.py                              # GUI with model selection + live webcam prediction
├── webcam_capture.py                    # Model inference + preprocessing pipeline (old) used as code reference (can be deleted) 
├── ETL.py                               # Extract-Transform-Load script to preprocess raw videos
├── confusion_matrix.py                  # Model evaluation using confusion matrix + metrics
├── cohen_kappa_rating_sheet.csv         # Dataset inter-rater agreement ratings
├── cohen_kappa.py                       # Python script to calculate Cohen's Kappa for our inter-rater agreements
├── requirements.txt                     # Required packages
└── README.md                            # ← You're here!
```

---

## Setup Instructions

### 1. Clone the Repo

```bash
git clone git@github.sfu.ca:irehman/CMPT_419_ML_Project.git
cd CMPT_419_ML_Project
```

> If you face a checkout error like `.DS_Store would be overwritten`, remove it before cloning:
> ```bash
> rm .DS_Store
> ```

---


### 2. Install Dependencies

Ensure you're using Python 3.10+.

```bash
pip install -r requirements.txt
```

### 3. Additional Requirements

- **Git LFS** is required to pull large model weights:
  ```bash
  git lfs install
  ```

- **ffmpeg** is required by `moviepy` for video processing:
  ```bash
  # On Mac (Homebrew)
  brew install ffmpeg

  # On Ubuntu
  sudo apt install ffmpeg
  ```

---

## How to Run

### Launch the GUI Application

```bash
python main.py
```

Features:
- Real-time webcam feed
- Select between 3 models (2D CNN, 3D CNN, SwinTransformer)
- Class probabilities + inference time display
- Works on macOS (MPS), CUDA, or CPU

---

### Preprocess Video Clips (Optional)

```bash
python ETL.py
```

Converts raw videos in:
```
extreme-helpless/
little_helplessness/
no-helpless/
```
...into 90-frame sequences under `processed_frames/`.

---

### Train the Grayscale 2D CNN + LSTM (Optional)

```bash
jupyter notebook classifier_model/cnn_2d_model/CNN_LSTM_training.ipynb
```

Uses `data/train` and `data/val` folders for training.

---

### Train the 3D CNN (RGB) (Optional)

```bash
jupyter notebook classifier_model/training.ipynb
```

Uses `classifier_model/dataset.py` for data loading.

---

### Evaluate Model (Confusion Matrix)

```bash
python confusion_matrix.py
```

Outputs:
- Confusion matrix (matplotlib)
- Precision, Recall, F1, Accuracy

---

## Model Architectures

### Grayscale 2D CNN + LSTM

- Input: (B, T, 1, 112, 112)
- Three conv layers → LSTM → FC
- Output: 3 classes

### RGB 3D CNN

- Input: (B, 3, T, 224, 224)
- 4 x 3D Conv blocks → GlobalAvgPool → FC layers

### Pre-trained SwinTransformer3D

- Based on `torchvision.models.video.swin3d_t`
- Inference-only model using transfer learning

---

## Self-Evaluation & Reflection

This section reflects on how our project went compared to what we originally proposed, what we added or changed, and any challenges or future ideas we had while working on the project.

---

### What We Planned vs. What We Did

| Objective                                 | Status       | Notes                                                                 |
|------------------------------------------|--------------|-----------------------------------------------------------------------|
| Build 2D CNN + LSTM model (grayscale)    | ✅ Completed | Trained on preprocessed grayscale frame sequences                     |
| Build 3D CNN model (RGB)                 | ✅ Completed | Implemented with 3D conv layers and temporal depth                    |
| Real-time webcam GUI with model selector | ✅ Completed | Supports live prediction for all 3 models                             |
| Data preprocessing pipeline              | ✅ Completed | Extracted consistent-length frame sequences from all raw videos       |
| Dataset of short video clips             | ✅ Completed | Collected 200 samples covering all three levels of helplessness       |
| Evaluate our model performance + dataset | ✅ Completed | Calculated confusion matrices, accuracy and inter-rater agreement     |
| Pre-trained transformer model            | ✅ Added     | We added SwinTransformer3D (not part of original proposal)            |

---

### Changes and Improvements

- **Added a third model** (SwinTransformer3D) to compare performance with pre-trained weights.
- Switched from OpenCV to `tkinter` for GUI to make it work cross-platform (macOS & Windows).
- Improved the GUI to show:
  - Real-time class probabilities
  - Inference time per batch
  - Live webcam feed alongside predictions
- Built a consistent preprocessing pipeline (ETL) to make sure every model received the correct input format.

---

### Things We Could Have Done (Future Improvements)

- **Dimensionality Reduction**  
  We could have reduced frame sequence size or feature dimensions before feeding into the models:
  - Using PCA (Principal Component Analysis) on frame features
  - Using Autoencoders for noise reduction
  - Would improve training speed, especially for 3D CNNs and LSTMs

- **Motion-Aware Cropping**  
  We thought about using frame differencing or optical flow to crop to areas with movement, which could potentially help the model focus more on the subject's body language.

- **Synthetic Data Generation**  
  Our dataset was relatively small and imbalanced. With more time, we could have:
  - Added synthetic data 
  - Simply collected more data

- **Model Optimization**  
  Training took a long time on some systems:
  - In the future, converting models to ONNX or pruning them would help run them faster on edge devices or in production.

---

### Notes for the TA

- We tested our code on **macOS (MPS)** and **Windows (CPU/GPU)**.
- Model weights are stored using **Git LFS** (please install it to get full repo).
- If anything fails, the GUI or models will fall back to CPU automatically.


## Reproducibility Notes

We have organized our code, datasets, and training pipelines to be reproducible.  
All scripts can be run as-is if the directory structure is preserved.  

Models can also be re-trained from scratch using the training notebooks.

If any issues occur due to `.DS_Store` or macOS artifacts, simply delete them:
```bash
find . -name '.DS_Store' -delete
```

---

## Author

**Ibrahim Rehman: irehman@sfu.ca**,
**Greg Parent: gparent@sfu.ca**,
**Daniel Nguyen: dvn1@sfu.ca**  
Course: CMPT 419/724 - Affective Computing
Simon Fraser University

---

