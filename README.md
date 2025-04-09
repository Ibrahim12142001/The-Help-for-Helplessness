# CMPT_419_ML_Project

# Helplessness Classifier (CMPT 419 ML Project)

This project classifies levels of helplessness in video data using multiple deep learning models, including a 2D CNN + LSTM, a 3D CNN, and a pre-trained SwinTransformer3D. It includes full preprocessing, training pipelines, and a live webcam-based GUI for real-time inference.

---

## 📁 Repository Structure

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
│   │   └── model_weights.pth            # Trained 3D CNN weights
│   │
│   ├── pre_trained_transformer_model/
│   │   └── model.py                     # SwinTransformer3D model (uses Torchvision's weights)
│   │
│   ├── dataset.py                       # Generic video dataset loader
│   ├── find_mean_std.py                # Utility to compute mean and std of dataset
│   └── training.ipynb                  # General training loop (can be adapted)
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
├── webcam_capture.py                    # Model inference + preprocessing pipeline
├── ETL.py                               # Extract-Transform-Load script to preprocess raw videos
├── confusion_matrix.py                  # Model evaluation using confusion matrix + metrics
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

Make sure you have Python 3.9+ and `pip` installed.

Install the packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

#### ➕ Git LFS Support

We use [Git Large File Storage](https://git-lfs.com/) for storing model weights and large files:

```bash
git lfs install
```

---

## Running the Application

### Train a Model

#### Option 1: Train the 2D CNN + LSTM (Grayscale)

```bash
jupyter notebook classifier_model/cnn_2d_model/CNN_LSTM_training.ipynb
```

Make sure your `train/` and `val/` sets are inside `data/`.

#### Option 2: Train 3D CNN

Use `classifier_model/training.ipynb` and point to `cnn_3d_model/model.py`.

---

### Preprocess Raw Videos (ETL)

This will convert `.mp4` or `.mov` videos into resized frames for training.

```bash
python ETL.py
```

Outputs are saved to `processed_frames/`.

---

### Evaluate Model (Confusion Matrix)

```bash
python confusion_matrix.py
```

This loads the 3D model (by default), runs on the validation set, and plots the confusion matrix with precision, recall, and F1.

---

### Run the Live GUI

```bash
python main.py
```

The GUI allows you to:

- Select between:
  - **2D CNN + LSTM (Grayscale)**
  - **3D CNN (RGB)**
  - **Pre-trained Swin Transformer (RGB)**
- See **real-time webcam feed**
- Display:
  - **Prediction label**
  - **Probabilities for each class**
  - **Inference time per window**

Press **`q`** to close the webcam feed.

---

## Model Architectures

### 2D CNN + LSTM

- Input: (B, T, 1, 112, 112)
- CNN: 3 conv blocks
- LSTM: 1 layer, hidden dim = 128
- Output: 3-class classification

### 3D CNN

- Input: (B, 3, T, 224, 224)
- 4 stacked 3D Conv blocks
- Final FC after global pooling

### SwinTransformer3D

- TorchVision pre-trained model
- Input: RGB sequence
- Output: 3-class logits

---

## Data Structure

### Raw Video Data

```
extreme-helpless/
little_helplessness/
no-helpless/
```

Each folder contains `.mp4` or `.mov` files.

---

### After Preprocessing

```
processed_frames/
├── extreme-helpless/
│   └── <video_name>/frame_000.jpg ...
├── little_helplessness/
├── no-helpless/
```

---

### Final Dataset Split

```
data/
├── train/
│   └── <class>/<video_name>/frame_*.jpg
├── val/
```

This is the data fed into the models.

---

## Author

**Ibrahim Rehman**  
**Greg Parent**
**Daniel Nguyen**  
Course: CMPT 419 — Machine Learning  
Simon Fraser University

---

## Notes

- All models use 3-class classification: `No`, `Little`, `Extreme` helplessness.
- You must have `ffmpeg` installed for MoviePy to work.
- Supports both macOS and Windows (with MPS or CUDA if available).

---
```
