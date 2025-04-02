import cv2
import threading
import time
import os
# MPS fallback for unimplemented ops:
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import torch.nn.functional as F

import numpy as np

from classifier_model.cnn_2d_model.model import HelplessnessClassifier as GrayscaleClassifier
from classifier_model.cnn_3d_model.model import HelplessnessClassifier as RGBClassifier


################################################
# Constants / Paths
################################################
model_2d_path = os.path.join(
    os.path.dirname(__file__),
    "..", "classifier_model", "cnn_2d_model", "grayscale_cnn_lstm.pth"
)
model_3d_path = os.path.join(
    os.path.dirname(__file__),
    "..", "classifier_model", "cnn_3d_model", "model_weights.pth"
)

MODEL_ONE = "2d_cnn"
MODEL_TWO = "3d_cnn"
NUM_FRAMES = 90
FRAMERATE = 30
PROCESS_DELAY = 2
CLASS_LABELS = ["No Helplessness", "Little Helplessness", "Extreme Helplessness"]

buffer = []
buffer_lock = threading.Lock()
stop_event = threading.Event()
model_type = None
current_model = None

################################################
# Device Detection
################################################
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device for acceleration.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device for acceleration.")
else:
    device = torch.device("cpu")
    print("Using CPU (no MPS or CUDA).")

################################################
# set_model
################################################
def set_model(model_choice):
    """
    Select and load the desired model. 
    MODEL_ONE => grayscale 2D CNN + LSTM
    MODEL_TWO => 3D CNN
    """
    global model_type, current_model, device

    model_type = model_choice
    if model_choice == MODEL_ONE:
        # 2D CNN-LSTM for grayscale
        current_model = GrayscaleClassifier()
        model_path = model_2d_path

        # Let MPS or CUDA handle it if possible
        device = torch.device("mps" if torch.backends.mps.is_available() else
                              "cuda" if torch.cuda.is_available() else "cpu")
    else:
        # 3D CNN for RGB
        current_model = RGBClassifier()
        model_path = model_3d_path

        # Typically 3D CNN may have partial MPS support. We still attempt MPS
        # but some layers might fallback to CPU. 
        device = torch.device("mps" if torch.backends.mps.is_available() else
                              "cuda" if torch.cuda.is_available() else "cpu")

    # Load weights
    current_model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    current_model.to(device)
    current_model.eval()
    print(f"Loaded {model_choice} model onto {device}.")

################################################
# preprocess_frames
################################################
def preprocess_frames(frames):
    """
    Preprocess frames depending on which model was set:
      2D => convert to grayscale, resize 112x112, shape => (1, T, 1, 112,112)
      3D => convert BGR->RGB, resize 224x224, shape => (1, 3, T, 224,224)
    """
    global model_type, device

    if model_type == MODEL_ONE:
        # 2D CNN (grayscale)
        processed = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (112, 112))  # match training
            # float in [0..255]
            processed.append(gray.astype(np.float32))

        # shape => (T, H, W)
        arr = np.stack(processed, axis=0)  # => (T, 112, 112)
        # add channel => (T, 1, 112, 112)
        arr = np.expand_dims(arr, axis=1)
        # add batch => (1, T, 1, 112, 112)
        arr = np.expand_dims(arr, axis=0)
        # to tensor
        tensor_frames = torch.tensor(arr, dtype=torch.float32).to(device)
        return tensor_frames

    else:
        # 3D CNN (RGB)
        # If you want color normalization like your original code:
        mean = np.array([0.41500069, 0.36530493, 0.33830512], dtype=np.float32)
        std  = np.array([0.29042152, 0.27499218, 0.27738131], dtype=np.float32)

        processed = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (224, 224))
            rgb = rgb.astype(np.float32) / 255.0
            # Normalize each channel
            rgb = (rgb - mean) / std
            processed.append(rgb)

        # Now shape => (T, 224,224,3)
        arr = np.stack(processed, axis=0)  # => (T,224,224,3)
        # reorder => (3, T, 224, 224)
        arr = np.transpose(arr, (3, 0, 1, 2))
        # add batch => (1, 3, T, 224, 224)
        arr = np.expand_dims(arr, axis=0)
        tensor_frames = torch.tensor(arr, dtype=torch.float32).to(device)
        return tensor_frames

################################################
# process_frames
################################################
def process_frames(update_prediction_label):
    """
    Processes the frames in buffer with the selected model,
    updates the GUI label via update_prediction_label().
    """
    global buffer
    while not stop_event.is_set():
        current_frames = []
        with buffer_lock:
            if buffer:
                current_frames = buffer.copy()
                buffer.clear()

        if current_frames:
            print(f"Processing last {len(current_frames)} frames with {model_type} model...")
            input_tensor = preprocess_frames(current_frames)
            with torch.no_grad():
                outputs = current_model(input_tensor)
                probs = F.softmax(outputs, dim=1)

            pred_idx = torch.argmax(probs, dim=1).item()
            pred_label = CLASS_LABELS[pred_idx]
            confidence = probs[0][pred_idx].item()
            label_str = f"Prediction: {pred_label} ({confidence:.3f} confidence)"
            print(label_str)
            # Update the GUI or console
            update_prediction_label(label_str)

            time.sleep(PROCESS_DELAY)

################################################
# capture_frames
################################################
def capture_frames():
    """
    Continuously capture frames from webcam (90 frames),
    store them in buffer for process_frames() to handle.
    """
    global buffer
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not find a webcam.")
        return

    try:
        while not stop_event.is_set():
            temp_buffer = []
            for _ in range(NUM_FRAMES):
                ret, frame = cap.read()
                if not ret:
                    print("WARNING: Failed to capture a frame")
                    break
                temp_buffer.append(frame)
                time.sleep(1.0 / FRAMERATE)

            with buffer_lock:
                buffer = temp_buffer
    finally:
        cap.release()
        cv2.destroyAllWindows()
