# Library for continuously extracting webcam data using cv2 from OpenCV
# SOURCE: https://pypi.org/project/opencv-python/

import cv2
import threading
import time
import torch
import torch.nn.functional as F
import os
from classifier_model.cnn_2d_model.model import HelplessnessClassifier as GrayscaleClassifier
from classifier_model.cnn_3d_model.model import HelplessnessClassifier as RGBClassifier

# Fallback for using mps because some features are not included:
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

################################################
# Constants
################################################
model_2d_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "classifier_model",
    "cnn_2d_model",
    "grayscale_cnn_lstm.pth"
)
model_3d_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "classifier_model",
    "cnn_3d_model",
    "model_weights.pth"
)

MODEL_ONE = "2d_cnn"
MODEL_TWO = "3d_cnn"
NUM_FRAMES = 90
FRAMERATE = 30
PROCESS_DELAY = 2
CLASS_LABELS = [
    "No Helplessness",
    "Little Helplessness",
    "Extreme Helplessness"
]

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
# Functions
################################################
def set_model(model_choice):
    """Set and load the desired model choice"""
    global model_type, current_model

    model_type = model_choice
    if model_choice == MODEL_ONE:
        current_model = GrayscaleClassifier()
        model_path = model_2d_path
    else:
        current_model = RGBClassifier()
        model_path = model_3d_path

    # Load the corresponding model weights and set to evaluation mode
    current_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
    current_model.to(device)
    current_model.eval()

def preprocess_frames(frames):
    """Preprocess frames according to the selected model"""
    processed_frames = []

    if model_type == MODEL_ONE:
        # 2D CNN Transformations
        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (112, 112))
            processed_frames.append(frame)
        tensor_frames = torch.tensor(processed_frames, dtype=torch.float32).unsqueeze(1)
        return tensor_frames.unsqueeze(0).to(device)
    else:
        # 3D CNN Transformations
        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            processed_frames.append(frame)
        tensor_frames = torch.tensor(processed_frames, dtype=torch.float32).permute(3, 0, 1, 2) / 255.0
        return tensor_frames.unsqueeze(0).to(device)

################################################
# Thread to process frames
################################################
def process_frames(update_prediction_label):
    """Process frames using the selected model and return prediction"""
    global buffer

    while not stop_event.is_set():
        current_frames = []
        with buffer_lock:
            if buffer:
                current_frames = buffer.copy()
                buffer.clear()

        if current_frames:
            print(f"Processing the last 90 frames...")
            input_tensor = preprocess_frames(current_frames)

            with torch.no_grad():
                prediction = current_model(input_tensor)
                prediction = F.softmax(prediction, dim=1)

            predicted_class_idx = torch.argmax(prediction, dim=1).item()
            predicted_label = CLASS_LABELS[predicted_class_idx]

            confidence = prediction[0][predicted_class_idx]
            label = f"Prediction: {predicted_label} ({confidence:.3f} confidence)"
            print(label)
            update_prediction_label(label)

            time.sleep(PROCESS_DELAY)

################################################
# Thread to capture frames
################################################
def capture_frames():
    """Capture frames from live webcam feed"""
    global buffer
    capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        print("ERROR: Could not find a webcam")
        return

    try:
        while not stop_event.is_set():
            temp_buffer = []
            # Capture exactly 90 frames
            for _ in range(NUM_FRAMES):
                ret, frame = capture.read()
                if not ret:
                    print("WARNING: Failed to capture a frame")
                    break
                temp_buffer.append(frame)
                delay = 1.0 / FRAMERATE
                time.sleep(delay)

            with buffer_lock:
                buffer = temp_buffer
    finally:
        capture.release()
        cv2.destroyAllWindows()
