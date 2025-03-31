# Library for continuously extracting webcam data using cv2 from OpenCV
# SOURCE: https://pypi.org/project/opencv-python/

# Imports
import cv2
import threading
import time
import torch
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'classifier_model', 'greg')))
from model import HelplessnessClassifier

# Constants
FRAMERATE = 30
PROCESS_DELAY = 2
NUM_FRAMES = 90
MODEL_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'classifier_model', 'greg', 'model_weights.pth')

buffer = []
buffer_lock = threading.Lock()
stop_event = threading.Event()

CLASS_LABELS = ["No Helplessness", "Little Helplessness", "Extreme Helplessness"]

# Load model and set to evaluation mode
model = HelplessnessClassifier()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')), strict=False)
model.eval()


def preprocess_frames(frames):
    """Convert frames to RGB, resize, and rearrange for 3D CNN"""
    processed_frames = []

    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        processed_frames.append(frame)

    # Convert to tensor
    tensor_frames = torch.tensor(processed_frames, dtype=torch.float32).permute(3, 0, 1, 2)
    tensor_frames /= 255.0

    return tensor_frames.unsqueeze(0)

def process_frames():
    """Feed the current frames into the model and return the prediction"""
    global buffer

    while not stop_event.is_set():
        current_frames = []
        with buffer_lock:
            if buffer:
                current_frames = buffer.copy()
                buffer.clear()

        if current_frames and len(current_frames) == 90:
            print("Processing 90 frames captured...")

            input_tensor = preprocess_frames(current_frames)

            with torch.no_grad():
                prediction = model(input_tensor)
                prediction = F.softmax(prediction, dim=1)

            predicted_class_idx = torch.argmax(prediction, dim=1).item()
            predicted_label = CLASS_LABELS[predicted_class_idx]

            print(f"Prediction: {predicted_label}, ({prediction[0][predicted_class_idx]:.4f} confidence)")
            time.sleep(PROCESS_DELAY)


def capture_frames():
    """Continuously capture exactly 90 frames from the webcam"""
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