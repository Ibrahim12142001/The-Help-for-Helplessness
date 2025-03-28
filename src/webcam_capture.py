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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'classifier_model')))
from model import HelplessnessClassifier

# Constants
CAPTURE_TIME = 3
FRAMERATE = 30
PROCESS_DELAY = 2
NUM_CLASSES = 3
MODEL_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'classifier_model', 'grayscale_cnn_lstm.pth')

buffer = []
buffer_lock = threading.Lock()
stop_event = threading.Event()

# Load model and set to evaluation mode
model = HelplessnessClassifier(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()

def preprocess_frames(frames):
    """Convert frames to grayscale and resize"""
    processed_frames = []

    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (112, 112))
        processed_frames.append(frame)

    tensor_frames = torch.tensor(processed_frames, dtype=torch.float32).unsqueeze(1)
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

        if current_frames:
            print("Processing frames captured in last 3 seconds...")
            input_tensor = preprocess_frames(current_frames)

            with torch.no_grad():
                prediction = model(input_tensor)
                prediction = F.softmax(prediction, dim=1)

            print(f"Prediction: {prediction}")
            time.sleep(PROCESS_DELAY)


def capture_frames():
    """Continuously capture frames from webcam every three seconds"""
    global buffer
    capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        print("ERROR: Could not find a webcam")
        return

    try:
        while not stop_event.is_set():
            start_time = time.time()
            temp_buffer = []

            # Capture three seconds worth of frames
            while time.time() - start_time < CAPTURE_TIME:
                ret, frame = capture.read()
                if not ret:
                    print("WARNING: Failed to capture a frame")
                    break
                temp_buffer.append(frame)

                delay = 1.0 / FRAMERATE
                time.sleep(delay)

            # Update global buffer with current frames
            with buffer_lock:
                buffer = temp_buffer
    finally:
        capture.release()
        cv2.destroyAllWindows()