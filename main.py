import cv2
import threading
import tkinter as tk
import sys
import time
import os
# MPS fallback for unimplemented ops:
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import torch.nn.functional as F
import numpy as np

from classifier_model.cnn_2d_model.model import HelplessnessClassifier as GrayscaleClassifier
from classifier_model.cnn_3d_model.model import HelplessnessClassifier as RGBClassifier
from classifier_model.pre_trained_transformer_model.model import create_swin3d_t_model_inference

############################################
# Constants and Globals
############################################
TITLE = "Helplessness Classifier"
SELECT_MODEL_LABEL = "Select Model:"
MODEL_ONE_LABEL = "2D CNN + LSTM (Grayscale)"
MODEL_TWO_LABEL = "3D CNN (RGB)"
MODEL_THREE_LABEL = "Pre-trained SwinTransformer (RGB)"
EXIT_LABEL = "Exit Program"
PREDICTION_LABEL = "Prediction: "
FONT = "Roboto"
MODEL_ONE = "2d_cnn"
MODEL_TWO = "3d_cnn"
MODEL_THREE = "pre_trained"

NUM_FRAMES = 90
FRAMERATE = 30
PROCESS_DELAY = 2
CLASS_LABELS = ["No Helplessness", "Little Helplessness", "Extreme Helplessness"]

model_2d_path = "classifier_model/cnn_2d_model/grayscale_cnn_lstm.pth"
model_3d_path = "classifier_model/cnn_3d_model/model_weights.pth"
model_pre_trained_path = "classifier_model/pre_trained_transformer_model/model_weights.pth"

buffer = []
buffer_lock = threading.Lock()
stop_event = threading.Event()
model_type = None
current_model = None
live_frame = None  # Global variable to store the most recent frame

# Device Detection
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device for acceleration.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device for acceleration.")
else:
    device = torch.device("cpu")
    print("Using CPU (no MPS or CUDA).")

############################################
# Model Setup and Frame Preprocessing
############################################
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
        device = torch.device("mps" if torch.backends.mps.is_available() else
                              "cuda" if torch.cuda.is_available() else "cpu")
    elif model_choice == MODEL_TWO:
        # 3D CNN for RGB
        current_model = RGBClassifier()
        model_path = model_3d_path
        device = torch.device("mps" if torch.backends.mps.is_available() else
                              "cuda" if torch.cuda.is_available() else "cpu")
    elif model_choice == MODEL_THREE:
        # Pre-trained SwinTransformer for RGB
        current_model = create_swin3d_t_model_inference()
        model_path = model_pre_trained_path
        device = torch.device("mps" if torch.backends.mps.is_available() else
                              "cuda" if torch.cuda.is_available() else "cpu")
    # Load weights and set model to eval mode
    current_model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    current_model.to(device)
    current_model.eval()
    print(f"Loaded {model_choice} model onto {device}.")

def preprocess_frames(frames):
    """
    Preprocess frames depending on the selected model.
    2D => convert to grayscale, resize to 112x112, normalize with mean=0.5 and std=0.5,
          reshape to (1, T, 1, 112,112)
    3D, Pre-trained => convert BGR to RGB, resize to 224x224, shape => (1, 3, T, 224,224)
    """
    global model_type, device
    if model_type == MODEL_ONE:
        processed = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (112, 112))  # resizing this to match training size
            processed.append(gray.astype(np.float32))
        arr = np.stack(processed, axis=0)       # shape: (T, 112, 112)
        # Normalize grayscale values: scale to [0, 1] then normalize with mean=0.5 and std=0.5
        arr = arr / 255.0
        arr = (arr - 0.5) / 0.5
        # Add channel dimension: (T, 1, 112, 112)
        arr = np.expand_dims(arr, axis=1)
        # Add batch dimension: (1, T, 1, 112, 112)
        arr = np.expand_dims(arr, axis=0)
        tensor_frames = torch.tensor(arr, dtype=torch.float32).to(device)
        return tensor_frames
    else: # also works for transformer model
        mean = np.array([0.41500069, 0.36530493, 0.33830512], dtype=np.float32)
        std  = np.array([0.29042152, 0.27499218, 0.27738131], dtype=np.float32)
        processed = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (224, 224))
            rgb = rgb.astype(np.float32) / 255.0
            rgb = (rgb - mean) / std
            processed.append(rgb)
        arr = np.stack(processed, axis=0)       # shape: (T, 224, 224, 3)
        arr = np.transpose(arr, (3, 0, 1, 2))     # shape: (3, T, 224, 224)
        arr = np.expand_dims(arr, axis=0)         # shape: (1, 3, T, 224, 224)
        tensor_frames = torch.tensor(arr, dtype=torch.float32).to(device)
        return tensor_frames

############################################
# Frame Processing and Capture Functions
############################################
def process_frames(update_prediction_label):
    """
    Process the frames in the buffer with the selected model,
    and update the GUI label via update_prediction_label().
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
            summary_str = f"Prediction: {pred_label} ({confidence:.3f} confidence)"
            detailed_str = (f"No Helplessness: {probs[0][0]:.3f}, "
                            f"Little Helplessness: {probs[0][1]:.3f}, "
                            f"Extreme Helplessness: {probs[0][2]:.3f} | "
                            f"Selected: {pred_label} ({confidence:.3f})")
            print(detailed_str)
            update_prediction_label(summary_str)

def capture_frames():
    """
    Continuously capture frames from the webcam (NUM_FRAMES at a time)
    and store them in a buffer for process_frames() to handle.
    The live frame is stored globally so that it can be shown by the main thread.
    """
    global buffer, live_frame
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
                # update the global live_frame
                live_frame = frame
                temp_buffer.append(frame)
                time.sleep(1.0 / FRAMERATE)
            with buffer_lock:
                buffer = temp_buffer
    finally:
        cap.release()

############################################
# Tkinter GUI and OpenCV Window Update
############################################
window = None
prediction_label = None
model_label = None
webcam_thread = None
processing_thread = None

def initialize_gui():
    """Initialize the Tkinter GUI and start the main loop."""
    global prediction_label, model_label, window
    window = tk.Tk()
    window.title(TITLE)
    window.geometry("500x300")
    
    model_label = tk.Label(window, text=SELECT_MODEL_LABEL, font=(FONT, 15, "bold"))
    model_label.pack(pady=(20, 10))
    
    # Frame for the buttons
    btn_frame = tk.Frame(window)
    btn_frame.pack(pady=10)
    
    btn_model_one = tk.Button(
        btn_frame,
        text=MODEL_ONE_LABEL,
        command=lambda: start_capture(MODEL_ONE),
        width=25
    )
    btn_model_two = tk.Button(
        btn_frame,
        text=MODEL_TWO_LABEL,
        command=lambda: start_capture(MODEL_TWO),
        width=25
    )
    btn_model_three = tk.Button(
        btn_frame,
        text=MODEL_THREE_LABEL,
        command=lambda: start_capture(MODEL_THREE),
        width=25
    )
    btn_exit = tk.Button(btn_frame, text=EXIT_LABEL, command=stop_capture, width=25)
    
    btn_model_one.grid(row=0, column=0, pady=5)
    btn_model_two.grid(row=1, column=0, pady=5)
    btn_model_three.grid(row=2, column=0, pady=5)
    btn_exit.grid(row=3, column=0, pady=5)
    
    prediction_label = tk.Label(window, text=PREDICTION_LABEL, font=(FONT, 12, "bold"))
    prediction_label.pack(pady=(20, 10))
    
    # Schedule the OpenCV window update to run in the main thread
    window.after(0, update_live_frame)
    window.mainloop()

def update_live_frame():
    """
    Called periodically from the Tkinter main loop to update the OpenCV window.
    This ensures cv2.imshow is always called from the main thread.
    """
    global live_frame
    if live_frame is not None:
        cv2.imshow('Live Webcam Feed', live_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_capture()
            return
    if not stop_event.is_set():
        window.after(10, update_live_frame)

def start_capture(model_type_choice):
    """Start the webcam capture and processing threads with the selected model."""
    global webcam_thread, processing_thread
    set_model(model_type_choice)
    if model_type_choice == MODEL_ONE:
        update_model_label(f"Model: {MODEL_ONE_LABEL}")
    elif model_type_choice == MODEL_TWO:
        update_model_label(f"Model: {MODEL_TWO_LABEL}")
    elif model_type_choice == MODEL_THREE:
        update_model_label(f"Model: {MODEL_THREE_LABEL}")
    
    if not webcam_thread or not webcam_thread.is_alive():
        webcam_thread = threading.Thread(target=capture_frames, daemon=True)
        webcam_thread.start()
    if not processing_thread or not processing_thread.is_alive():
        processing_thread = threading.Thread(target=process_frames, args=(update_prediction_label,), daemon=True)
        processing_thread.start()

def stop_capture():
    """Stop the capture and processing threads, destroy windows, and exit the program."""
    stop_event.set()
    if webcam_thread:
        webcam_thread.join()
    if processing_thread:
        processing_thread.join()
    cv2.destroyAllWindows()
    window.quit()
    sys.exit()

def update_prediction_label(prediction):
    """Update the prediction label on the GUI."""
    prediction_label.config(text=f"{prediction}")

def update_model_label(text):
    """Update the model label on the GUI."""
    model_label.config(text=f"{text}")

############################################
# Main
############################################
if __name__ == "__main__":
    initialize_gui()
