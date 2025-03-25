# Library for continuously extracting webcam data using cv2 from OpenCV
# SOURCE: https://pypi.org/project/opencv-python/

# Imports
import cv2
import threading
import time

# Constants
CAPTURE_TIME = 3
FRAMERATE = 30
PROCESS_DELAY = 2

buffer = []
buffer_lock = threading.Lock()
stop_event = threading.Event()
# TODO: integrate model for prediction of frames
# model = load_model()

def process_frames():
    global buffer

    while not stop_event.is_set():
        current_frames = []
        with buffer_lock:
            if buffer:
                current_frames = buffer.copy()
                buffer.clear()

        if current_frames:
            print("Processing frames captured in last 3 seconds...")
            # prediction = model.predict(current_frames)
            print("Prediction: N/A")
            time.sleep(PROCESS_DELAY)


def capture_frames():
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