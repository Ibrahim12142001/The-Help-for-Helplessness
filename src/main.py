from webcam_capture import capture_frames, stop_event, process_frames, set_model
import threading
import tkinter as tk

# Label Constants
TITLE = "Helplessness Classifier"
SELECT_MODEL_LABEL = "Select Model:"
MODEL_ONE_LABEL = "2D CNN + LSTM (Grayscale)"
MODEL_TWO_LABEL = "3D CNN (RGB)"
STOP_LABEL = "Stop Program"
PREDICTION_LABEL = "Prediction: "
FONT = "Roboto"
MODEL_ONE = "2d_cnn"
MODEL_TWO = "3d_cnn"

webcam_thread = None
processing_thread = None

def initialize_gui():
    """Initialize GUI to facilitate program"""
    # SOURCE: https://realpython.com/python-gui-tkinter/
    global prediction_label, model_label

    # Initialize tkinter window
    window = tk.Tk()
    window.title(TITLE)
    window.geometry("500x300")
    model_label = tk.Label(window, text=SELECT_MODEL_LABEL, font=(FONT, 15, "bold"))
    model_label.pack(pady=(20, 10))

    # Display button options
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
    btn_stop = tk.Button(btn_frame, text=STOP_LABEL, command=stop_capture, width=25)
    btn_model_one.grid(row=0, column=0, pady=5)
    btn_model_two.grid(row=1, column=0, pady=5)
    btn_stop.grid(row=2, column=0, pady=5)

    # Display prediction result:
    prediction_label = tk.Label(window, text=PREDICTION_LABEL, font=(FONT, 12, "bold"))
    prediction_label.pack(pady=(20, 10))

    window.mainloop()

def start_capture(model_type):
    """Start webcam capture and processing threads with selected model"""
    global webcam_thread, processing_thread

    # Signal program to load desired model
    set_model(model_type)
    if model_type == MODEL_ONE:
        update_model_label(f"Model: {MODEL_ONE_LABEL}")
    elif model_type == MODEL_TWO:
        update_model_label(f"Model: {MODEL_TWO_LABEL}")

    # Start threads if not already started
    if not webcam_thread or not webcam_thread.is_alive():
        webcam_thread = threading.Thread(target=capture_frames, daemon=True)
        webcam_thread.start()
    if not processing_thread or not processing_thread.is_alive():
        processing_thread = threading.Thread(target=process_frames, args=(update_prediction_label,), daemon=True)
        processing_thread.start()

def stop_capture():
    """Stop webcam capture and processing threads"""
    stop_event.set()
    if webcam_thread:
        webcam_thread.join()
    if processing_thread:
        processing_thread.join()

def update_prediction_label(prediction):
    """Update prediction label to the GUI screen"""
    prediction_label.config(text=f"{prediction}")

def update_model_label(model_type):
    """Update model label to the GUI screen"""
    model_label.config(text=f"{model_type}")

if __name__ == "__main__":
    initialize_gui()
