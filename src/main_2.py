import threading
import time
import webcam_capture_2

webcam_thread = None
processing_thread = None

if __name__ == "__main__":
    try:
        webcam_thread = threading.Thread(target=webcam_capture_2.capture_frames, daemon=True)
        processing_thread = threading.Thread(target=webcam_capture_2.process_frames, daemon=True)

        webcam_thread.start()
        processing_thread.start()

        while True:
            time.sleep(1)
    except KeyboardInterrupt:  # CTRL+C to terminate program
        print("\nShutting down program")
        webcam_capture_2.stop_event.set()
        webcam_thread.join()
        processing_thread.join()