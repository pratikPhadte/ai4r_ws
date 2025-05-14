import cv2
import torch
import numpy as np
from ultralytics import YOLO
from threading import Thread, Lock

# Load YOLOv8n (Nano) model
model = YOLO('../yolov8n.pt', verbose=False)

# Load MiDaS depth estimation model
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)

# Known parameters
REAL_HEIGHT = 1.7  # Real-world height of the person in meters
FOCAL_LENGTH = 800  # Approximate focal length in pixels (adjust for your camera)
RESIZE_WIDTH = 320
RESIZE_HEIGHT = 240

# Multi-threading management
frame_lock = Lock()
yolo_results = None
depth_map = None

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

def yolo_thread(frame):
    global yolo_results
    with frame_lock:
        resized_frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
        yolo_results = model(resized_frame)

def depth_thread(crop):
    global depth_map
    with frame_lock:
        input_batch = midas_transforms(crop).to(device)
        with torch.no_grad():
            prediction = midas(input_batch)
            depth_map = prediction.squeeze().cpu().numpy()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Start YOLO thread
    yolo_t = Thread(target=yolo_thread, args=(frame,))
    yolo_t.start()
    yolo_t.join()

    # Flag to check if one person is detected
    person_detected = False

    if yolo_results is not None:
        for result in yolo_results:
            for detection in result.boxes:
                if detection.cls == 0:  # Class 0 corresponds to 'person'
                    bbox = detection.xyxy[0]
                    x1, y1, x2, y2 = map(int, bbox)

                    # Scale bounding box back to original resolution
                    x1 = int(x1 * frame.shape[1] / RESIZE_WIDTH)
                    y1 = int(y1 * frame.shape[0] / RESIZE_HEIGHT)
                    x2 = int(x2 * frame.shape[1] / RESIZE_WIDTH)
                    y2 = int(y2 * frame.shape[0] / RESIZE_HEIGHT)

                    # Get the height of the bounding box
                    pixel_height = y2 - y1
                    
                    # Crop the region of the person for depth estimation
                    person_crop = frame[y1:y2, x1:x2]

                    # Start MiDaS thread
                    depth_t = Thread(target=depth_thread, args=(person_crop,))
                    depth_t.start()
                    depth_t.join()

                    # **Convert to metric distance:**
                    if pixel_height > 0 and depth_map is not None:
                        metric_distance = (REAL_HEIGHT * FOCAL_LENGTH) / pixel_height
                    else:
                        metric_distance = 0

                    # Draw the bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Display the metric distance
                    cv2.putText(frame, f"Distance: {metric_distance:.2f} m", 
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                2, (0, 255, 0), 2)

                    person_detected = True
                    break
            if person_detected:
                break

    # Show the annotated frame
    cv2.imshow('Optimized YOLO + Depth Estimation', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
