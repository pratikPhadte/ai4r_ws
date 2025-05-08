import cv2
from ultralytics import YOLO
import numpy as np

# Load the YOLOv8 model (pre-trained weights)
model = YOLO('yolov8n.pt')  # Nano model for faster inference; use 'yolov8s.pt' or others for better accuracy

# Initialize webcam
cap = cv2.VideoCapture(1)  # 0 is typically the default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Perform inference
        results = model(frame, classes=[0])  # Filter for class ID 0 (person)

        # Initialize list to store x-coordinates of detected persons
        x_coordinates = []

        # Process results
        for result in results:
            boxes = result.boxes  # Get bounding boxes
            for box in boxes:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integers
                conf = box.conf[0]  # Confidence score
                cls = int(box.cls[0])  # Class ID (should be 0 for person)

                # Calculate center x-coordinate
                center_x = (x1 + x2) // 2
                x_coordinates.append(center_x)

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
                label = f'Person: {conf:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display x-coordinates
        for i, x in enumerate(x_coordinates):
            print(f"Person {i+1} x-coordinate: {x}")

        # Display the frame with detections
        cv2.imshow('YOLOv8 Person Detection', frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()