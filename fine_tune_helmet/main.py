import cv2
from ultralytics import YOLO

# Load both models
person_model = YOLO('../yolov8n.pt')          # Pretrained COCO
hat_model = YOLO('../hardhat.pt')  # Fine-tuned Hardhat

# Initialize Video Capture
cap = cv2.VideoCapture(0)  # Change to video path if not using webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect persons
    person_results = person_model.predict(frame, classes=[0], conf=0.5)
    
    # Detect hats and helmets
    hat_results = hat_model.predict(frame, conf=0.3)

    # Draw bounding boxes
    for person in person_results[0].boxes:
        px1, py1, px2, py2 = map(int, person.xyxy[0])
        person_box = (px1, py1, px2, py2)

        print(f"Person box: {person_box}")

        # Check if any hat/helmet is inside the person box
        is_wearing_hat = False
        for hat in hat_results[0].boxes:
            
            hx1, hy1, hx2, hy2 = map(int, hat.xyxy[0])
            print(f"Hat box: {hx1, hy1, hx2, hy2}")
            if (hx1 > px1 and hy1 < py1 and hx2 < px2 and hy2 > py2):
                is_wearing_hat = True
                break

        # Draw on frame
        label = "Person_with_Helmet" if is_wearing_hat else "Person_without_Helmet"
        color = (0, 255, 0) if is_wearing_hat else (0, 0, 255)
        cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
        cv2.putText(frame, label, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imshow("Helmet Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
