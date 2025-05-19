import cv2
from ultralytics import YOLO
import torch
from threading import Thread, Lock

person_model = YOLO('../yolov8n.pt') # Pretrained COCO
hat_model = YOLO('../hardhat.pt') # Fine-tuned Hardhat


cap = cv2.VideoCapture(0)  # Change to video path if not using webcam


REAL_HEIGHT = 1.7  # Real-world height of the person in meters
FOCAL_LENGTH = 800  # Approximate focal length in pixels (adjust for your camera)

depth_map = None
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)

frame_lock = Lock()
def depth_thread(crop):
    global depth_map

    if depth_map is not None:
        return depth_map

    with frame_lock:
        input_batch = midas_transforms(crop).to(device)
        with torch.no_grad():
            prediction = midas(input_batch)
            depth_map = prediction.squeeze().cpu().numpy()

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


        is_wearing_hat = False
        for hr in hat_results:
            hat_boxes = hr.boxes
            for hat_box in hat_boxes:
                cls = int(hat_box.cls[0])
                if cls == 1:
                    hx1, hy1, hx2, hy2 = map(int, hat_box.xyxy[0])
                    cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 0, 255), 3)
                    if (hx1 > px1 and hx2 < px2):
                        is_wearing_hat = True
                        break
        
        person_crop = frame[py1:py2, px1:px2]
        person_pixel_height = py2 - py1

        depth_t = Thread(target=depth_thread, args=(person_crop,))
        depth_t.start()
        depth_t.join()

        if person_pixel_height > 0 and depth_map is not None:
            metric_distance = (REAL_HEIGHT * FOCAL_LENGTH) / person_pixel_height
        else:
            metric_distance = 0

        label = "Person_with_Helmet" if is_wearing_hat else "Person"
        color = (0, 255, 0) if is_wearing_hat else (0, 0, 255)
        cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
        
        cv2.putText(frame, f"Distance: {metric_distance:.2f} m", 
            (px1, py1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.putText(frame, label, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
    
    cv2.imshow("Person with helmet Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
