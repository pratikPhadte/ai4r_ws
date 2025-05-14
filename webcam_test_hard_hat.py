# source from https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993
# Edited

from ultralytics import YOLO
import cv2
import math 

# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# model
model = YOLO("Hard_hat_model.pt")

# object classes for labelling
classNames = ['head', 'helmet']


while True:
    success, img = cap.read()
    
    # Only save results of the 'person' class
    #results = model(img, stream=True, conf=0.7, iou=0.5)
    results = model(img, stream=True)
    # conf=0.7 to only include boxes where the model has a confidence of 0.7 or higher
    # iou=0.5 to only include boxes that overlap for over half with the object they track/detect
    # classes=[0] to only include detections of the class 'person'

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()