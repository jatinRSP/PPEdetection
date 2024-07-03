import cv2
import logging
from datetime import datetime, timedelta
import math
import pyttsx3  # Import pyttsx3 for text-to-speech

from ultralytics import YOLO
import cvzone 

# Uncomment this line to use the webcam
cap = cv2.VideoCapture(0)  # 0 is the default ID for the built-in webcam
cap.set(3, 1280)  # Set the width
cap.set(4, 720)  # Set the height
# Set the FPS to 25 fps
cap.set(cv2.CAP_PROP_FPS, 30)

model = YOLO("./ppe.pt")
classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

log_filename = None
logger = None
last_log_time = datetime.now()

# Initialize pyttsx3 engine
engine = pyttsx3.init()

def setup_logging(log_filename):
    handler = logging.FileHandler(log_filename)
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def speak_notification(labels):
    message = ""
    for label in labels:
        message += f"{label}\n"
    engine.say(message)
    engine.runAndWait()

while True:
    # Capture webcam feed using OpenCV
    success, img = cap.read()
    # flip the image vertically
    img = cv2.flip(img, 1)
    if not success:
        print("Failed to read frame from webcam")
        break

    results = model(img, stream=True)
    log_labels = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            if cls >= len(classNames) or cls in [0, 1, 5, 6, 7, 9]:
                continue  # Skip classes that are not in classNames

            currentClass = classNames[cls]

            if currentClass in ['NO-Safety Vest', 'NO-Hardhat', 'NO-Mask']:
                if conf > 0.5:
                    log_labels.append(f"{currentClass} {conf}")

            elif currentClass in ['Mask', 'Hardhat', 'Safety Vest']:
                if conf > 0.5:
                    speak_notification([f"Wear {currentClass}"])

            if currentClass in ['NO-Safety Vest', 'NO-Hardhat', 'NO-Mask', 'Hardhat', 'Safety Vest', 'Mask']:
                if conf > 0.5:
                    if currentClass in ['NO-Hardhat', 'NO-Safety Vest', 'NO-Mask']:
                        myColor = (0, 0, 255)
                    elif currentClass in ['Hardhat', 'Safety Vest', 'Mask']:
                        myColor = (0, 255, 0)
                    else:
                        myColor = (255, 0, 0)

                    cvzone.putTextRect(img, f'{currentClass} {conf}',
                                       (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                       colorT=(255, 255, 255), colorR=myColor, offset=5)
                    cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

    # Check if it's time to update the log file
    current_time = datetime.now()
    if last_log_time + timedelta(minutes=2) <= current_time or log_filename is None:
        last_log_time = current_time
        today = current_time.date()
        log_filename = f"log_{today}.txt"
        if logger is None:
            logger = setup_logging(log_filename)
        else:
            logger.handlers[0].close()
            logger.removeHandler(logger.handlers[0])
            logger = setup_logging(log_filename)

        if log_labels:
            logger.info(f"{current_time.strftime('%Y-%m-%d %H:%M:%S')}\n" + "\n".join(log_labels))

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()