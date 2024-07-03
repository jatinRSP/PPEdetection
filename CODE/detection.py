import streamlit as st
import cv2
import logging
from datetime import datetime, timedelta
import math
import pygame  # Import pygame for audio playback

from ultralytics import YOLO
import cvzone

# Initialize pygame for audio playback
pygame.mixer.init()

# Audio file paths
audio_files = {
    'NO-Hardhat': './AUDIO/helmet.mp3',
    'NO-Mask': './AUDIO/mask.mp3',
    'NO-Safety': './AUDIO/west.mp3'
}

log_filename = None
logger = None
last_log_time = datetime.now()

def setup_logging(log_filename):
    handler = logging.FileHandler(log_filename)
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def play_audio_notification(labels):
    for label in labels:
        if label in audio_files:
            pygame.mixer.music.load(audio_files[label])
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():  # Wait for the audio file to finish playing
                continue

def run_detection(video_stream_url):
    global log_filename, logger, last_log_time

    cap = cv2.VideoCapture(video_stream_url)
    cap.set(3, 1280)  # Set the width
    cap.set(4, 720)  # Set the height

    model = YOLO("./ppe.pt")
    classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

    stop_button_key = "stop_button"

    # Create a placeholder for the video frame
    video_placeholder = st.empty()

    while True:
        success, img = cap.read()
        if not success:
            st.error("Failed to read frame from webcam")
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
                        play_audio_notification([f"Wear {currentClass}"])

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

        current_time = datetime.now()
        if last_log_time + timedelta(minutes=1) <= current_time or log_filename is None:
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
                set1 = set()
                logger.info(f"{current_time.strftime('%Y-%m-%d %H:%M:%S')}," + ",".join(log_labels))
                for i in log_labels:
                    splitdata = i.split()
                    set1.add(splitdata[0])

                if set1:
                    play_audio_notification(list(set1))
                set1.clear()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        video_placeholder.image(img)
        # if st.button('Stop', key=stop_button_key):
        #     break

    cap.release()
    cv2.destroyAllWindows()

def main():
    st.title("PPE Detection System")
    video_stream_url = st.text_input("Enter video stream URL:")
    video_stream_url = "http://" + video_stream_url + ":5000/video_feed"
    if st.button("Connect"):
        run_detection(video_stream_url)

if __name__ == "__main__":
    main()