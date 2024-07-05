# Automated PPE Violation Detection and Logging System

## Project Description
We created an Automated PPE Violation Detection and Logging System to enhance safety compliance on construction sites. This system uses AI to detect if individuals are wearing essential safety equipment such as helmets, masks, and safety vests. If any violations are detected, the system logs the incident and announces instructions via speakers to ensure immediate corrective action.

## Features
- *Real-time Detection:* Uses YOLO and OpenCV for real-time detection of PPE violations.
- *Logging:* Records detected violations with timestamps for review and analysis.
- *Audio Notifications:* Provides immediate audio instructions to individuals not wearing the required safety equipment.
- *Multiple PPE Categories:* Detects helmets, masks, and safety vests.

## Technologies Used
- *Python:* Core programming language.
- *YOLO (You Only Look Once):* Object detection model for identifying PPE.
- *OpenCV:* Library for real-time computer vision.
- *pyttsx3:* Text-to-speech conversion library for audio notifications.
- *pygame:* Library for playing audio files.

## System Architecture
1. *Video Capture:* Webcam captures real-time video feed.
2. *PPE Detection:* YOLO model processes frames to detect PPE compliance.
3. *Violation Logging:* Logs detected violations with timestamps.
4. *Audio Notification:* Announces instructions via speakers when a violation is detected.

## Installation
1. Clone the repository:
    bash
    git clone https://github.com/jatinRSP/PPEdetection
    cd CODE
    
2. Install the required packages:
    pip install -r requirements.txt
    
3. Download the YOLO model and place it in the project directory.

## Usage
1. Run the camera script for capturing video feed:
    python camera.py
    
2. The system will start capturing video from the webcam and this feed be used by another program.

3. Run the detection script which takes IP of camera feed as input and perform the PPE violation detection.
    python detection.py
   
5. Detected violations will be logged and announced via speakers.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.

## TEAM CODER OP:
- Kathan Shah
- Dev Nayak
- Jeet Shah
- Jatin Prajapati
