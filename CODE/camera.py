from flask import Flask, Response
import cv2

app = Flask(__name__)
video_capture = cv2.VideoCapture(0)  # Use 0 for the primary camera

def generate_frames():
    while True:
        success, frame = video_capture.read()  # Capture frame-by-frame
        # flip the frame
        frame = cv2.flip(frame, 1)
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "Go to /video_feed to see the video stream."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)