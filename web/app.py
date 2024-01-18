import cv2
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from interfaces import IModel, IFaceDetector

from flask_socketio import SocketIO
from flask import Flask, render_template, jsonify

app = Flask(__name__)
socketio = SocketIO(app)

# default global variables
clf = None
face_detector = None

def predict_and_overlay(frame):
    # face detection
    faces = face_detector.detect_faces(frame)
    
    # drawing lines on detected faces
    for idx, (x, y, width, height) in enumerate(faces, 1):
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), thickness=1)
        result_dict = clf.predict(frame[x:x+width, y:y+height])
        # show 2 highest probability classes
        emotions = ", ".join(list(result_dict.keys())[:2])
        cv2.putText(frame, f"face-{idx} : {emotions}", (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
    
    return frame

@socketio.on('image')
def handle_image(data):
    # Decode base64 image data and convert to NumPy array
    image_data = data['image_data']
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    image = np.array(image)

    # Perform prediction and overlay result on the image
    result_image = predict_and_overlay(image)

    # Convert the modified image to base64 for sending to the frontend
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode('.jpg', result_image)
    result_image_data = base64.b64encode(buffer).decode('utf-8')

    # Send the modified image back to the frontend through WebSocket
    socketio.emit('result_image', {'result_image_data': result_image_data})

@app.route('/')
def index():
    return render_template('index.html')

# driver function
def start_web_app(port, custom_clf : IModel, custom_face_detector : IFaceDetector):
    global clf, face_detector
    clf = custom_clf
    face_detector = custom_face_detector

    socketio.run(app, port=port, debug=False)

if __name__ == '__main__':
    pass