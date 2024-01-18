import cv2
import base64
import numpy as np
from PIL import Image
from io import BytesIO

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import keras

from flask_socketio import SocketIO
from flask import Flask, render_template, jsonify

app = Flask(__name__)
socketio = SocketIO(app)

EMOTIONS = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt', 'unknown', 'NF']
MODEL_FILE = "clfs//model_with_ferplus.h5" 
IMAGE_SHAPE = (48, 48, 1)

model = keras.models.load_model(MODEL_FILE)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def predict_and_overlay(frame):
    # face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40))

    # drawing lines on detected faces
    for idx, (x, y, width, height) in enumerate(faces, 1):
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0))
        
        # preprocessing
        img = frame[y:y+height, x:x+width]
        img = np.asarray(Image.fromarray(img).convert("L"))
        img = cv2.resize(img, IMAGE_SHAPE[:2])
        img = img.flatten()
        img = img.reshape(-1, *IMAGE_SHAPE)
        img = (img-127.5)/127.5

        # predict 
        y_pred = model.predict(img, verbose=False)
        y_idx = np.argmax(y_pred)
        y_label = EMOTIONS[y_idx]

        # show result
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.7
        color = (0, 255, 0)
        thickness = 1

        y_sorted = np.flip(y_pred.argsort())
        emotions = ' '.join([EMOTIONS[idx] for idx in y_sorted[0][:2]])
        cv2.putText(frame, f"face-{idx} : {emotions}", (x, y-10), font, font_scale, color, thickness, cv2.LINE_AA)

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
def start_web_app(port):
    socketio.run(app, port=port, debug=False)

if __name__ == '__main__':
    socketio.run(app, debug=True)
