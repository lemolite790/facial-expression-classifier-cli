# Imports
import cv2
import keras
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

def predict_facial_expression(img):

    # face detector
    model = keras.models.load_model('clfs//model_with_ferplus.h5')
    # LABLES = ['Happy', 'Sad']
    LABLES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    LABLES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    LABLES = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt', 'unknown', 'NF']

    # open method used to open different extension image file 
    img = Image.open(img)  
    img = np.array(img)[:, :, 1]

    img = cv2.resize(img, (48, 48))
    img = img.flatten()
    img = img.reshape(-1,48,48,1)
    img = (img-127.5)/127.5
    y_pred = model.predict(img, verbose=False)
    y_label = np.argmax(y_pred[0])

    print("Predicted Label :", LABLES[y_label])
    # detailed 
    print()
    for idx in np.flip(np.argsort(y_pred[0])):
        print(f"{LABLES[idx]:<10s} : {y_pred[0][idx] * 100:.2f}%")
    