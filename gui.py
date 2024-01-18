import os
import cv2
import tkinter as tk
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk


import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import keras

EMOTIONS = {
    2 : ['Happy', 'Sad'],
    7 : ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'],
    8 : ['surprise', 'fear', 'neutral', 'sad', 'disgust', 'contempt', 'happy', 'anger'],
    10 : ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt', 'unknown', 'NF']
}


class VideoDevice: 
    VIDEO_SOURCE = 0

    def __init__(self, video_source = None):
        
        if video_source is None:
            video_source = self.VIDEO_SOURCE
        # open the video source
        self.vid = cv2.VideoCapture(video_source)
        
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
        
        # get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)
    
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


class EmotionClassifier(tk.Tk):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("Emotion Classifier")
        self.geometry("600x600")

        self.model = None
        self.emotion = None
        self.img_shape = None
        self.vid = VideoDevice()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
        self.main_pannel = tk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.main_pannel.pack()
        
        self.canvas = tk.Canvas(self, width=self.vid.width, height=self.vid.height, border=3)
        self.canvas.pack()

        self.frame_show_box = tk.Frame(self)
        self.frame_show_box.pack(side=tk.TOP)
        self.flag_show_box = tk.IntVar()
        tk.Checkbutton(self.frame_show_box, variable=self.flag_show_box).pack(side=tk.LEFT)
        tk.Label(self.frame_show_box, text="Show box").pack()

        self.model_frame = tk.Frame(self)
        self.model_frame.pack()
        tk.Label(self.model_frame, text="Current Model : ").pack(side=tk.LEFT)
        self.model_name = tk.Label(self.model_frame, text=(self.model or "No model loaded"))
        self.model_name.pack(side=tk.LEFT)

        tk.Button(self, text="Select Model", command=self.select_model).pack()
        
        # default model
        self.load_model("clfs//model_with_ferplus.h5")
        self.flag_show_box.set(1)

        self.delay = 15
        self.update()

        
    def select_model(self):
        file = askopenfile('r', defaultextension=('h5', 'keras'), initialdir="clfs")
        self.load_model(file.name)
    
    def load_model(self, model_file):
        if not os.path.exists(model_file):
            print(f"model file {model_file} doesn't exist")
            return
        self.model = keras.models.load_model(model_file)
        self.emotion = EMOTIONS[
            self.model.layers[-1].output_shape[1]
        ]
        self.img_shape = self.model.layers[0].input_shape[1:]
        self.model_name.configure(text=model_file.split('/')[-1])
        self.model_name.update()
        
    def update(self):
        ret, frame = self.vid.get_frame()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            box = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

            # drawing lines on detected box
            for idx, (x, y, width, height) in enumerate(box, 1):
                if self.flag_show_box.get():
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0))
                
                if self.model is None : continue
                 # preprocessing
                img = frame[y:y+height, x:x+width]
                img = np.asarray(Image.fromarray(img).convert("L"))
                # cv2.imshow(f"face-{idx}", img)
                img = cv2.resize(img, self.img_shape[:2])
                # cv2.imshow(f"resized-{idx}", img)
                img = img.flatten()
                img = img.reshape(-1, *self.img_shape)
                img = (img-127.5)/127.5

                # predict 
                y_pred = self.model.predict(img, verbose=False)
                y_idx = np.argmax(y_pred)
                y_label = self.emotion[y_idx]

                # show result
                font = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.7
                color = (0, 255, 0)
                thickness = 1

                y_sorted = np.flip(y_pred.argsort())
                emotions = ' '.join([self.emotion[idx] for idx in y_sorted[0][:2]])
                # emotions = ' '.join([f'{EMOTIONS[idx]}:{y_pred[0][idx]:.2f}' for idx in y_sorted[0][:2]])
                cv2.putText(frame, f"face-{idx} : {emotions}", (x, y-10), font, font_scale, color, thickness, cv2.LINE_AA)
                # print(y_label, y_idx, y_pred[0][y_idx])

                
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        
        self.after(self.delay, self.update)
        

if __name__ == '__main__':
    app = EmotionClassifier()
    app.mainloop()
