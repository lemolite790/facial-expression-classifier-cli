import cv2
import tkinter as tk
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
from interfaces import IModel, IFaceDetector

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


class FECApp(tk.Tk):
    
    def __init__(self, clf : IModel, face_detector : IFaceDetector, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("Facial Expression Classifier")
        self.geometry("600x600")
        self.clf = clf
        self.face_detector = face_detector

        self.model = None
        self.emotion = None
        self.img_shape = None
        self.vid = VideoDevice()
        self.main_pannel = tk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.main_pannel.pack()
        
        self.canvas = tk.Canvas(self, width=self.vid.width, height=self.vid.height, border=3)
        self.canvas.pack()

        self.delay = 15
        self.update()

        
    def update(self):
        ret, frame = self.vid.get_frame()
        if ret:
            
            faces = self.face_detector.detect_faces(frame)

            # drawing lines on detected box
            for idx, (x, y, width, height) in enumerate(faces, 1):
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), thickness=1)
                
                result_dict = self.clf.predict(frame[x:x+width, y:y+height])
                # show 2 highest probability classes
                emotions = ", ".join(list(result_dict.keys())[:2])
                cv2.putText(frame, f"face-{idx} : {emotions}", (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
                
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        
        self.after(self.delay, self.update)
        

if __name__ == '__main__':
    from classifiers import FacialExpressionClassifier
    from face_detectors import HaarCascadeFaceDetector
    clf = FacialExpressionClassifier.from_json("model.json", "model_with_ferplus")
    detecotr = HaarCascadeFaceDetector()
    app = FECApp(clf, detecotr)
    app.mainloop()
