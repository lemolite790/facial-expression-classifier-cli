import cv2
from interfaces import IFaceDetector

class HaarCascadeFaceDetector(IFaceDetector):
    SCALE_FACTOR = 1.1
    MIN_NEIGHBORS = 5
    MIN_SIZE = (40, 40)

    def __init__(self) -> None:
        super().__init__()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect_faces(self, image):
        grayscale = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            grayscale, 
            scaleFactor = self.SCALE_FACTOR, 
            minNeighbors = self.MIN_NEIGHBORS, 
            minSize = self.MIN_SIZE
        )
        return faces