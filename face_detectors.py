from typing import List, Tuple
import cv2
import mediapipe as mp
from numpy import ndarray
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
    
class MediaPipeFaceDetector(IFaceDetector):

    def __init__(self) -> None:
        super().__init__()
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=0.3
        )
        
    def detect_faces(self, image: ndarray) -> List[Tuple[int, int, int, int]]:
        faces = []
        rgb_frame = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, width, height = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                faces.append((x, y, width, height))
        return faces
