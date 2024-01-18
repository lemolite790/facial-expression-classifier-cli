# Imports
import numpy as np
from PIL import Image
from interfaces import IModel, IFaceDetector

def predict_facial_expression(source_image:np.ndarray, clf:IModel, face_detector:IFaceDetector):

    # open image
    image = Image.open(source_image)
    image = np.asarray(image).copy()

    # detect faces
    faces = face_detector.detect_faces(image)

    # drawing lines on detected box
    for idx, (x, y, width, height) in enumerate(faces, 1):
        result_dict = clf.predict(image[x:x+width, y:y+height])
        # show 2 highest probability classes
        emotions = ", ".join(list(result_dict.keys())[:2])
        print(f"face-{idx} : {emotions}")

if __name__ == '__main__':
    from face_detectors import HaarCascadeFaceDetector
    from facial_expression_classifier import FacialExpressionClassifier
    detector = HaarCascadeFaceDetector()
    clf = FacialExpressionClassifier.from_json("model.json", "model_with_ferplus")
    predict_facial_expression(".tmp//smile.jpg", clf, detector)