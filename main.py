from options import get_options

from gui.app import FECApp
from web.app import start_web_app
from image.image import predict_facial_expression

from classifiers import FacialExpressionClassifier
from face_detectors import HaarCascadeFaceDetector, MediaPipeFaceDetector

face_detector = MediaPipeFaceDetector()
clf = FacialExpressionClassifier.from_json("model.json", "model_with_ferplus")

args = get_options()

if args.mode == 'web':
    start_web_app(args.port, clf, face_detector)

elif args.mode == 'gui':
    FECApp(clf, face_detector, args.window_size).mainloop()

elif args.mode == 'image':
    predict_facial_expression(args.image_file, clf, face_detector)
