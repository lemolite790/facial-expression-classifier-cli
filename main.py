from options import get_options
from web.app import start_web_app
from gui import EmotionClassifier as GUI
from image import predict_facial_expression

args = get_options()

if args.mode == 'web':
    start_web_app(args.port)

elif args.mode == 'gui':
    GUI().mainloop()

elif args.mode == 'image':
    predict_facial_expression(args.image_file)
