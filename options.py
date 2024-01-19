import os
import imghdr
import argparse

def get_options():
    description = """A command line API to use facial expression classifier with different input modes."""

    parser = argparse.ArgumentParser("facial expression classifier", description=description)
    parser.add_argument("-m", "--mode", help="select mode", choices=["web", "gui", "image"], default="gui")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-f", "--image-file", help="select image file (for mode = 'image')", type=str, required=False)
    group.add_argument("-p", "--port", help="select port to host the web (for mode = 'web')", type=int, required=False, default=3108)
    group.add_argument("-s", "--window-size", help="select window size (width) (for mode = 'gui')", type=int, required=False, default=600)
    
    args = parser.parse_args()

    # validating
    if args.mode == 'image':
        if args.image_file is None:
            parser.error("\n\t> Missing image file argument for mode 'image'")
        elif not os.path.exists(args.image_file):
            parser.error(f"\n\t> Image file '{args.image_file}' not found")
        elif imghdr.what(args.image_file) is None:
            parser.error(f"\n\t> '{args.image_file}' can not be recognized as image file")

        
    return args

if __name__ == '__main__':
    print(get_options())