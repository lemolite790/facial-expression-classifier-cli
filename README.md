# Facial Expression Classifier CLI

A command line API to use facial expression classifier with different input modes.

## Installation

    git clone https://github.com/lemolite790/facial-expression-classifier-cli
    cd facial-expression-classifier-cli
    pip install -r requirements.txt


## Usage
```bash
facial_expression_classifier [-h] [-m {web,gui,image}] [-f IMAGE_FILE | -p PORT | -s WINDOW_SIZE]
```

## Options

- `-h, --help`: Show help message and exit.

- `-m {web,gui,image}, --mode {web,gui,image}`: Select input mode. default : gui

- `-f IMAGE_FILE, --image-file IMAGE_FILE`: Select image file (for mode = 'image').

- `-p PORT, --port PORT`: Select port to host the web (for mode = 'web'). default : 8080

- `-s WINDOW_SIZE, --window-size WINDOW_SIZE`: Select window size (width) (for mode = 'gui'). default : 640

## Example Usages

1. To use the facial expression classifier in web mode on port 5000: 
```bash    
    python main.py -m web -p 5000
```

2. To use the facial expression classifier in GUI mode with a specific window size: 
```bash    
    python main.py -m gui -s 800
```

3. To use the facial expression classifier on a specific image file: 
```bash    
    python main.py -m image -f path/to/image.jpg
```
