# custom facial expression classifier class as utility class to handle models in various applications
import os
import cv2
import json
import numpy as np
from PIL import Image
from typing import List
from tensorflow import keras
from interfaces import IModel
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

class ModelNotFoundError(Exception):
    def __init__(self, model_name):
        self.model_name = model_name
        self.message = f"Error: Model '{model_name}' not found in the JSON file."

class FacialExpressionClassifier(IModel):

    # factory function
    @classmethod
    def from_json(cls, json_file:str, model_name) -> object:
        with open(json_file, "r") as f:
            obj = json.load(f)

        if model_name not in obj.keys(): 
            raise ModelNotFoundError(model_name)

        model_obj = obj[model_name]
        return cls(model_obj["modelPath"], model_obj["inputSize"], model_obj["labels"])

    def __init__(self, model_path:str, input_size:(int, int, int), labels:List[str]) -> None:
        self.model_path = model_path
        self.input_size = input_size
        self.labels = labels

        # load model
        self.__model = keras.models.load_model(model_path)

    # update this as per model
    def preprocess_image(self, image:np.ndarray) -> np.ndarray:
        image = np.asarray(Image.fromarray(image).convert("L"))
        image = cv2.resize(image, self.input_size[:2])
        image = image.flatten()
        image = image.reshape(-1, *self.input_size)
        image = (image-127.5)/127.5
        return image
    
    def predict(self, image:np.ndarray):
        
        image = self.preprocess_image(image)
        predictions = self.__model.predict(image, verbose=False)[0]

        indices = np.flip(np.argsort(predictions))

        ret = {}
        for idx in indices:
            ret[self.labels[idx]] = predictions[idx]
        return ret
    
    def __repr__(self) -> str:
        return f"Model(path={self.model_path}, input_size={self.input_size}, labels={self.labels})"
    

if __name__ == '__main__':

    model = FacialExpressionClassifier.from_json("model.json", "model_with_ferplus")
    print(model)
    print(model.predict(np.asarray(Image.open("smile.jpg"))))