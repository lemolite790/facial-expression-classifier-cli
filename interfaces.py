import numpy as np
from typing import List, Tuple
from abc import ABC, abstractmethod

class IModel(ABC):

    @abstractmethod
    def predict(self, image:np.ndarray) -> np.ndarray:
        """
        predict : predicts the label for multilabelclassification problem
        input   : original image
        output  : dictionary of (key, value) : (label : probability) in sorted order from high to low
        """
        pass


class IFaceDetector(ABC):

    @abstractmethod
    def detect_faces(self, image:np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        detect_faces : find faces from input images
        input      : original image
        output     : list of (x, y, width, height) for each face
        """
        pass