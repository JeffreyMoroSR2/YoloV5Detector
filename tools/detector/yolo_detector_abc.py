from typing import List, Tuple, Any
from abc import ABCMeta, abstractmethod
import numpy as np


class LPDetector(metaclass=ABCMeta):

    @abstractmethod
    def load_model(self):
        """
        method that set self._model
        :return: self
        """
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def get_car_plates_boxes(self, images: np.ndarray, images_to_det) -> Any:
        """
        method that returns image and bboxes on image
        :param images:
        :param images_to_det: np.ndarray
        :return: bboxes, orig_image
        """
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def predict(self, orig_image: np.ndarray) -> Tuple[Any, np.ndarray]:
        """
        method that predict objects on image
        :param orig_image: np.ndarray
        :return: bboxes
        """
        raise NotImplementedError("Method not implemented")
