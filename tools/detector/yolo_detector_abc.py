from typing import List, Tuple, Any
from abc import ABCMeta, abstractmethod
import numpy as np


class LPDetector(metaclass=ABCMeta):

    @abstractmethod
    def start(self):
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

    @staticmethod
    @abstractmethod
    def draw_boxes(frame_to_draw, detections):
        """
        method to draw boxes on a frame
        :param frame_to_draw: np.ndarray
        :param detections: list
        :return: np.ndarray
        """
        raise NotImplementedError("Method not implemented")
