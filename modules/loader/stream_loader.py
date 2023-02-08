import cv2
import time
import threading
import numpy as np

from objects.source import Source
from typing import List
from modules.loader.stream_loader_abc import SL

from libs.yolov5.utils.augmentations import letterbox
from libs.yolov5.utils.general import check_img_size


class StreamLoader(SL):
    def __init__(self, sources: List[Source]):
        self._sources = sources

        img_size = (320, 320)
        self._stride = 32
        self._imgsz = check_img_size(img_size, s=self._stride)
        self._auto = False

    def __len__(self):
        return len(self._sources)

    @staticmethod
    def init_capture(source: Source):
        cap = cv2.VideoCapture(source.stream)
        while True:
            if not cap.isOpened():
                print(f'Cap {source.stream_id} is not opened, re-opening')
                time.sleep(5)
                cap = cv2.VideoCapture(source.stream)
                continue
            else:
                return cap

    def start(self):
        for source in self._sources:
            threading.Thread(target=self.update, args=(source,), daemon=True).start()

        return self

    def update(self, source: Source):
        cap = self.init_capture(source)
        while True:
            success, frame = cap.read()
            if not success:
                time.sleep(5)
                cap = self.init_capture(source)
                continue
            source.image = frame

    @property
    def streams(self):
        ids_list = []
        images_list = []
        images_to_detect_list = []
        for source in self._sources:
            if isinstance(source.image, np.ndarray):
                ids_list.append(source.stream_id)
                images_list.append(source.image)
                images_to_detect_list.append(letterbox(source.image, self._imgsz, stride=self._stride,
                                                       auto=self._auto)[0])

        return ids_list, images_list, images_to_detect_list
