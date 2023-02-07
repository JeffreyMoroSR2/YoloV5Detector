from typing import List, Tuple, Any
from abc import ABCMeta, abstractmethod
from yolo_detector_abc import LPDetector
from detection_obj import DetectionObject
from Centroid.centroidtracker import CentroidTracker
import torch
import logging
import numpy as np
from typing import List, Any

from yolov5.utils.augmentations import letterbox  # find it in yolov5 directory
from yolov5.utils.torch_utils import select_device  # find it in yolov5 directory
from yolov5.models.common import DetectMultiBackend  # find it in yolov5 directory
from yolov5.utils.general import non_max_suppression, scale_boxes, xyxy2xywh, check_img_size  # find it in yolov5


# directory


class YoloDetector(LPDetector):
    def __init__(self, model_path, device='cuda:0', img_size=(320, 320), stride=32, auto=False, augment=False,
                 visualize=False, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                 labels=(), max_det=300, nm=0):
        self._model = None
        self._weights = model_path
        self._data = 'yolov5/data/coco128.yaml'

        self._device = select_device(device)
        self._imgsz = check_img_size(img_size, s=stride)

        self._stride = stride
        self._auto = auto
        self._augment = augment
        self._visualize = visualize

        self._conf_thres = conf_thres
        self._iou_thres = iou_thres
        self._classes = classes
        self._agnostic = agnostic
        self._multi_label = multi_label
        self._labels = labels
        self._max_det = max_det
        self._nm = nm  # Number of masks

    def load_model(self):
        self._model = DetectMultiBackend(self._weights, device=self._device, data=self._data)
        self._model.warmup(imgsz=(1, 3, *self._imgsz))
        return self

    def get_car_plates_boxes(self, orig_images, images_to_detect: List[np.ndarray]) -> Any:
        detection, images = self.predict(images_to_detect)
        result = []
        for i, det in enumerate(detection):
            im0 = orig_images[i].copy()
            det_objects = []
            if len(det):
                det[:, :4] = scale_boxes(images.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    xyxy = (torch.tensor(xyxy).view(1, 4)).view(-1).tolist()
                    xyxy = [int(x) for x in xyxy]
                    det_objects.append(CentroidTracker().update([DetectionObject(bbox=xyxy, id_=int(cls))]))
                    # det_objects.append(DetectionObject(bbox=xyxy, id_=int(cls)))

            result.append(det_objects)
        return result

    def predict(self, images: List[np.ndarray]) -> Tuple[Any, np.ndarray]:
        images = np.stack(images, 0)
        images = images[..., ::-1].transpose((0, 3, 1, 2))
        images = np.ascontiguousarray(images)
        images = torch.from_numpy(images).to(self._device)
        images = images.float()
        images /= 255

        if len(images.shape) == 3:
            images = images[None]

        pred = self._model(images, augment=self._augment, visualize=self._visualize)
        pred = non_max_suppression(pred, self._conf_thres, self._iou_thres, self._classes, self._agnostic,
                                   self._max_det)

        return pred, images
