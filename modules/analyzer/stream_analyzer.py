import cv2
from objects.source import Source

from tools.detector.yolo_detect import YoloDetector
from libs.centroid.centroidtracker import CentroidTracker
from modules.loader import StreamLoader


class StreamAnalyzer:
    def __init__(self, config, debugger, trackers, stream_loader):
        self.config = config
        self.debugger = debugger,

        self.stream_loader = stream_loader
        self.yolo_detector = YoloDetector(self.config, max_det=1000).start()
        self.trackers = trackers

    def start(self):
        while True:
            ids, images, imgs_det = self.stream_loader.streams
            if len(images) == 0:
                continue

            detections = self.yolo_detector.get_car_plates_boxes(images, imgs_det)
            for id_, image, det in zip(ids, images, detections):
                tracks = self.trackers[id_].update(det)
                cv2.imshow(f'Stream {id_}', self.yolo_detector.draw_boxes(image, tracks))
                if cv2.waitKey(1) == ord('q'):
                    break
