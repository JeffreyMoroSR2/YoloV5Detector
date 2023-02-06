from stream_loader import StreamLoader
from yolo_detect import YoloDetector

import cv2
import torch
import numpy as np
from source import Source


def draw_boxes(frame_to_draw, detection, id_):
    if len(detection) == 0:
        return frame_to_draw

    for det in range(len(detection)):
        frame_to_draw = cv2.rectangle(frame_to_draw, (detection[det].bbox[0], detection[det].bbox[1]),
                                      (detection[det].bbox[2], detection[det].bbox[3]), (0, 0, 255), 2)
        frame_to_draw = cv2.putText(frame_to_draw, f'{det}', (detection[det].bbox[0], detection[det].bbox[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    return frame_to_draw


if __name__ == '__main__':
    streams = ['../plate.jpg',
               '../plates.jpg']
    sources = []
    for i, stream in enumerate(streams):
        sources.append(Source(stream=stream, stream_id=i))

    stream_loader = StreamLoader(sources).start()

    yolo_detector = YoloDetector('yolov5/custom_models/carplates_det.pt', max_det=1000).load_model()

    while True:
        ids, images, imgs_det = stream_loader.streams
        if len(images) == 0:
            continue
        detections = yolo_detector.get_car_plates_boxes(images, imgs_det)
        for id_, image, det in zip(ids, images, detections):
            print(f'Stream - {id_} --- {det}')
            cv2.imshow(f'Stream {id_}', draw_boxes(image, det, id_))

            if cv2.waitKey(1) == ord('q'):
                break
