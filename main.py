import cv2

from modules.loader.stream_loader import StreamLoader
from tools import YoloDetector
from libs.centroid import CentroidTracker
from objects.source import Source


def draw_boxes(frame_to_draw, detections):
    if len(detections) == 0:
        return frame_to_draw

    for detection in detections:
        frame_to_draw = cv2.rectangle(frame_to_draw, (detection.bbox[0], detection.bbox[1]),
                                      (detection.bbox[2], detection.bbox[3]), (0, 0, 255), 2)
        frame_to_draw = cv2.putText(frame_to_draw, f'{detection.uniq_id}', (detection.bbox[0], detection.bbox[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    return frame_to_draw


if __name__ == '__main__':
    streams = ['rtsp://admdin:admin@192.168.1.2:554',
               'rtsp://admin:admin@192.168.1.2:554']
    sources = []
    for i, stream in enumerate(streams):
        sources.append(Source(stream=stream, stream_id=i))

    stream_loader = StreamLoader(sources).start()
    trackers = [CentroidTracker() for _ in range(len(stream_loader))]

    yolo_detector = YoloDetector('external/weights/detector/carplates_det.pt', max_det=1000).load_model()

    while True:
        ids, images, imgs_det = stream_loader.streams
        if len(images) == 0:
            continue
        detections = yolo_detector.get_car_plates_boxes(images, imgs_det)
        for id_, image, det in zip(ids, images, detections):
            tracks = trackers[id_].update(det)
            print(tracks)
            cv2.imshow(f'Stream {id_}', draw_boxes(image, tracks))

            if cv2.waitKey(1) == ord('q'):
                break
