from scipy.spatial import distance as dist
from collections import OrderedDict
from detection_obj import DetectionObject
from typing import List, Any
from uuid import uuid4
import numpy as np
import json

config = json.loads(open('external/config.json', 'r').read())


class CentroidTracker:
    def __init__(self, maxDisappeared=config['YoloV5Detector']['centroidtracker']['maxDisappeared']):

        self.nextObjectID = str(uuid4())
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        self.maxDisappeared = maxDisappeared

        self.detection_obj = DetectionObject

    def register(self, det_obj: DetectionObject):
        self.objects[self.nextObjectID] = det_obj
        det_obj.uniq_id = self.nextObjectID
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID = str(uuid4())

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, det_objects: List[DetectionObject]):
        result = []
        if len(det_objects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            return result

        det_objects_copy = det_objects.copy()

        if len(self.objects) == 0:
            for i in range(0, len(det_objects_copy)):
                self.register(det_objects_copy[i])
        else:
            objectIDs = list(self.objects.keys())
            object_bboxes = [det_object.bbox for det_object in self.objects.values()]

            D = dist.cdist(np.array(object_bboxes), [det_object.bbox for det_object in det_objects_copy])
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID].hit_count += 1
                det_objects_copy[col].uniq_id = objectID
                det_objects_copy[col].hit_count = self.objects[objectID].hit_count
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(det_objects_copy[col])

        return det_objects_copy
