from typing import List

from app.domain.entities.bbox_entity import BoundingBox
from app.domain.usecases.detect_objects_usecase import DetectObjectsUseCase


class FakeDetector:
    def __init__(self, boxes: List[BoundingBox]):
        self._boxes = boxes

    def detect(self, frame):
        return self._boxes


def test_detect_usecase_filters_by_class_and_conf():
    boxes = [
        BoundingBox(x1=0, y1=0, x2=10, y2=10, cls="bus", conf=0.49),
        BoundingBox(x1=10, y1=10, x2=20, y2=20, cls="bus", conf=0.90),
        BoundingBox(x1=30, y1=30, x2=40, y2=40, cls="car", conf=0.95),
    ]
    detector = FakeDetector(boxes)
    uc = DetectObjectsUseCase(detector_adapter=detector, allowed_classes=["bus"], min_conf=0.5)
    out = uc.detect(frame=None)
    assert len(out) == 1
    assert out[0].cls == "bus" and out[0].conf >= 0.5

