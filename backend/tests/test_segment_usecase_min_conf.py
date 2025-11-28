from typing import List, Tuple

from app.domain.entities.segmentation_entity import SegmentationObject
from app.domain.usecases.segment_objects_usecase import SegmentObjectsUseCase


class FakeSegmenter:
    def __init__(self, objs: List[SegmentationObject]):
        self._objs = objs

    def segment(self, frame):
        return self._objs


def test_segment_usecase_filters_by_class_and_conf():
    objs = [
        SegmentationObject(polygon=[(0, 0), (10, 0), (10, 10)], cls="bus", conf=0.49, bbox=(0, 0, 10, 10)),
        SegmentationObject(polygon=[(20, 20), (30, 20), (30, 30)], cls="bus", conf=0.90, bbox=(20, 20, 30, 30)),
        SegmentationObject(polygon=[(40, 40), (50, 40), (50, 50)], cls="car", conf=0.95, bbox=(40, 40, 50, 50)),
    ]
    seg = FakeSegmenter(objs)
    uc = SegmentObjectsUseCase(detector_adapter=seg, allowed_classes=["bus"], min_conf=0.5)
    out = uc.segment(frame=None)
    assert len(out) == 1
    assert out[0].cls == "bus" and out[0].conf >= 0.5

