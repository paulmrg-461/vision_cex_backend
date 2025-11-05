from typing import List, Optional, Tuple
import cv2

from app.domain.entities.bbox_entity import BoundingBox


class DetectObjectsUseCase:
    """Use case that runs detection over frames and returns bounding boxes.

    It delegates inference to a detector adapter injected via DI.
    """

    def __init__(self, detector_adapter):
        self._detector = detector_adapter

    def detect(self, frame, roi: Optional[Tuple[int, int, int, int]] = None) -> List[BoundingBox]:
        """Detect objects on a frame.

        roi: optional rectangle (x, y, w, h) limiting detection to a region.
        """
        if roi is not None:
            x, y, w, h = roi
            roi_frame = frame[y:y + h, x:x + w]
            boxes = self._detector.detect(roi_frame)
            # translate boxes back to global coordinates
            translated = []
            for b in boxes:
                translated.append(
                    BoundingBox(x1=b.x1 + x, y1=b.y1 + y, x2=b.x2 + x, y2=b.y2 + y, cls=b.cls, conf=b.conf)
                )
            return translated
        else:
            return self._detector.detect(frame)

    @staticmethod
    def draw_boxes(frame, boxes: List[BoundingBox]):
        for b in boxes:
            cv2.rectangle(frame, (b.x1, b.y1), (b.x2, b.y2), (0, 255, 0), 2)
            label = f"{b.cls or 'obj'} {b.conf:.2f}" if b.conf is not None else (b.cls or "obj")
            cv2.putText(frame, label, (b.x1, max(0, b.y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)