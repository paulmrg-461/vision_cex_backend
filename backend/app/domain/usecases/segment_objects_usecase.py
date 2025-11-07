from typing import List, Optional, Tuple
import cv2
import numpy as np

from app.domain.entities.segmentation_entity import SegmentationObject


class SegmentObjectsUseCase:
    """Use case that runs segmentation over frames and returns instance polygons.

    Delegates inference to a detector/segmenter adapter injected via DI.
    """

    def __init__(self, detector_adapter):
        self._detector = detector_adapter

    def segment(self, frame, roi: Optional[Tuple[int, int, int, int]] = None) -> List[SegmentationObject]:
        """Run instance segmentation on a frame.

        roi: optional rectangle (x, y, w, h) limiting segmentation to a region.
        """
        if roi is not None:
            x, y, w, h = roi
            roi_frame = frame[y:y + h, x:x + w]
            objs = self._detector.segment(roi_frame) if hasattr(self._detector, 'segment') else []
            translated: List[SegmentationObject] = []
            for o in objs:
                # translate polygon points back to global coordinates
                poly = [(px + x, py + y) for (px, py) in o.polygon]
                bbox = None
                if o.bbox is not None:
                    bx1, by1, bx2, by2 = o.bbox
                    bbox = (bx1 + x, by1 + y, bx2 + x, by2 + y)
                translated.append(SegmentationObject(polygon=poly, cls=o.cls, conf=o.conf, bbox=bbox))
            return translated
        else:
            return self._detector.segment(frame) if hasattr(self._detector, 'segment') else []

    @staticmethod
    def draw_masks(frame, instances: List[SegmentationObject], alpha: float = 0.4, draw_bboxes: bool = False, box_color: tuple = (0, 255, 0), box_thickness: int = 2):
        """Overlay filled polygons on frame with semi-transparency and optional labels.

        alpha: blending factor for mask overlay.
        draw_bboxes: if True, also draw bounding boxes around each mask instance.
        box_color: color for bounding boxes.
        box_thickness: thickness for bounding boxes.
        """
        if not instances:
            return
        overlay = frame.copy()
        for o in instances:
            if not o.polygon:
                continue
            pts = np.array(o.polygon, dtype=np.int32).reshape((-1, 1, 2))
            color = (0, 255, 0)  # green by default; could be class-based
            cv2.fillPoly(overlay, [pts], color)
            label = f"{o.cls or 'seg'} {o.conf:.2f}" if o.conf is not None else (o.cls or "seg")
            # place label near first point or bbox top-left if available
            if o.bbox is not None:
                x1, y1, _, _ = o.bbox
                pos = (x1, max(0, y1 - 5))
            else:
                pos = (o.polygon[0][0], max(0, o.polygon[0][1] - 5))
            cv2.putText(overlay, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # optional bounding box drawing
            if draw_bboxes:
                if o.bbox is not None:
                    x1, y1, x2, y2 = o.bbox
                else:
                    # compute bbox from polygon
                    xs = [p[0] for p in o.polygon]
                    ys = [p[1] for p in o.polygon]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, box_thickness)
        # blend overlay onto original frame
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)