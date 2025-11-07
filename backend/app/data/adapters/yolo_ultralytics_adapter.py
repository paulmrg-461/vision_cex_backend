from typing import List, Tuple

try:
    from ultralytics import YOLO  # type: ignore
    _ultralytics_available = True
except Exception:
    YOLO = None  # type: ignore
    _ultralytics_available = False

try:
    import torch  # type: ignore
    _torch_available = True
except Exception:
    torch = None  # type: ignore
    _torch_available = False

import numpy as np
from app.domain.entities.bbox_entity import BoundingBox
from app.domain.entities.segmentation_entity import SegmentationObject
from app.core.utils.logger import get_logger

_logger = get_logger("yolo_ultralytics")


class YoloUltralyticsAdapter:
    """Adapter using Ultralytics YOLO for detection and segmentation.

    If Ultralytics is not available, it acts as a no-op detector.
    """

    def __init__(self, weights_path: str = "yolov8n.pt", device: str = "auto", imgsz: int = 640, conf: float = 0.25):
        self._enabled = _ultralytics_available
        self._model = None
        self._imgsz = imgsz
        self._conf = conf
        if self._enabled:
            try:
                self._model = YOLO(weights_path)
                # 'auto' lets Ultralytics choose CUDA if available
                if device == "auto":
                    if _torch_available and torch.cuda.is_available():
                        self._device = "0"  # primera GPU
                    else:
                        self._device = "cpu"
                else:
                    self._device = device
                _logger.info("Ultralytics YOLO cargado: %s (device=%s)", weights_path, self._device)
            except Exception:
                self._enabled = False
                _logger.error("No se pudo cargar Ultralytics YOLO con pesos '%s'", weights_path)

    def detect(self, frame: np.ndarray) -> List[BoundingBox]:
        if not self._enabled or self._model is None:
            return []
        # Run inference; results[0] is per-image
        try:
            half_arg = False if str(self._device).lower() == "cpu" else True
            results = self._model.predict(source=frame, device=self._device, verbose=False, imgsz=self._imgsz, conf=self._conf, half=half_arg)
        except Exception as e:
            _logger.error("Error en predict de Ultralytics: %s", e)
            return []
        boxes: List[BoundingBox] = []
        if not results:
            return boxes
        r = results[0]
        if r.boxes is None:
            return boxes
        for b in r.boxes:
            xyxy = b.xyxy.cpu().numpy().astype(int)[0]
            conf = float(b.conf.cpu().numpy()[0]) if hasattr(b, 'conf') and b.conf is not None else None
            cls_id = int(b.cls.cpu().numpy()[0]) if hasattr(b, 'cls') and b.cls is not None else None
            cls_name = None
            if cls_id is not None and hasattr(r, 'names') and r.names is not None:
                cls_name = r.names.get(cls_id, str(cls_id)) if isinstance(r.names, dict) else str(cls_id)
            boxes.append(BoundingBox(x1=int(xyxy[0]), y1=int(xyxy[1]), x2=int(xyxy[2]), y2=int(xyxy[3]), cls=cls_name, conf=conf))
        return boxes

    def segment(self, frame: np.ndarray) -> List[SegmentationObject]:
        """Run Ultralytics segmentation and return polygons per instance.

        If the loaded weights are not a segmentation model, this returns an empty list.
        """
        instances: List[SegmentationObject] = []
        if not self._enabled or self._model is None:
            return instances
        try:
            half_arg = False if str(self._device).lower() == "cpu" else True
            results = self._model.predict(source=frame, device=self._device, verbose=False, imgsz=self._imgsz, conf=self._conf, half=half_arg)
        except Exception as e:
            _logger.error("Error en predict (segmentación) de Ultralytics: %s", e)
            return instances
        if not results:
            return instances
        r = results[0]
        # r.masks is Ultralytics Masks object for segmentation models
        if not hasattr(r, 'masks') or r.masks is None:
            return instances
        # Obtain polygons in image coordinates
        try:
            polys = r.masks.xy  # List[np.ndarray] of shape (N_i, 2)
        except Exception:
            polys = None
        names = r.names if hasattr(r, 'names') else None
        # Build bbox list for convenience
        boxes_xyxy: List[Tuple[int, int, int, int]] = []
        confs: List[float] = []
        clsnames: List[str] = []
        if r.boxes is not None:
            for b in r.boxes:
                xyxy = b.xyxy.cpu().numpy().astype(int)[0]
                boxes_xyxy.append((int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])))
                confs.append(float(b.conf.cpu().numpy()[0]) if hasattr(b, 'conf') and b.conf is not None else None)  # type: ignore
                cls_id = int(b.cls.cpu().numpy()[0]) if hasattr(b, 'cls') and b.cls is not None else None
                if cls_id is not None and names is not None:
                    clsnames.append(names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id))
                else:
                    clsnames.append(None)  # type: ignore

        # Match polygons to boxes by index
        if polys is None:
            return instances
        for i, poly_arr in enumerate(polys):
            pts = [(int(p[0]), int(p[1])) for p in poly_arr]
            bbox = boxes_xyxy[i] if i < len(boxes_xyxy) else None
            conf = confs[i] if i < len(confs) else None
            cls_name = clsnames[i] if i < len(clsnames) else None
            instances.append(SegmentationObject(polygon=pts, cls=cls_name, conf=conf, bbox=bbox))
        return instances