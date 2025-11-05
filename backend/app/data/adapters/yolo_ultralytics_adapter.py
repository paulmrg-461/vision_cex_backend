from typing import List

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
from app.core.utils.logger import get_logger

_logger = get_logger("yolo_ultralytics")


class YoloUltralyticsAdapter:
    """Adapter using Ultralytics YOLO for detection.

    If Ultralytics is not available, it acts as a no-op detector.
    """

    def __init__(self, weights_path: str = "yolov8n.pt", device: str = "auto"):
        self._enabled = _ultralytics_available
        self._model = None
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
            results = self._model.predict(source=frame, device=self._device, verbose=False)
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