from typing import List, Tuple, Optional
from io import BytesIO

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

from PIL import Image
import requests

from app.core.utils.logger import get_logger
from app.domain.entities.license_plate_entity import LicensePlateDetection

_logger = get_logger("yolo_license_plate_adapter")


class YoloLicensePlateAdapter:
    """Adapter especializado para detección de placas usando Ultralytics YOLO.

    Carga los pesos indicados y expone un método `detect(image)` que devuelve
    detecciones de placas con bbox y confianza.
    """

    def __init__(self, weights_path: str, device: str = "auto", imgsz: int = 640, conf: float = 0.25):
        self._enabled = _ultralytics_available
        self._model = None
        self._imgsz = imgsz
        self._conf = conf
        self._device = "cpu"
        self._names: Optional[dict] = None
        self._default_classes: Optional[List[int]] = None
        if self._enabled:
            try:
                self._model = YOLO(weights_path)
                if device == "auto":
                    if _torch_available and torch.cuda.is_available():
                        self._device = "0"  # primera GPU
                    else:
                        self._device = "cpu"
                else:
                    self._device = device
                # Obtener nombres de clases del modelo si están disponibles
                try:
                    self._names = getattr(self._model, "names", None)
                    if isinstance(self._names, dict):
                        candidates: List[int] = []
                        for k, v in self._names.items():
                            name = str(v).lower()
                            if "plate" in name or "placa" in name:
                                try:
                                    candidates.append(int(k))
                                except Exception:
                                    continue
                        self._default_classes = candidates if candidates else None
                except Exception:
                    self._default_classes = None
                _logger.info("YOLO License Plates cargado: %s (device=%s)", weights_path, self._device)
            except Exception as e:
                self._enabled = False
                _logger.error("No se pudo cargar YOLO para placas con pesos '%s': %s", weights_path, e)

    @staticmethod
    def _sanitize_source(src: str) -> str:
        s = src.strip().strip('`"')
        # Reescribir enlaces de Google Drive que devuelven HTML a descarga directa
        try:
            if "drive.usercontent.google.com" in s and "export=view" in s:
                s = s.replace("export=view", "export=download")
        except Exception:
            pass
        return s

    def _load_image(self, image_source: str) -> Image.Image:
        image_source = self._sanitize_source(image_source)
        if image_source.startswith("http"):
            resp = requests.get(image_source, timeout=30)
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content)).convert("RGB")
        elif image_source.startswith("data:image"):
            header, encoded = image_source.split(",", 1)
            import base64
            data = base64.b64decode(encoded)
            return Image.open(BytesIO(data)).convert("RGB")
        else:
            return Image.open(image_source).convert("RGB")

    @staticmethod
    def _crop_roi(image: Image.Image, roi: Optional[Tuple[int, int, int, int]]) -> Image.Image:
        if not roi:
            return image
        x1, y1, x2, y2 = roi
        try:
            x1 = max(0, min(image.width - 1, x1))
            y1 = max(0, min(image.height - 1, y1))
            x2 = max(x1 + 1, min(image.width, x2))
            y2 = max(y1 + 1, min(image.height, y2))
            return image.crop((x1, y1, x2, y2))
        except Exception:
            return image

    def detect(
        self,
        image_source: str,
        max_detections: int = 50,
        conf_override: float | None = None,
        imgsz_override: Optional[int] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[LicensePlateDetection]:
        if not self._enabled or self._model is None:
            _logger.error("Ultralytics no disponible en el contenedor para detección de placas.")
            return []

        image = self._load_image(image_source)
        image = self._crop_roi(image, roi)
        conf_val = self._conf if conf_override is None else float(conf_override)
        imgsz_val = int(imgsz_override) if imgsz_override else self._imgsz
        half_precision = (_torch_available and self._device != "cpu")
        try:
            results = self._model.predict(
                source=image,
                imgsz=imgsz_val,
                conf=conf_val,
                device=self._device,
                classes=self._default_classes,  # si hay clases de placa detectadas
                half=half_precision,
                verbose=False,
            )
        except TypeError:
            # En algunas versiones, 'half' no es válido; reintentar sin él
            results = self._model.predict(
                source=image,
                imgsz=imgsz_val,
                conf=conf_val,
                device=self._device,
                classes=self._default_classes,
                verbose=False,
            )

        detections: List[LicensePlateDetection] = []
        for r in results:
            names = getattr(r, "names", {})
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            for b in boxes:
                try:
                    xyxy = b.xyxy[0].tolist()  # [x1, y1, x2, y2]
                    x1, y1, x2, y2 = [int(v) for v in xyxy]
                    conf = float(b.conf[0].item()) if hasattr(b, "conf") else 0.0
                    cls_idx = int(b.cls[0].item()) if hasattr(b, "cls") else -1
                    label = names.get(cls_idx, str(cls_idx)) if isinstance(names, dict) else str(cls_idx)
                    detections.append(LicensePlateDetection(label=label, confidence=conf, bbox=(x1, y1, x2, y2)))
                except Exception:
                    continue

        # Ordenar por confianza y limitar
        detections.sort(key=lambda d: d.confidence, reverse=True)
        if max_detections and max_detections > 0:
            detections = detections[:max_detections]
        return detections
