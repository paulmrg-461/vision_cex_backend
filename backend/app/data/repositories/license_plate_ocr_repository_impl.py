from typing import Optional, Tuple

from PIL import Image

from app.domain.entities.license_plate_entity import LicensePlateReadResponse
from app.data.adapters.yolo_license_plate_adapter import YoloLicensePlateAdapter
from app.data.adapters.paddle_ocr_adapter import PaddleOcrAdapter


class LicensePlateOcrRepositoryImpl:
    def __init__(self, detector: YoloLicensePlateAdapter, ocr: PaddleOcrAdapter) -> None:
        self._detector = detector
        self._ocr = ocr

    def _crop(self, img: Image.Image, bbox: Tuple[int, int, int, int], pad_ratio: float = 0.2) -> Image.Image:
        """Recorta con margen alrededor del bbox para evitar cortar caracteres.

        pad_ratio: porcentaje de padding relativo al ancho/alto de la caja.
        """
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(img.width - 1, x1))
        y1 = max(0, min(img.height - 1, y1))
        x2 = max(x1 + 1, min(img.width, x2))
        y2 = max(y1 + 1, min(img.height, y2))
        w = x2 - x1
        h = y2 - y1
        pad_x = int(w * pad_ratio)
        pad_y = int(h * pad_ratio)
        nx1 = max(0, x1 - pad_x)
        ny1 = max(0, y1 - pad_y)
        nx2 = min(img.width, x2 + pad_x)
        ny2 = min(img.height, y2 + pad_y)
        return img.crop((nx1, ny1, nx2, ny2))

    def _upscale(self, img: Image.Image) -> Image.Image:
        """Aumenta resolución del recorte para que el OCR tenga más detalle."""
        w, h = img.size
        # Escalar al menos al doble o a 480 px de ancho, manteniendo aspecto
        target_w = max(480, w * 2)
        scale = target_w / float(w)
        target_h = int(h * scale)
        return img.resize((int(target_w), target_h), Image.BICUBIC)

    def read(
        self,
        image_url: str,
        conf: float = 0.25,
        imgsz: Optional[int] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> LicensePlateReadResponse:
        # Cargar la imagen
        img = self._detector._load_image(image_url)

        # Si viene ROI, recortar antes de detectar
        if roi:
            img = self._detector._crop_roi(img, roi)

        # Detectar placas y elegir la mejor
        dets = self._detector.detect(image_source=image_url, conf_override=conf, imgsz_override=imgsz, roi=roi, max_detections=20)
        best_bbox = None
        best_conf = 0.0
        for d in dets:
            if d.confidence > best_conf:
                best_conf = d.confidence
                best_bbox = d.bbox

        # Si no hay detecciones, intentar OCR sobre ROI o imagen completa
        if best_bbox is None:
            crop_img = img
        else:
            crop_img = self._crop(img, best_bbox, pad_ratio=0.2)
        crop_img = self._upscale(crop_img)

        text, ocr_conf = self._ocr.read_text(crop_img)
        return LicensePlateReadResponse(image_url=image_url, text=text, confidence=ocr_conf, bbox=best_bbox)
