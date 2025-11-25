from typing import Optional, Tuple

from PIL import Image

from app.domain.entities.license_plate_entity import LicensePlateReadResponse
from app.data.adapters.yolo_license_plate_adapter import YoloLicensePlateAdapter
from app.data.adapters.paddle_ocr_adapter import PaddleOcrAdapter


class LicensePlateOcrRepositoryImpl:
    def __init__(self, detector: YoloLicensePlateAdapter, ocr: PaddleOcrAdapter) -> None:
        self._detector = detector
        self._ocr = ocr

    def _crop(self, img: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(img.width - 1, x1))
        y1 = max(0, min(img.height - 1, y1))
        x2 = max(x1 + 1, min(img.width, x2))
        y2 = max(y1 + 1, min(img.height, y2))
        return img.crop((x1, y1, x2, y2))

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
            crop_img = self._crop(img, best_bbox)

        text, ocr_conf = self._ocr.read_text(crop_img)
        return LicensePlateReadResponse(image_url=image_url, text=text, confidence=ocr_conf, bbox=best_bbox)

