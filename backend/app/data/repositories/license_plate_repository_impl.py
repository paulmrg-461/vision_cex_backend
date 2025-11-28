from typing import Optional, Tuple

from app.domain.entities.license_plate_entity import (
    LicensePlateDetectionsResponse,
)
from app.domain.repositories.license_plate_repository import LicensePlateRepository
from app.data.adapters.yolo_license_plate_adapter import YoloLicensePlateAdapter


class LicensePlateRepositoryImpl(LicensePlateRepository):
    def __init__(self, adapter: YoloLicensePlateAdapter) -> None:
        self._adapter = adapter

    def detect(
        self,
        image_url: str,
        conf: float = 0.25,
        max_detections: int = 50,
        imgsz: Optional[int] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> LicensePlateDetectionsResponse:
        # Permitir ajuste dinámico de umbral, tamaño de entrada e ROI por petición
        detections = self._adapter.detect(
            image_source=image_url,
            max_detections=max_detections,
            conf_override=conf,
            imgsz_override=imgsz,
            roi=roi,
        )
        return LicensePlateDetectionsResponse(image_url=image_url, detections=detections)
