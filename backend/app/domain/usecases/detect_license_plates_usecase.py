from typing import Optional, Tuple

from app.domain.entities.license_plate_entity import LicensePlateDetectionsResponse
from app.domain.repositories.license_plate_repository import LicensePlateRepository


class DetectLicensePlatesUseCase:
    def __init__(self, repository: LicensePlateRepository) -> None:
        self._repo = repository

    def execute(
        self,
        image_url: str,
        conf: float = 0.25,
        max_detections: int = 50,
        imgsz: Optional[int] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> LicensePlateDetectionsResponse:
        if not image_url or not image_url.strip():
            raise ValueError("La URL de la imagen no puede estar vacía")
        return self._repo.detect(
            image_url=image_url.strip(), conf=conf, max_detections=max_detections, imgsz=imgsz, roi=roi
        )
