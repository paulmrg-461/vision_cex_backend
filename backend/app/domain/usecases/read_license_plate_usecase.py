from typing import Optional, Tuple

from app.domain.entities.license_plate_entity import LicensePlateReadResponse
from app.data.repositories.license_plate_ocr_repository_impl import LicensePlateOcrRepositoryImpl


class ReadLicensePlateUseCase:
    def __init__(self, repository: LicensePlateOcrRepositoryImpl) -> None:
        self._repo = repository

    def execute(
        self,
        image_url: str,
        conf: float = 0.25,
        imgsz: Optional[int] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> LicensePlateReadResponse:
        if not image_url or not image_url.strip():
            raise ValueError("La URL de la imagen no puede estar vacía")
        return self._repo.read(image_url=image_url.strip(), conf=conf, imgsz=imgsz, roi=roi)

