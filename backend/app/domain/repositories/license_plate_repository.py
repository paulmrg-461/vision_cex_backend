from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from app.domain.entities.license_plate_entity import LicensePlateDetectionsResponse


class LicensePlateRepository(ABC):
    @abstractmethod
    def detect(
        self,
        image_url: str,
        conf: float = 0.25,
        max_detections: int = 50,
        imgsz: Optional[int] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> LicensePlateDetectionsResponse:
        """Detecta placas de vehículos en una imagen.

        Devuelve un objeto con la URL y la lista de detecciones con bbox y confianza.
        """
        raise NotImplementedError
