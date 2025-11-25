from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class LicensePlateDetection:
    """Representa una detección de placa de vehículo.

    - label: etiqueta de clase (ej. 'license_plate')
    - confidence: confianza [0.0 - 1.0]
    - bbox: caja delimitadora (x1, y1, x2, y2) en píxeles
    """
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]


@dataclass
class LicensePlateDetectionsResponse:
    image_url: str
    detections: List[LicensePlateDetection]


@dataclass
class LicensePlateReadResponse:
    image_url: str
    text: str
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]]

