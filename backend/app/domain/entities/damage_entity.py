from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class DamageFinding:
    """Representa un hallazgo de daño en el bus.

    - part: parte del bus afectada (ej. 'parachoques delantero', 'ventana izquierda')
    - type: tipo de daño (ej. 'scratch', 'dent', 'crack', 'broken_window', 'rust')
    - severity: severidad ('minor', 'moderate', 'major')
    - confidence: confianza del hallazgo [0.0 - 1.0]
    - description: descripción libre del daño
    - bbox: caja delimitadora opcional en píxeles (x1, y1, x2, y2)
    """
    part: Optional[str] = None
    type: Optional[str] = None
    severity: Optional[str] = None
    confidence: Optional[float] = None
    description: Optional[str] = None
    bbox: Optional[Tuple[int, int, int, int]] = None


@dataclass
class DamageReport:
    """Reporte agregado de daños detectados en el bus."""
    findings: List[DamageFinding]
    summary: Optional[str] = None