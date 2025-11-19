from typing import List
from abc import ABC, abstractmethod

from app.domain.entities.damage_entity import DamageReport


class DamageAnalysisRepository(ABC):
    """Contrato para analizar daños en un bus dado un listado de imágenes."""

    @abstractmethod
    def analyze(self, images: List[str], locale: str = "es") -> DamageReport:
        """
        Analiza daños usando un backend (p.ej. DeepSeek) a partir de imágenes.

        - images: lista de URLs http(s) o data URLs (data:image/...;base64,...) que representan fotos del bus.
        - locale: idioma preferido para el resumen/descripciones.
        """
        raise NotImplementedError