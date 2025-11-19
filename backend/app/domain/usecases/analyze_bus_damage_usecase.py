from typing import List

from app.domain.entities.damage_entity import DamageReport
from app.domain.repositories.damage_analysis_repository import DamageAnalysisRepository


class AnalyzeBusDamageUseCase:
    """Caso de uso para analizar daños en un bus a partir de múltiples fotos."""

    def __init__(self, repository: DamageAnalysisRepository):
        self._repo = repository

    def analyze(self, images: List[str], locale: str = "es") -> DamageReport:
        """Ejecuta el análisis delegando al repositorio (DeepSeek)."""
        return self._repo.analyze(images, locale=locale)