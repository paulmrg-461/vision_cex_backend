from typing import List, Dict, Any

from app.domain.entities.damage_entity import DamageReport, DamageFinding
from app.domain.repositories.damage_analysis_repository import DamageAnalysisRepository
from app.data.adapters.deepseek_client import DeepSeekClient
from app.core.utils.logger import get_logger

_logger = get_logger("damage_repo")


class DamageAnalysisRepositoryImpl(DamageAnalysisRepository):
    """Implementación del repositorio que usa DeepSeekClient para análisis de daños."""

    def __init__(self, client: DeepSeekClient):
        self._client = client

    def analyze(self, images: List[str], locale: str = "es") -> DamageReport:
        data = self._client.analyze_images(images, locale=locale)

        try:
            # Si el cliente devolvió un JSON ya estructurado, mapear a entidades.
            if isinstance(data, dict) and "findings" in data:
                findings = []
                raw_findings = data.get("findings", [])
                if isinstance(raw_findings, list):
                    for f in raw_findings:
                        if not isinstance(f, dict):
                            continue
                        bbox_val = f.get("bbox")
                        bbox_tuple = tuple(bbox_val) if isinstance(bbox_val, list) and len(bbox_val) == 4 else None
                        findings.append(
                            DamageFinding(
                                part=f.get("part"),
                                type=f.get("type"),
                                severity=f.get("severity"),
                                confidence=f.get("confidence"),
                                description=f.get("description"),
                                bbox=bbox_tuple,
                            )
                        )
                return DamageReport(findings=findings, summary=data.get("summary"))
        except Exception as e:
            _logger.error("Error mapeando respuesta de DeepSeek: %s", e)

        # Fallback: si la respuesta no es el JSON esperado, construir un reporte genérico.
        _logger.warning("Respuesta DeepSeek no contiene 'findings' estándar; devolviendo reporte básico.")
        # Truncar y serializar con seguridad
        try:
            import json
            summary_text = json.dumps(data)[:1000] if not isinstance(data, str) else data[:1000]
        except Exception:
            summary_text = str(data)[:1000]
        return DamageReport(findings=[], summary=summary_text)