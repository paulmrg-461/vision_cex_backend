from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, HttpUrl

from app.core.di.service_locator import ServiceLocator
from app.domain.entities.damage_entity import DamageReport, DamageFinding
from app.core.utils.logger import get_logger


router = APIRouter(prefix="/api/v1/damage", tags=["damage"])
logger = get_logger("damage_router")


class AnalyzeDamageRequest(BaseModel):
    images: List[str]  # URLs http(s) o data URLs (data:image/...;base64,...)
    locale: Optional[str] = "es"


class DamageFindingResponse(BaseModel):
    part: Optional[str] = None
    type: Optional[str] = None
    severity: Optional[str] = None
    confidence: Optional[float] = None
    description: Optional[str] = None
    bbox: Optional[List[int]] = None  # [x1, y1, x2, y2]


class DamageReportResponse(BaseModel):
    findings: List[DamageFindingResponse]
    summary: Optional[str] = None


@router.post("/analyze", response_model=DamageReportResponse)
def analyze_damage(req: AnalyzeDamageRequest):
    # Validaciones mínimas y saneo de URLs
    if not req.images or len(req.images) == 0:
        raise HTTPException(status_code=400, detail="Debe proporcionar al menos una imagen.")

    cfg = ServiceLocator.config()
    if not cfg.deepseek_api_key:
        raise HTTPException(status_code=500, detail="DEEPSEEK_API_KEY no configurada en entorno.")

    # Sanitizar entradas: quitar espacios y backticks accidentales
    sanitized_images: List[str] = []
    for s in req.images:
        if not isinstance(s, str):
            continue
        val = s.strip().strip('`').strip('"').strip("'")
        if val:
            sanitized_images.append(val)
    if not sanitized_images:
        raise HTTPException(status_code=400, detail="Las imágenes proporcionadas no son válidas.")

    usecase = ServiceLocator.analyze_damage_usecase()
    try:
        report: DamageReport = usecase.analyze(sanitized_images, locale=req.locale or "es")
    except Exception as e:
        logger.error("Error en análisis de daños: %s", e)
        raise HTTPException(status_code=502, detail=f"Error llamando servicio de análisis: {e}")

    # Mapear a response_model
    resp_findings: List[DamageFindingResponse] = []
    for f in report.findings:
        resp_findings.append(
            DamageFindingResponse(
                part=f.part,
                type=f.type,
                severity=f.severity,
                confidence=f.confidence,
                description=f.description,
                bbox=list(f.bbox) if f.bbox is not None else None,
            )
        )

    return DamageReportResponse(findings=resp_findings, summary=report.summary)