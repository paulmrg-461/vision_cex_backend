from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.di.service_locator import ServiceLocator


router = APIRouter(prefix="/api/v1/license-plates", tags=["license-plates"])


class LicensePlateDetectRequest(BaseModel):
    image_url: str = Field(..., description="URL de la imagen (http/https o data URL, también puede ser ruta local dentro del contenedor)")
    conf: float = Field(0.25, ge=0.0, le=1.0, description="Umbral de confianza mínimo")
    max_detections: int = Field(50, ge=1, le=200, description="Número máximo de detecciones a devolver")
    imgsz: int | None = Field(
        None,
        description="Tamaño de entrada del modelo (por ejemplo 640, 960, 1280). Si no se envía, usa el valor por defecto",
    )
    roi: str | None = Field(
        None,
        description=(
            "Área de interés opcional para recortar antes de detectar. Formatos admitidos: 'x1,y1,x2,y2' o 'x,y,w,h'"
        ),
    )


class LicensePlateDetectionResponse(BaseModel):
    label: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]


class LicensePlateDetectionsResponse(BaseModel):
    image_url: str
    detections: List[LicensePlateDetectionResponse]


def _sanitize_url(url: str) -> str:
    return url.strip().strip('`"')


def _parse_roi(roi: str | None):
    if not roi:
        return None
    try:
        parts = [int(float(p.strip())) for p in roi.split(',')]
        if len(parts) == 4:
            x1, y1, a, b = parts
            # decidir si es (x1,y1,x2,y2) o (x,y,w,h)
            if a > x1 and b > y1:
                # asumimos x2,y2
                return (x1, y1, a, b)
            else:
                # asumimos w,h
                return (x1, y1, x1 + max(1, a), y1 + max(1, b))
    except Exception:
        return None
    return None


@router.post("/detect", response_model=LicensePlateDetectionsResponse)
def detect_license_plates(req: LicensePlateDetectRequest):
    cfg = ServiceLocator.config()
    # No hay dependencias externas obligatorias, pero validamos entrada
    if not req.image_url or not _sanitize_url(req.image_url):
        raise HTTPException(status_code=400, detail="image_url inválida")

    try:
        usecase = ServiceLocator.detect_license_plates_usecase()
        resp = usecase.execute(
            image_url=_sanitize_url(req.image_url),
            conf=req.conf,
            max_detections=req.max_detections,
            imgsz=req.imgsz,
            roi=_parse_roi(req.roi),
        )
        return LicensePlateDetectionsResponse(
            image_url=resp.image_url,
            detections=[
                LicensePlateDetectionResponse(label=d.label, confidence=d.confidence, bbox=list(d.bbox))
                for d in resp.detections
            ],
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error detectando placas: {e}")


class LicensePlateReadRequest(BaseModel):
    image_url: str = Field(..., description="URL de la imagen o ruta local dentro del contenedor")
    conf: float = Field(0.25, ge=0.0, le=1.0, description="Umbral de confianza mínimo para la detección previa")
    imgsz: int | None = Field(None, description="Tamaño de entrada del modelo de detección")
    roi: str | None = Field(None, description="Área de interés opcional para recortar antes de detectar/leer")


class LicensePlateReadResponse(BaseModel):
    image_url: str
    text: str
    confidence: float
    bbox: List[int] | None


@router.post("/read", response_model=LicensePlateReadResponse)
def read_license_plate(req: LicensePlateReadRequest):
    try:
        usecase = ServiceLocator.read_license_plate_usecase()
        resp = usecase.execute(
            image_url=_sanitize_url(req.image_url),
            conf=req.conf,
            imgsz=req.imgsz,
            roi=_parse_roi(req.roi),
        )
        return LicensePlateReadResponse(
            image_url=resp.image_url,
            text=resp.text,
            confidence=resp.confidence,
            bbox=list(resp.bbox) if resp.bbox else None,
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error leyendo placa: {e}")
