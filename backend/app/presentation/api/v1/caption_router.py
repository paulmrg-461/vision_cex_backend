from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, HttpUrl

from app.core.di.service_locator import ServiceLocator
from app.domain.entities.caption_entity import CaptionResponse


router = APIRouter(prefix="/api/v1/caption", tags=["caption"])


class CaptionRequest(BaseModel):
    image_url: HttpUrl = Field(..., description="URL de la imagen a describir")
    top_k: int = Field(1, ge=1, le=5, description="Número de descripciones candidatas a devolver")


@router.post("/describe", response_model=CaptionResponse)
def describe_image(body: CaptionRequest):
    try:
        usecase = ServiceLocator.describe_usecase()
        return usecase.execute(image_url=str(body.image_url), top_k=body.top_k)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fallo al describir la imagen: {e}")