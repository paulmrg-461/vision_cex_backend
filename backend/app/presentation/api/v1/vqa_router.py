from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.di.service_locator import ServiceLocator


router = APIRouter(prefix="/api/v1/vqa", tags=["vqa"])


class VqaRequest(BaseModel):
    image_url: str = Field(..., description="URL de la imagen")
    question: str = Field(..., description="Pregunta a responder sobre la imagen")
    top_k: int = Field(1, ge=1, le=10, description="Número de respuestas a devolver")


class VqaAnswerResponse(BaseModel):
    answer: str
    score: Optional[float] = None


class VqaResponse(BaseModel):
    question: str
    image_url: str
    answers: List[VqaAnswerResponse]


def _sanitize_url(url: str) -> str:
    return url.strip().strip('`"')


@router.post("/answer", response_model=VqaResponse)
def answer_vqa(req: VqaRequest):
    cfg = ServiceLocator.config()
    # Validate token for API mode
    if cfg.hf_vqa_mode.lower() == "api" and not cfg.huggingface_token:
        raise HTTPException(status_code=400, detail="HUGGINGFACE_TOKEN faltante para el modo API")

    try:
        usecase = ServiceLocator.vqa_usecase()
        resp = usecase.execute(question=req.question, image_url=_sanitize_url(req.image_url), top_k=req.top_k)
        return VqaResponse(
            question=resp.question,
            image_url=resp.image_url,
            answers=[VqaAnswerResponse(answer=a.answer, score=a.score) for a in resp.answers],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error en VQA: {e}")