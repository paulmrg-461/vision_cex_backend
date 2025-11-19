from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.di.service_locator import ServiceLocator

router = APIRouter(prefix="/api/v1/ernie", tags=["ernie"])

class ErnieRequest(BaseModel):
    image_url: str = Field(..., description="URL of the image, base64 string, or local path (server-side)")
    prompt: str = Field(..., description="Prompt for the model")

class ErnieResponse(BaseModel):
    response: str

@router.post("/generate", response_model=ErnieResponse)
def generate_response(req: ErnieRequest):
    try:
        usecase = ServiceLocator.ask_ernie_usecase()
        response = usecase.execute(image_url=req.image_url, prompt=req.prompt)
        return ErnieResponse(response=response)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {e}")
