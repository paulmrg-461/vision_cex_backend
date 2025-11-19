from typing import List

from app.data.adapters.hf_caption_client import HuggingFaceCaptionClient
from app.domain.entities.caption_entity import CaptionItem, CaptionResponse
from app.domain.repositories.caption_repository import CaptionRepository


class CaptionRepositoryImpl(CaptionRepository):
    def __init__(self, client: HuggingFaceCaptionClient) -> None:
        self._client = client

    def describe(self, image_url: str, top_k: int = 1) -> CaptionResponse:
        raw = self._client.describe(image_url=image_url, top_k=top_k)
        items: List[CaptionItem] = []
        for obj in raw:
            if isinstance(obj, dict):
                text = str(obj.get("generated_text", obj.get("text", "")))
                score = obj.get("score")
                items.append(CaptionItem(text=text, score=score))
            else:
                items.append(CaptionItem(text=str(obj)))
        return CaptionResponse(image_url=image_url, captions=items)