from typing import List

from app.data.adapters.hf_vqa_client import HuggingFaceVqaClient
from app.domain.entities.vqa_entity import VqaAnswer, VqaResponse
from app.domain.repositories.vqa_repository import VqaRepository


class VqaRepositoryImpl(VqaRepository):
    def __init__(self, client: HuggingFaceVqaClient) -> None:
        self._client = client

    def answer(self, question: str, image_url: str, top_k: int = 1) -> VqaResponse:
        raw_answers = self._client.answer(question=question, image_url=image_url, top_k=top_k)
        answers: List[VqaAnswer] = []
        for item in raw_answers:
            if isinstance(item, dict):
                answers.append(VqaAnswer(answer=str(item.get("answer", "")), score=item.get("score")))
            else:
                answers.append(VqaAnswer(answer=str(item)))
        return VqaResponse(question=question, image_url=image_url, answers=answers)