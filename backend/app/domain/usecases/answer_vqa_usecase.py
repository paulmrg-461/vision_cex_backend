from typing import Optional

from app.domain.entities.vqa_entity import VqaResponse
from app.domain.repositories.vqa_repository import VqaRepository


class AnswerVqaUseCase:
    def __init__(self, repository: VqaRepository) -> None:
        self._repo = repository

    def execute(self, question: str, image_url: str, top_k: int = 1) -> VqaResponse:
        # Basic validation
        if not question or not question.strip():
            raise ValueError("La pregunta no puede estar vacía")
        if not image_url or not image_url.strip():
            raise ValueError("La URL de la imagen no puede estar vacía")
        return self._repo.answer(question=question.strip(), image_url=image_url.strip(), top_k=top_k)