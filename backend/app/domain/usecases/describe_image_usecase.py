from app.domain.entities.caption_entity import CaptionResponse
from app.domain.repositories.caption_repository import CaptionRepository


class DescribeImageUseCase:
    def __init__(self, repository: CaptionRepository) -> None:
        self._repo = repository

    def execute(self, image_url: str, top_k: int = 1) -> CaptionResponse:
        if not image_url or not image_url.strip():
            raise ValueError("La URL de la imagen no puede estar vacía")
        return self._repo.describe(image_url=image_url.strip(), top_k=top_k)