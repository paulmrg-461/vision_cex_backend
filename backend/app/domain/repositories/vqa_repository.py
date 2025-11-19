from abc import ABC, abstractmethod
from typing import List

from app.domain.entities.vqa_entity import VqaAnswer, VqaResponse


class VqaRepository(ABC):
    @abstractmethod
    def answer(self, question: str, image_url: str, top_k: int = 1) -> VqaResponse:
        """Answer a visual question for a given image URL.

        Returns a VqaResponse with up to top_k answers.
        """
        raise NotImplementedError