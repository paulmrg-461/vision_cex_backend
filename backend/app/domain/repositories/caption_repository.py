from abc import ABC, abstractmethod

from app.domain.entities.caption_entity import CaptionResponse


class CaptionRepository(ABC):
    @abstractmethod
    def describe(self, image_url: str, top_k: int = 1) -> CaptionResponse:
        """Generate captions for the given image URL."""
        raise NotImplementedError