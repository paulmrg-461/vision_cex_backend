from dataclasses import dataclass
from typing import List


@dataclass
class CaptionItem:
    text: str
    score: float | None = None


@dataclass
class CaptionResponse:
    image_url: str
    captions: List[CaptionItem]