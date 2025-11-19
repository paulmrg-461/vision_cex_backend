from dataclasses import dataclass
from typing import List, Optional


@dataclass
class VqaAnswer:
    answer: str
    score: Optional[float] = None


@dataclass
class VqaResponse:
    question: str
    image_url: str
    answers: List[VqaAnswer]