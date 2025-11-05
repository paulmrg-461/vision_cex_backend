from dataclasses import dataclass
from typing import Optional


@dataclass
class BoundingBox:
    """Represents a detected object bounding box.

    Coordinates are in pixel space of the original frame.
    """
    x1: int
    y1: int
    x2: int
    y2: int
    cls: Optional[str] = None
    conf: Optional[float] = None