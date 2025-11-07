from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class SegmentationObject:
    """Represents a segmented object as a polygon with optional class and confidence.

    - polygon: list of (x, y) points in pixel coordinates of the original frame.
    - cls: optional class name
    - conf: optional confidence score
    - bbox: optional bounding box for convenience (x1, y1, x2, y2)
    """
    polygon: List[Tuple[int, int]]
    cls: Optional[str] = None
    conf: Optional[float] = None
    bbox: Optional[Tuple[int, int, int, int]] = None