from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Any


@dataclass
class BusReport:
    id: Optional[int]
    license_plate: str
    event_datetime: datetime
    damages: List[Any]

