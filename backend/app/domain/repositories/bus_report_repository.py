from typing import List, Optional
from abc import ABC, abstractmethod
from app.domain.entities.bus_report_entity import BusReport


class BusReportRepository(ABC):
    @abstractmethod
    def create(self, report: BusReport) -> BusReport:
        ...

    @abstractmethod
    def get_by_id(self, report_id: int) -> Optional[BusReport]:
        ...

    @abstractmethod
    def list(self, limit: int = 50, offset: int = 0) -> List[BusReport]:
        ...

    @abstractmethod
    def update(self, report_id: int, report: BusReport) -> Optional[BusReport]:
        ...

    @abstractmethod
    def delete(self, report_id: int) -> bool:
        ...

