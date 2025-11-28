from typing import List, Optional
from datetime import datetime
from app.domain.entities.bus_report_entity import BusReport
from app.domain.repositories.bus_report_repository import BusReportRepository


class CreateBusReportUseCase:
    def __init__(self, repo: BusReportRepository):
        self._repo = repo

    def execute(self, license_plate: str, event_datetime: datetime, damages: Optional[list] = None) -> BusReport:
        return self._repo.create(BusReport(id=None, license_plate=license_plate, event_datetime=event_datetime, damages=damages or []))


class GetBusReportUseCase:
    def __init__(self, repo: BusReportRepository):
        self._repo = repo

    def execute(self, report_id: int) -> Optional[BusReport]:
        return self._repo.get_by_id(report_id)


class ListBusReportsUseCase:
    def __init__(self, repo: BusReportRepository):
        self._repo = repo

    def execute(self, limit: int = 50, offset: int = 0) -> List[BusReport]:
        return self._repo.list(limit=limit, offset=offset)


class UpdateBusReportUseCase:
    def __init__(self, repo: BusReportRepository):
        self._repo = repo

    def execute(self, report_id: int, license_plate: str, event_datetime: datetime, damages: Optional[list] = None) -> Optional[BusReport]:
        return self._repo.update(report_id, BusReport(id=report_id, license_plate=license_plate, event_datetime=event_datetime, damages=damages or []))


class DeleteBusReportUseCase:
    def __init__(self, repo: BusReportRepository):
        self._repo = repo

    def execute(self, report_id: int) -> bool:
        return self._repo.delete(report_id)

