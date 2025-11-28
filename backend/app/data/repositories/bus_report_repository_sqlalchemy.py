from typing import List, Optional
import re
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.core.db.sqlalchemy import SQLAlchemySession, Base
from app.domain.entities.bus_report_entity import BusReport
from app.domain.repositories.bus_report_repository import BusReportRepository
from app.data.models.bus_report_model import BusReportModel


_PLATE_RE = re.compile(r"^[A-Z]{3}[0-9]{3}$")


class BusReportRepositorySqlAlchemy(BusReportRepository):
    def __init__(self):
        # Ensure tables exist
        engine, _ = SQLAlchemySession.init()
        Base.metadata.create_all(bind=engine)

    def _validate_plate(self, plate: str) -> str:
        plate = (plate or "").strip().upper()
        if not _PLATE_RE.match(plate):
            raise ValueError("license_plate inválido: formato esperado ABC123")
        return plate

    def _to_entity(self, m: BusReportModel) -> BusReport:
        return BusReport(id=m.id, license_plate=m.license_plate, event_datetime=m.event_datetime, damages=m.damages or [])

    def create(self, report: BusReport) -> BusReport:
        plate = self._validate_plate(report.license_plate)
        with SQLAlchemySession.session() as s:
            m = BusReportModel(license_plate=plate, event_datetime=report.event_datetime, damages=report.damages or [])
            s.add(m)
            s.commit()
            s.refresh(m)
            return self._to_entity(m)

    def get_by_id(self, report_id: int) -> Optional[BusReport]:
        with SQLAlchemySession.session() as s:
            m = s.get(BusReportModel, int(report_id))
            return self._to_entity(m) if m else None

    def list(self, limit: int = 50, offset: int = 0) -> List[BusReport]:
        with SQLAlchemySession.session() as s:
            stmt = select(BusReportModel).order_by(BusReportModel.id.desc()).limit(int(limit)).offset(int(offset))
            rows = s.execute(stmt).scalars().all()
            return [self._to_entity(m) for m in rows]

    def update(self, report_id: int, report: BusReport) -> Optional[BusReport]:
        plate = self._validate_plate(report.license_plate)
        with SQLAlchemySession.session() as s:
            m = s.get(BusReportModel, int(report_id))
            if not m:
                return None
            m.license_plate = plate
            m.event_datetime = report.event_datetime
            m.damages = report.damages or []
            s.commit()
            s.refresh(m)
            return self._to_entity(m)

    def delete(self, report_id: int) -> bool:
        with SQLAlchemySession.session() as s:
            m = s.get(BusReportModel, int(report_id))
            if not m:
                return False
            s.delete(m)
            s.commit()
            return True

