from typing import List, Optional, Any
from datetime import datetime
import re

from app.domain.entities.bus_report_entity import BusReport
from app.domain.repositories.bus_report_repository import BusReportRepository
from app.core.db.postgres import PostgresClient
from psycopg2.extras import Json


_PLATE_RE = re.compile(r"^[A-Z]{3}[0-9]{3}$")


class BusReportRepositoryImpl(BusReportRepository):
    def _validate_plate(self, plate: str) -> str:
        plate = (plate or "").strip().upper()
        if not _PLATE_RE.match(plate):
            raise ValueError("license_plate inválido: formato esperado ABC123")
        return plate

    def create(self, report: BusReport) -> BusReport:
        plate = self._validate_plate(report.license_plate)
        dt = report.event_datetime
        damages = report.damages or []
        rows = PostgresClient.execute(
            """
            INSERT INTO bus_reports (license_plate, event_datetime, damages)
            VALUES (%s, %s, %s)
            RETURNING id, license_plate, event_datetime, damages
            """,
            (plate, dt, Json(damages)),
        )
        rid, lic, edt, djs = rows[0]
        return BusReport(id=rid, license_plate=lic, event_datetime=edt, damages=list(djs))

    def get_by_id(self, report_id: int) -> Optional[BusReport]:
        rows = PostgresClient.execute(
            "SELECT id, license_plate, event_datetime, damages FROM bus_reports WHERE id=%s",
            (int(report_id),),
        )
        if not rows:
            return None
        rid, lic, edt, djs = rows[0]
        return BusReport(id=rid, license_plate=lic, event_datetime=edt, damages=list(djs))

    def list(self, limit: int = 50, offset: int = 0) -> List[BusReport]:
        rows = PostgresClient.execute(
            "SELECT id, license_plate, event_datetime, damages FROM bus_reports ORDER BY id DESC LIMIT %s OFFSET %s",
            (int(limit), int(offset)),
        ) or []
        return [BusReport(id=r[0], license_plate=r[1], event_datetime=r[2], damages=list(r[3])) for r in rows]

    def update(self, report_id: int, report: BusReport) -> Optional[BusReport]:
        plate = self._validate_plate(report.license_plate)
        dt = report.event_datetime
        damages = report.damages or []
        rows = PostgresClient.execute(
            """
            UPDATE bus_reports SET license_plate=%s, event_datetime=%s, damages=%s
            WHERE id=%s
            RETURNING id, license_plate, event_datetime, damages
            """,
            (plate, dt, Json(damages), int(report_id)),
        )
        if not rows:
            return None
        rid, lic, edt, djs = rows[0]
        return BusReport(id=rid, license_plate=lic, event_datetime=edt, damages=list(djs))

    def delete(self, report_id: int) -> bool:
        PostgresClient.execute("DELETE FROM bus_reports WHERE id=%s", (int(report_id),))
        # cannot easily return affected rows with psycopg2 fetch; a follow-up exists check
        rows = PostgresClient.execute("SELECT 1 FROM bus_reports WHERE id=%s", (int(report_id),))
        return not rows

