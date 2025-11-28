from typing import List, Optional, Any
from datetime import datetime
import re

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, validator

from app.core.di.service_locator import ServiceLocator
from app.domain.entities.bus_report_entity import BusReport


router = APIRouter(prefix="/api/v1/bus_reports", tags=["bus_reports"])


PLATE_RE = re.compile(r"^[A-Z]{3}[0-9]{3}$")


class BusReportCreatePayload(BaseModel):
    license_plate: str = Field(..., description="Placa en formato ABC123")
    event_datetime: datetime
    damages: Optional[List[Any]] = Field(default_factory=list)

    @validator("license_plate")
    def validate_plate(cls, v: str) -> str:
        v2 = (v or "").strip().upper()
        if not PLATE_RE.match(v2):
            raise ValueError("license_plate inválido: usar formato ABC123")
        return v2


class BusReportResponse(BaseModel):
    id: int
    license_plate: str
    event_datetime: datetime
    damages: List[Any] = Field(default_factory=list)

    @classmethod
    def from_entity(cls, e: BusReport) -> "BusReportResponse":
        return cls(id=e.id or 0, license_plate=e.license_plate, event_datetime=e.event_datetime, damages=e.damages or [])


@router.post("/", response_model=BusReportResponse)
def create_report(payload: BusReportCreatePayload):
    report = ServiceLocator.create_bus_report_usecase().execute(
        license_plate=payload.license_plate,
        event_datetime=payload.event_datetime,
        damages=payload.damages or [],
    )
    return BusReportResponse.from_entity(report)


@router.get("/{report_id}", response_model=BusReportResponse)
def get_report(report_id: int):
    report = ServiceLocator.get_bus_report_usecase().execute(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="BusReport no encontrado")
    return BusReportResponse.from_entity(report)


@router.get("/", response_model=List[BusReportResponse])
def list_reports(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    items = ServiceLocator.list_bus_reports_usecase().execute(limit=limit, offset=offset)
    return [BusReportResponse.from_entity(i) for i in items]


@router.put("/{report_id}", response_model=BusReportResponse)
def update_report(report_id: int, payload: BusReportCreatePayload):
    report = ServiceLocator.update_bus_report_usecase().execute(
        report_id=report_id,
        license_plate=payload.license_plate,
        event_datetime=payload.event_datetime,
        damages=payload.damages or [],
    )
    if report is None:
        raise HTTPException(status_code=404, detail="BusReport no encontrado")
    return BusReportResponse.from_entity(report)


@router.delete("/{report_id}", response_model=dict)
def delete_report(report_id: int):
    ok = ServiceLocator.delete_bus_report_usecase().execute(report_id)
    if not ok:
        raise HTTPException(status_code=404, detail="BusReport no encontrado")
    return {"deleted": True, "id": report_id}

