import re
from datetime import datetime

from fastapi.testclient import TestClient

from app.presentation.api.main import app


client = TestClient(app)


def _create_sample():
    payload = {
        "license_plate": "ABC123",
        "event_datetime": datetime.utcnow().isoformat(),
        "damages": [],
    }
    res = client.post("/api/v1/bus_reports/", json=payload)
    assert res.status_code == 200, res.text
    body = res.json()
    assert re.fullmatch(r"[A-Z]{3}[0-9]{3}", body["license_plate"]) is not None
    assert isinstance(body["id"], int)
    return body


def test_create_get_update_delete_success():
    created = _create_sample()
    rid = created["id"]

    # get
    res = client.get(f"/api/v1/bus_reports/{rid}")
    assert res.status_code == 200
    assert res.json()["id"] == rid

    # list
    res = client.get("/api/v1/bus_reports/?limit=10&offset=0")
    assert res.status_code == 200
    assert isinstance(res.json(), list)

    # update
    payload = {
        "license_plate": "DEF456",
        "event_datetime": datetime.utcnow().isoformat(),
        "damages": [{"part": "door", "severity": "low"}],
    }
    res = client.put(f"/api/v1/bus_reports/{rid}", json=payload)
    assert res.status_code == 200
    assert res.json()["license_plate"] == "DEF456"

    # delete
    res = client.delete(f"/api/v1/bus_reports/{rid}")
    assert res.status_code == 200
    assert res.json()["deleted"] is True


def test_invalid_plate_failure():
    payload = {
        "license_plate": "invalid-plate",
        "event_datetime": datetime.utcnow().isoformat(),
        "damages": [],
    }
    res = client.post("/api/v1/bus_reports/", json=payload)
    assert res.status_code == 422


def test_security_injection_plate_rejected():
    payload = {
        "license_plate": "ABC123; DROP TABLE bus_reports;--",
        "event_datetime": datetime.utcnow().isoformat(),
        "damages": [],
    }
    res = client.post("/api/v1/bus_reports/", json=payload)
    # Pydantic validator should reject format
    assert res.status_code == 422

