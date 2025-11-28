import pytest

from app.core.utils.license_plate_utils import normalize_license_plate, is_valid_license_plate


def test_normalize_success_with_dash():
    assert normalize_license_plate("abc-123") == "ABC123"


def test_normalize_success_plain():
    assert normalize_license_plate("ABC123") == "ABC123"


def test_normalize_failure_format():
    assert normalize_license_plate("AB1234") is None
    assert normalize_license_plate("ABCD12") is None
    assert normalize_license_plate("A1B2C3") is None


def test_security_unexpected_inputs():
    assert normalize_license_plate("") is None
    assert normalize_license_plate(None) is None
    assert normalize_license_plate("<script>alert(1)</script>") is None


def test_is_valid_license_plate():
    assert is_valid_license_plate("ABC-123") is True
    assert is_valid_license_plate("ABC123") is True
    assert is_valid_license_plate("AB1234") is False

