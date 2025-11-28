import re

PLATE_RE = re.compile(r"^[A-Z]{3}[0-9]{3}$")


def normalize_license_plate(text: str | None) -> str | None:
    """Normaliza el texto de placa y valida formato ABC123.

    - Convierte a mayúsculas
    - Elimina espacios a los extremos
    - Elimina guiones medios
    - Valida patrón 3 letras + 3 números
    Devuelve la placa normalizada si es válida, de lo contrario None.
    """
    if not text:
        return None
    s = str(text).strip().upper().replace("-", "")
    if PLATE_RE.match(s or ""):
        return s
    return None


def is_valid_license_plate(text: str | None) -> bool:
    """Valida si el texto normalizado cumple ABC123."""
    norm = normalize_license_plate(text)
    return norm is not None

