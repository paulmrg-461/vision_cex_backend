import logging
from typing import Optional, Tuple, List

from PIL import Image

_logger = logging.getLogger(__name__)

try:
    from paddleocr import PaddleOCR
    _paddle_available = True
except Exception as e:
    _paddle_available = False
    _logger.error("PaddleOCR no disponible: %s", e)


class PaddleOcrAdapter:
    def __init__(self, lang: str = "en") -> None:
        self._enabled = _paddle_available
        self._ocr: Optional[PaddleOCR] = None
        self._lang = lang
        if self._enabled:
            try:
                # use_angle_cls mejora resultados en textos inclinados
                self._ocr = PaddleOCR(use_angle_cls=True, lang=self._lang)
                _logger.info("PaddleOCR inicializado (lang=%s)", self._lang)
            except Exception as e:
                self._enabled = False
                _logger.error("No se pudo inicializar PaddleOCR: %s", e)

    def read_text(self, image: Image.Image) -> Tuple[str, float]:
        """Devuelve (texto, confianza). Si falla, ('', 0.0)."""
        if not self._enabled or self._ocr is None:
            return "", 0.0
        try:
            # PaddleOCR espera rutas o ndarray; convertimos PIL -> ndarray
            import numpy as np
            arr = np.array(image)
            result = self._ocr.ocr(arr, cls=True)
            # result es lista por imagen; tomamos mejor línea por confianza
            best_text = ""
            best_conf = 0.0
            for page in result or []:
                for line in page or []:
                    try:
                        text = line[1][0]
                        conf = float(line[1][1])
                        if conf > best_conf:
                            best_text = text
                            best_conf = conf
                    except Exception:
                        continue
            # Normalizar probable formato de placa
            normalized = best_text.upper().replace(" ", "").replace("_", "-")
            return normalized, best_conf
        except Exception as e:
            _logger.error("Error OCR: %s", e)
            return "", 0.0

