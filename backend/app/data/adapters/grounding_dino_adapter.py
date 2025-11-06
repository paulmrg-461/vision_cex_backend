from typing import List, Optional

import os
import cv2
import numpy as np
import requests

from app.domain.entities.bbox_entity import BoundingBox
from app.core.utils.logger import get_logger

_logger = get_logger("grounding_dino")

_grounding_available = False
_sam_available = False

try:
    from groundingdino.util.inference import Model as GroundingModel  # type: ignore
    _grounding_available = True
except Exception:
    GroundingModel = None  # type: ignore
    _grounding_available = False

try:
    from segment_anything import sam_model_registry, SamPredictor  # type: ignore
    _sam_available = True
except Exception:
    sam_model_registry = None  # type: ignore
    SamPredictor = None  # type: ignore
    _sam_available = False


def _ensure_file(path: str, url: Optional[str]) -> Optional[str]:
    if os.path.isfile(path):
        return path
    if not url:
        return None
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _logger.info("Descargando modelo: %s", url)
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        return path if os.path.isfile(path) else None
    except Exception as e:
        _logger.error("No se pudo descargar '%s': %s", url, e)
        return None


class GroundingDinoSamAdapter:
    """Open-vocabulary detector using GroundingDINO (text prompts).

    Optionally integrates SAM to refine masks, though for bounding boxes we can rely on DINO.
    """

    def __init__(
        self,
        device: str = "auto",
        prompt: str = "car wheel . car door . car window . bus wheel . bus door . bus window",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        download_dir: str = "models",
        grounding_config_path: Optional[str] = None,
        grounding_config_url: Optional[str] = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinB_cfg.py",
        grounding_weights_path: Optional[str] = None,
        grounding_weights_url: Optional[str] = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0/groundingdino_swinb_cogcoor.pth",
        sam_checkpoint_path: Optional[str] = None,
        sam_checkpoint_url: Optional[str] = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        sam_model_type: str = "vit_h",
    ):
        self._enabled = _grounding_available
        self._model = None
        self._sam_predictor = None
        self._device = device
        self._prompt = prompt
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold

        if not self._enabled:
            _logger.error("GroundingDINO no está disponible. Instala el paquete 'groundingdino'.")
            return

        # Resolve paths and download if needed
        if grounding_config_path is None:
            grounding_config_path = os.path.join(download_dir, "GroundingDINO_SwinB_cfg.py")
        if grounding_weights_path is None:
            grounding_weights_path = os.path.join(download_dir, "groundingdino_swinb_cogcoor.pth")
        if sam_checkpoint_path is None:
            sam_checkpoint_path = os.path.join(download_dir, f"sam_{sam_model_type}.pth")

        cfg_ok = _ensure_file(grounding_config_path, grounding_config_url)
        w_ok = _ensure_file(grounding_weights_path, grounding_weights_url)
        if not cfg_ok or not w_ok:
            _logger.error("No se pudieron preparar los archivos de GroundingDINO.")
            self._enabled = False
            return

        try:
            self._model = GroundingModel(
                model_config_path=grounding_config_path,
                model_checkpoint_path=grounding_weights_path,
                device=self._device if self._device != "auto" else ("cuda" if _torch_has_cuda() else "cpu"),
            )
            _logger.info("GroundingDINO cargado (device=%s)", self._model.device)
        except Exception as e:
            _logger.error("Error cargando GroundingDINO: %s", e)
            self._enabled = False
            return

        # Optional SAM for masks (not mandatory for bounding boxes)
        if _sam_available:
            chk = _ensure_file(sam_checkpoint_path, sam_checkpoint_url)
            if chk:
                try:
                    sam = sam_model_registry[sam_model_type](checkpoint=chk)
                    dev = self._model.device if isinstance(self._model.device, str) else "cuda" if _torch_has_cuda() else "cpu"
                    sam.to(dev)
                    self._sam_predictor = SamPredictor(sam)
                    _logger.info("SAM cargado (%s)", sam_model_type)
                except Exception as e:
                    _logger.warning("SAM no se pudo cargar: %s", e)
            else:
                _logger.warning("No se encontró checkpoint de SAM; se omitirá segmentación.")

    def detect(self, frame: np.ndarray) -> List[BoundingBox]:
        boxes: List[BoundingBox] = []
        if not self._enabled or self._model is None:
            return boxes

        try:
            # Convert BGR to RGB as models expect RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pred_boxes, pred_phrases = self._model.predict_with_prompt(
                image=rgb,
                prompt=self._prompt,
                box_threshold=self._box_threshold,
                text_threshold=self._text_threshold,
            )
            # pred_boxes: np.ndarray Nx4 (xyxy), pred_phrases: List[str]
            if pred_boxes is None or len(pred_boxes) == 0:
                return boxes
            for i in range(len(pred_boxes)):
                x1, y1, x2, y2 = pred_boxes[i].astype(int).tolist()
                cls_name = pred_phrases[i] if i < len(pred_phrases) else None
                boxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, cls=cls_name, conf=None))
        except Exception as e:
            _logger.error("Error en GroundingDINO detect: %s", e)
            return boxes

        return boxes


def _torch_has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False