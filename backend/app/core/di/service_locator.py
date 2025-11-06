from typing import Optional

from app.core.config.environment_config import EnvironmentConfig
from app.data.adapters.yolo_ultralytics_adapter import YoloUltralyticsAdapter
from app.data.adapters.yolo_onnx_adapter import YoloOnnxAdapter
from app.data.adapters.grounding_dino_adapter import GroundingDinoSamAdapter
from app.domain.usecases.detect_objects_usecase import DetectObjectsUseCase


class ServiceLocator:
    _config: Optional[EnvironmentConfig] = None
    _detector_adapter: Optional[YoloUltralyticsAdapter] = None
    _detect_usecase: Optional[DetectObjectsUseCase] = None

    @classmethod
    def config(cls) -> EnvironmentConfig:
        if cls._config is None:
            cls._config = EnvironmentConfig()
        return cls._config

    @classmethod
    def detector_adapter(cls):
        if cls._detector_adapter is None:
            cfg = cls.config()
            if cfg.model_backend.lower() == "onnx":
                cls._detector_adapter = YoloOnnxAdapter(onnx_path=cfg.yolo_weights, device=cfg.device, input_size=cfg.model_input_size)
            elif cfg.model_backend.lower() == "grounding":
                cls._detector_adapter = GroundingDinoSamAdapter(
                    device=cfg.device,
                    prompt=cfg.grounding_prompt,
                    box_threshold=cfg.grounding_box_threshold,
                    text_threshold=cfg.grounding_text_threshold,
                    download_dir=cfg.grounding_download_dir,
                    grounding_config_path=(cfg.grounding_config_path or None),
                    grounding_config_url=(cfg.grounding_config_url or None),
                    grounding_weights_path=(cfg.grounding_weights_path or None),
                    grounding_weights_url=(cfg.grounding_weights_url or None),
                    sam_checkpoint_path=(cfg.sam_checkpoint_path or None),
                    sam_checkpoint_url=(cfg.sam_checkpoint_url or None),
                    sam_model_type=cfg.sam_model_type,
                )
            else:
                cls._detector_adapter = YoloUltralyticsAdapter(weights_path=cfg.yolo_weights, device=cfg.device)
        return cls._detector_adapter

    @classmethod
    def detect_usecase(cls) -> DetectObjectsUseCase:
        if cls._detect_usecase is None:
            cls._detect_usecase = DetectObjectsUseCase(detector_adapter=cls.detector_adapter())
        return cls._detect_usecase