from typing import Optional

from app.core.config.environment_config import EnvironmentConfig
from app.data.adapters.yolo_ultralytics_adapter import YoloUltralyticsAdapter
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
    def detector_adapter(cls) -> YoloUltralyticsAdapter:
        if cls._detector_adapter is None:
            cfg = cls.config()
            cls._detector_adapter = YoloUltralyticsAdapter(weights_path=cfg.yolo_weights, device=cfg.device)
        return cls._detector_adapter

    @classmethod
    def detect_usecase(cls) -> DetectObjectsUseCase:
        if cls._detect_usecase is None:
            cls._detect_usecase = DetectObjectsUseCase(detector_adapter=cls.detector_adapter())
        return cls._detect_usecase