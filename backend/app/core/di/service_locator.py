from typing import Optional

from app.core.config.environment_config import EnvironmentConfig
from app.data.adapters.yolo_ultralytics_adapter import YoloUltralyticsAdapter
from app.data.adapters.yolo_onnx_adapter import YoloOnnxAdapter
from app.domain.usecases.detect_objects_usecase import DetectObjectsUseCase
from app.domain.usecases.segment_objects_usecase import SegmentObjectsUseCase


class ServiceLocator:
    _config: Optional[EnvironmentConfig] = None
    _detector_adapter: Optional[YoloUltralyticsAdapter] = None
    _detect_usecase: Optional[DetectObjectsUseCase] = None
    _segment_usecase: Optional[SegmentObjectsUseCase] = None

    @classmethod
    def config(cls) -> EnvironmentConfig:
        if cls._config is None:
            cls._config = EnvironmentConfig()
        return cls._config

    @classmethod
    def detector_adapter(cls) -> YoloUltralyticsAdapter:
        if cls._detector_adapter is None:
            cfg = cls.config()
            if cfg.model_backend.lower() == "onnx":
                cls._detector_adapter = YoloOnnxAdapter(onnx_path=cfg.yolo_weights, device=cfg.device, input_size=cfg.model_input_size)
            else:
                cls._detector_adapter = YoloUltralyticsAdapter(weights_path=cfg.yolo_weights, device=cfg.device, imgsz=cfg.model_input_size)
        return cls._detector_adapter

    @classmethod
    def detect_usecase(cls) -> DetectObjectsUseCase:
        if cls._detect_usecase is None:
            cls._detect_usecase = DetectObjectsUseCase(detector_adapter=cls.detector_adapter())
        return cls._detect_usecase

    @classmethod
    def segment_usecase(cls) -> SegmentObjectsUseCase:
        if cls._segment_usecase is None:
            cls._segment_usecase = SegmentObjectsUseCase(detector_adapter=cls.detector_adapter())
        return cls._segment_usecase