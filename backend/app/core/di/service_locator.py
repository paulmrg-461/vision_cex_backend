from typing import Optional

from app.core.config.environment_config import EnvironmentConfig
from app.data.adapters.yolo_ultralytics_adapter import YoloUltralyticsAdapter
from app.data.adapters.yolo_onnx_adapter import YoloOnnxAdapter
from app.domain.usecases.detect_objects_usecase import DetectObjectsUseCase
from app.domain.usecases.segment_objects_usecase import SegmentObjectsUseCase
from app.data.adapters.deepseek_client import DeepSeekClient
from app.data.repositories.damage_analysis_repository_impl import DamageAnalysisRepositoryImpl
from app.domain.usecases.analyze_bus_damage_usecase import AnalyzeBusDamageUseCase
from app.data.adapters.hf_vqa_client import HuggingFaceVqaClient, HuggingFaceVqaConfig
from app.data.repositories.vqa_repository_impl import VqaRepositoryImpl
from app.domain.usecases.answer_vqa_usecase import AnswerVqaUseCase
from app.data.adapters.hf_caption_client import HuggingFaceCaptionClient, HuggingFaceCaptionConfig
from app.data.repositories.caption_repository_impl import CaptionRepositoryImpl
from app.domain.usecases.describe_image_usecase import DescribeImageUseCase
from app.data.adapters.ernie_client import ErnieClient
from app.data.repositories.ernie_repository_impl import ErnieRepositoryImpl
from app.domain.usecases.ask_ernie_usecase import AskErnieUseCase


class ServiceLocator:
    _config: Optional[EnvironmentConfig] = None
    _detector_adapter: Optional[YoloUltralyticsAdapter] = None
    _detect_usecase: Optional[DetectObjectsUseCase] = None
    _segment_usecase: Optional[SegmentObjectsUseCase] = None
    _deepseek_client: Optional[DeepSeekClient] = None
    _damage_repo: Optional[DamageAnalysisRepositoryImpl] = None
    _analyze_damage_usecase: Optional[AnalyzeBusDamageUseCase] = None
    _hf_vqa_client: Optional[HuggingFaceVqaClient] = None
    _vqa_repo: Optional[VqaRepositoryImpl] = None
    _vqa_usecase: Optional[AnswerVqaUseCase] = None
    _hf_caption_client: Optional[HuggingFaceCaptionClient] = None
    _caption_repo: Optional[CaptionRepositoryImpl] = None
    _describe_usecase: Optional[DescribeImageUseCase] = None
    _ernie_client: Optional[ErnieClient] = None
    _ernie_repo: Optional[ErnieRepositoryImpl] = None
    _ask_ernie_usecase: Optional[AskErnieUseCase] = None

    @classmethod
    def config(cls) -> EnvironmentConfig:
        if cls._config is None:
            cls._config = EnvironmentConfig()
            try:
                print(
                    f"[config] HF_VQA_MODE={cls._config.hf_vqa_mode} HF_VQA_MODEL={cls._config.hf_vqa_model} HUGGINGFACE_TOKEN={'SET' if bool(cls._config.huggingface_token) else 'MISSING'} HF_INFERENCE_API_BASE={cls._config.hf_inference_base}"
                )
            except Exception:
                pass
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

    @classmethod
    def deepseek_client(cls) -> DeepSeekClient:
        if cls._deepseek_client is None:
            cfg = cls.config()
            cls._deepseek_client = DeepSeekClient(api_key=cfg.deepseek_api_key, base_url=cfg.deepseek_api_base, model=cfg.deepseek_model)
        return cls._deepseek_client

    @classmethod
    def damage_repo(cls) -> DamageAnalysisRepositoryImpl:
        if cls._damage_repo is None:
            cls._damage_repo = DamageAnalysisRepositoryImpl(client=cls.deepseek_client())
        return cls._damage_repo

    @classmethod
    def analyze_damage_usecase(cls) -> AnalyzeBusDamageUseCase:
        if cls._analyze_damage_usecase is None:
            cls._analyze_damage_usecase = AnalyzeBusDamageUseCase(repository=cls.damage_repo())
        return cls._analyze_damage_usecase

    @classmethod
    def hf_vqa_client(cls) -> HuggingFaceVqaClient:
        if cls._hf_vqa_client is None:
            cfg = cls.config()
            client_cfg = HuggingFaceVqaConfig(
                token=cfg.huggingface_token,
                model=cfg.hf_vqa_model,
                base_url=cfg.hf_inference_base,
                mode=cfg.hf_vqa_mode,
            )
            cls._hf_vqa_client = HuggingFaceVqaClient(config=client_cfg)
        return cls._hf_vqa_client

    @classmethod
    def vqa_repo(cls) -> VqaRepositoryImpl:
        if cls._vqa_repo is None:
            cls._vqa_repo = VqaRepositoryImpl(client=cls.hf_vqa_client())
        return cls._vqa_repo

    @classmethod
    def vqa_usecase(cls) -> AnswerVqaUseCase:
        if cls._vqa_usecase is None:
            cls._vqa_usecase = AnswerVqaUseCase(repository=cls.vqa_repo())
        return cls._vqa_usecase

    @classmethod
    def hf_caption_client(cls) -> HuggingFaceCaptionClient:
        if cls._hf_caption_client is None:
            cfg = cls.config()
            client_cfg = HuggingFaceCaptionConfig(
                token=cfg.huggingface_token,
                model=cfg.hf_caption_model,
                base_url=cfg.hf_inference_base,
                mode=cfg.hf_caption_mode,
            )
            cls._hf_caption_client = HuggingFaceCaptionClient(config=client_cfg)
        return cls._hf_caption_client

    @classmethod
    def caption_repo(cls) -> CaptionRepositoryImpl:
        if cls._caption_repo is None:
            cls._caption_repo = CaptionRepositoryImpl(client=cls.hf_caption_client())
        return cls._caption_repo

    @classmethod
    def describe_usecase(cls) -> DescribeImageUseCase:
        if cls._describe_usecase is None:
            cls._describe_usecase = DescribeImageUseCase(repository=cls.caption_repo())
        return cls._describe_usecase

    @classmethod
    def ernie_client(cls) -> ErnieClient:
        if cls._ernie_client is None:
            cls._ernie_client = ErnieClient()
        return cls._ernie_client

    @classmethod
    def ernie_repo(cls) -> ErnieRepositoryImpl:
        if cls._ernie_repo is None:
            cls._ernie_repo = ErnieRepositoryImpl(client=cls.ernie_client())
        return cls._ernie_repo

    @classmethod
    def ask_ernie_usecase(cls) -> AskErnieUseCase:
        if cls._ask_ernie_usecase is None:
            cls._ask_ernie_usecase = AskErnieUseCase(repository=cls.ernie_repo())
        return cls._ask_ernie_usecase