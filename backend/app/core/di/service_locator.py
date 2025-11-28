from typing import Optional

from app.core.config.environment_config import EnvironmentConfig
from app.data.adapters.yolo_ultralytics_adapter import YoloUltralyticsAdapter
from app.data.adapters.yolo_onnx_adapter import YoloOnnxAdapter
from app.data.adapters.yolo_license_plate_adapter import YoloLicensePlateAdapter
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
from app.data.repositories.license_plate_repository_impl import LicensePlateRepositoryImpl
from app.domain.usecases.detect_license_plates_usecase import DetectLicensePlatesUseCase
from app.data.adapters.paddle_ocr_adapter import PaddleOcrAdapter
from app.data.repositories.license_plate_ocr_repository_impl import LicensePlateOcrRepositoryImpl
from app.domain.usecases.read_license_plate_usecase import ReadLicensePlateUseCase
from app.data.adapters.ernie_client import ErnieClient
from app.data.repositories.ernie_repository_impl import ErnieRepositoryImpl
from app.domain.usecases.ask_ernie_usecase import AskErnieUseCase
from app.domain.repositories.bus_report_repository import BusReportRepository
from app.data.repositories.bus_report_repository_sqlalchemy import BusReportRepositorySqlAlchemy
from app.domain.usecases.bus_reports_usecases import (
    CreateBusReportUseCase,
    GetBusReportUseCase,
    ListBusReportsUseCase,
    UpdateBusReportUseCase,
    DeleteBusReportUseCase,
)


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
    _yolo_lp_adapter: Optional[YoloLicensePlateAdapter] = None
    _lp_repo: Optional[LicensePlateRepositoryImpl] = None
    _lp_usecase: Optional[DetectLicensePlatesUseCase] = None
    _paddle_ocr_adapter: Optional[PaddleOcrAdapter] = None
    _lp_ocr_repo: Optional[LicensePlateOcrRepositoryImpl] = None
    _lp_read_usecase: Optional[ReadLicensePlateUseCase] = None
    _bus_repo: Optional[BusReportRepository] = None
    _bus_create_uc: Optional[CreateBusReportUseCase] = None
    _bus_get_uc: Optional[GetBusReportUseCase] = None
    _bus_list_uc: Optional[ListBusReportsUseCase] = None
    _bus_update_uc: Optional[UpdateBusReportUseCase] = None
    _bus_delete_uc: Optional[DeleteBusReportUseCase] = None

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
    def license_plate_adapter(cls) -> YoloLicensePlateAdapter:
        if cls._yolo_lp_adapter is None:
            cfg = cls.config()
            cls._yolo_lp_adapter = YoloLicensePlateAdapter(
                weights_path=cfg.yolo_plates_weights,
                device=cfg.device,
                imgsz=cfg.model_input_size,
                conf=cfg.yolo_plates_conf,
            )
        return cls._yolo_lp_adapter

    @classmethod
    def detect_usecase(cls) -> DetectObjectsUseCase:
        if cls._detect_usecase is None:
            cfg = cls.config()
            allowed = [c.strip() for c in (cfg.detect_allowed_classes or "").split(",") if c.strip()]
            cls._detect_usecase = DetectObjectsUseCase(detector_adapter=cls.detector_adapter(), allowed_classes=allowed, min_conf=cfg.detect_min_conf)
        return cls._detect_usecase

    @classmethod
    def segment_usecase(cls) -> SegmentObjectsUseCase:
        if cls._segment_usecase is None:
            cfg = cls.config()
            allowed = [c.strip() for c in (cfg.segment_allowed_classes or "").split(",") if c.strip()]
            cls._segment_usecase = SegmentObjectsUseCase(detector_adapter=cls.detector_adapter(), allowed_classes=allowed, min_conf=cfg.segment_min_conf)
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
    def license_plate_repo(cls) -> LicensePlateRepositoryImpl:
        if cls._lp_repo is None:
            cls._lp_repo = LicensePlateRepositoryImpl(adapter=cls.license_plate_adapter())
        return cls._lp_repo

    @classmethod
    def detect_license_plates_usecase(cls) -> DetectLicensePlatesUseCase:
        if cls._lp_usecase is None:
            cls._lp_usecase = DetectLicensePlatesUseCase(repository=cls.license_plate_repo())
        return cls._lp_usecase

    @classmethod
    def paddle_ocr_adapter(cls) -> PaddleOcrAdapter:
        if cls._paddle_ocr_adapter is None:
            cfg = cls.config()
            cls._paddle_ocr_adapter = PaddleOcrAdapter(lang=cfg.lp_ocr_lang)
        return cls._paddle_ocr_adapter

    @classmethod
    def license_plate_ocr_repo(cls) -> LicensePlateOcrRepositoryImpl:
        if cls._lp_ocr_repo is None:
            cls._lp_ocr_repo = LicensePlateOcrRepositoryImpl(detector=cls.license_plate_adapter(), ocr=cls.paddle_ocr_adapter())
        return cls._lp_ocr_repo

    @classmethod
    def read_license_plate_usecase(cls) -> ReadLicensePlateUseCase:
        if cls._lp_read_usecase is None:
            cls._lp_read_usecase = ReadLicensePlateUseCase(repository=cls.license_plate_ocr_repo())
        return cls._lp_read_usecase

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

    # Bus reports bindings
    @classmethod
    def bus_report_repo(cls) -> BusReportRepository:
        if cls._bus_repo is None:
            cls._bus_repo = BusReportRepositorySqlAlchemy()
        return cls._bus_repo

    @classmethod
    def create_bus_report_usecase(cls) -> CreateBusReportUseCase:
        if cls._bus_create_uc is None:
            cls._bus_create_uc = CreateBusReportUseCase(repo=cls.bus_report_repo())
        return cls._bus_create_uc

    @classmethod
    def get_bus_report_usecase(cls) -> GetBusReportUseCase:
        if cls._bus_get_uc is None:
            cls._bus_get_uc = GetBusReportUseCase(repo=cls.bus_report_repo())
        return cls._bus_get_uc

    @classmethod
    def list_bus_reports_usecase(cls) -> ListBusReportsUseCase:
        if cls._bus_list_uc is None:
            cls._bus_list_uc = ListBusReportsUseCase(repo=cls.bus_report_repo())
        return cls._bus_list_uc

    @classmethod
    def update_bus_report_usecase(cls) -> UpdateBusReportUseCase:
        if cls._bus_update_uc is None:
            cls._bus_update_uc = UpdateBusReportUseCase(repo=cls.bus_report_repo())
        return cls._bus_update_uc

    @classmethod
    def delete_bus_report_usecase(cls) -> DeleteBusReportUseCase:
        if cls._bus_delete_uc is None:
            cls._bus_delete_uc = DeleteBusReportUseCase(repo=cls.bus_report_repo())
        return cls._bus_delete_uc
