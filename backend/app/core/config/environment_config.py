import os
from dataclasses import dataclass

from dotenv import load_dotenv


# Ensure .env values override any empty defaults from the container environment.
# Explicitly point to /app/.env (WORKDIR) and set override=True to avoid blanks.
try:
    load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"), override=True)
except Exception:
    # Fallback to default behavior
    load_dotenv(override=True)


@dataclass
class EnvironmentConfig:
    app_env: str = os.getenv("APP_ENV", "development")
    # Video source can be webcam index (e.g., "0"), a file path, or an RTSP/HTTP URL
    video_source: str = os.getenv("VIDEO_SOURCE", "0")
    # Default to ONNX weights to align with the ONNX backend
    yolo_weights: str = os.getenv("YOLO_WEIGHTS", "yolov8n.onnx")
    device: str = os.getenv("DEVICE", "auto")  # 'cuda', 'cpu', or 'auto'
    model_backend: str = os.getenv("MODEL_BACKEND", "onnx")  # 'onnx' or 'ultralytics'
    # Task selection: 'detect' for bounding boxes, 'segment' for masks (Ultralytics)
    model_task: str = os.getenv("MODEL_TASK", "detect")
    model_input_size: int = int(os.getenv("MODEL_INPUT_SIZE", "640"))
    # Confidence threshold for Ultralytics inference (detect/segment)
    yolo_conf: float = float(os.getenv("YOLO_CONF", "0.25"))
    # YOLO específico para reconocimiento de placas
    yolo_plates_weights: str = os.getenv("YOLO_PLATES_WEIGHTS", "models/yolo_license_plates.pt")
    yolo_plates_conf: float = float(os.getenv("YOLO_PLATES_CONF", "0.25"))
    # OCR para lectura de placas
    lp_ocr_engine: str = os.getenv("LP_OCR_ENGINE", "paddle")  # 'paddle' (PaddleOCR)
    lp_ocr_lang: str = os.getenv("LP_OCR_LANG", "en")  # idioma para OCR
    # For segmentation: also draw bounding boxes around masks if true
    segment_draw_bbox: bool = os.getenv("SEGMENT_DRAW_BBOX", "true").lower() in ("1", "true", "yes", "on")
    # Clases permitidas para segmentación (lista separada por comas). Ejemplo: "bus,person"
    segment_allowed_classes: str = os.getenv("SEGMENT_ALLOWED_CLASSES", "bus")
    # Confianza mínima adicional para el filtro de segmentación (post-procesado en UseCase)
    segment_min_conf: float = float(os.getenv("SEGMENT_MIN_CONF", "0.5"))
    # Clases permitidas para detección (lista separada por comas). Ejemplo: "bus,truck"
    detect_allowed_classes: str = os.getenv("DETECT_ALLOWED_CLASSES", "bus")
    # Confianza mínima adicional para el filtro de detección (post-procesado en UseCase)
    detect_min_conf: float = float(os.getenv("DETECT_MIN_CONF", "0.5"))
    # Motion gating (HLS performance): threshold ratio [0..1] to trigger detection
    motion_change_threshold: float = float(os.getenv("MOTION_CHANGE_THRESHOLD", "0.35"))
    # Snapshot settings for bus detection events
    snapshot_interval_seconds: float = float(os.getenv("SNAPSHOT_INTERVAL_SECONDS", "0.3"))
    snapshot_max_count: int = int(os.getenv("SNAPSHOT_MAX_COUNT", "10"))
    snapshot_save_dir: str = os.getenv("SNAPSHOT_SAVE_DIR", "/app/samples/snapshots")
    # Confidence threshold to trigger snapshots on bus detection
    snapshot_detect_min_conf: float = float(os.getenv("SNAPSHOT_DETECT_MIN_CONF", "0.5"))
    # Enable snapshots for all sources (not only HLS). Default false for performance.
    snapshot_enable_all_sources: bool = os.getenv("SNAPSHOT_ENABLE_ALL_SOURCES", "false").lower() in ("1", "true", "yes", "on")
    # DeepSeek integration
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_api_base: str = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    # Hugging Face VQA integration
    huggingface_token: str = os.getenv("HUGGINGFACE_TOKEN", "")
    hf_inference_base: str = os.getenv("HF_INFERENCE_API_BASE", "https://api-inference.huggingface.co")
    hf_vqa_model: str = os.getenv("HF_VQA_MODEL", "dandelin/vilt-b32-finetuned-vqa")
    # Mode: 'api' to use Hugging Face Inference API, 'local' to run transformers locally
    hf_vqa_mode: str = os.getenv("HF_VQA_MODE", "api")
    # Hugging Face captioning
    hf_caption_model: str = os.getenv("HF_CAPTION_MODEL", "Salesforce/blip-image-captioning-base")

    # Database configuration (Postgres)
    db_host: str = os.getenv("DB_HOST", "postgres")
    db_port: int = int(os.getenv("DB_PORT", "5432"))
    db_user: str = os.getenv("DB_USER", "vision_cex")
    db_password: str = os.getenv("DB_PASSWORD", "vision_cex_pwd")
    db_name: str = os.getenv("DB_NAME", "vision_cex")
    hf_caption_mode: str = os.getenv("HF_CAPTION_MODE", "local")

    def __post_init__(self):
        """Resolve env overrides at instantiation time.
        Dataclass defaults are evaluated at module import, so runtime env changes
        (e.g., in tests via monkeypatch) would not be reflected without this.
        This method updates only the motion gating and snapshot-related settings
        to honor current environment variables.
        """
        # Motion gating threshold
        val = os.getenv("MOTION_CHANGE_THRESHOLD")
        if val is not None:
            try:
                self.motion_change_threshold = float(val)
            except Exception:
                pass

        # Snapshot interval seconds
        val = os.getenv("SNAPSHOT_INTERVAL_SECONDS")
        if val is not None:
            try:
                self.snapshot_interval_seconds = float(val)
            except Exception:
                pass

        # Snapshot max count
        val = os.getenv("SNAPSHOT_MAX_COUNT")
        if val is not None:
            try:
                self.snapshot_max_count = int(val)
            except Exception:
                pass

        # Snapshot save directory
        val = os.getenv("SNAPSHOT_SAVE_DIR")
        if val is not None:
            self.snapshot_save_dir = val

        # Snapshot detection min confidence
        val = os.getenv("SNAPSHOT_DETECT_MIN_CONF")
        if val is not None:
            try:
                conf = float(val)
                # Keep within [0.0, 1.0] if possible
                if 0.0 <= conf <= 1.0:
                    self.snapshot_detect_min_conf = conf
            except Exception:
                pass

        # Snapshot enable for all sources
        val = os.getenv("SNAPSHOT_ENABLE_ALL_SOURCES")
        if val is not None:
            try:
                self.snapshot_enable_all_sources = str(val).lower() in ("1", "true", "yes", "on")
            except Exception:
                pass
