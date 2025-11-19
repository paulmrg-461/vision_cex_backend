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
    # For segmentation: also draw bounding boxes around masks if true
    segment_draw_bbox: bool = os.getenv("SEGMENT_DRAW_BBOX", "true").lower() in ("1", "true", "yes", "on")
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
    hf_caption_mode: str = os.getenv("HF_CAPTION_MODE", "local")