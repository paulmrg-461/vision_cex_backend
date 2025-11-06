import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass
class EnvironmentConfig:
    app_env: str = os.getenv("APP_ENV", "development")
    # Video source can be webcam index (e.g., "0"), a file path, or an RTSP/HTTP URL
    video_source: str = os.getenv("VIDEO_SOURCE", "0")
    # Weights for YOLO (Ultralytics or ONNX)
    yolo_weights: str = os.getenv("YOLO_WEIGHTS", "yolov8n.onnx")
    device: str = os.getenv("DEVICE", "auto")  # 'cuda', 'cpu', or 'auto'
    model_backend: str = os.getenv("MODEL_BACKEND", "onnx")  # 'onnx' | 'ultralytics' | 'grounding'
    model_input_size: int = int(os.getenv("MODEL_INPUT_SIZE", "640"))

    # GroundingDINO + SAM (open-vocabulary) settings
    grounding_prompt: str = os.getenv(
        "GROUNDING_PROMPT",
        "car wheel . car door . car window . bus wheel . bus door . bus window . llanta . puerta de coche . ventana de coche",
    )
    grounding_box_threshold: float = float(os.getenv("GROUNDING_BOX_THRESHOLD", "0.25"))
    grounding_text_threshold: float = float(os.getenv("GROUNDING_TEXT_THRESHOLD", "0.25"))
    grounding_download_dir: str = os.getenv("GROUNDING_DOWNLOAD_DIR", "models")
    grounding_config_path: str = os.getenv("GROUNDING_CONFIG_PATH", "")
    grounding_config_url: str = os.getenv(
        "GROUNDING_CONFIG_URL",
        "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinB_cfg.py",
    )
    grounding_weights_path: str = os.getenv("GROUNDING_WEIGHTS_PATH", "")
    grounding_weights_url: str = os.getenv(
        "GROUNDING_WEIGHTS_URL",
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0/groundingdino_swinb_cogcoor.pth",
    )
    sam_checkpoint_path: str = os.getenv("SAM_CHECKPOINT_PATH", "")
    sam_checkpoint_url: str = os.getenv(
        "SAM_CHECKPOINT_URL",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    )
    sam_model_type: str = os.getenv("SAM_MODEL_TYPE", "vit_h")