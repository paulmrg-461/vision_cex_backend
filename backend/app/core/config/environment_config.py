import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass
class EnvironmentConfig:
    app_env: str = os.getenv("APP_ENV", "development")
    video_source: int = int(os.getenv("VIDEO_SOURCE", "0"))
    yolo_weights: str = os.getenv("YOLO_WEIGHTS", "yolov8n.pt")
    device: str = os.getenv("DEVICE", "auto")  # 'cuda', 'cpu', or 'auto'