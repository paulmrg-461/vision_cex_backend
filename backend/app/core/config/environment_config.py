import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


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