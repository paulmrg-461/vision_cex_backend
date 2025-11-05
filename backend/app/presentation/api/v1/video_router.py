from typing import Optional, Tuple

import cv2
from fastapi import APIRouter, Response

from app.core.di.service_locator import ServiceLocator
from app.core.utils.logger import get_logger


router = APIRouter(prefix="/api/v1/video", tags=["video"])
logger = get_logger("video")


def parse_roi(roi: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if not roi:
        return None
    try:
        x, y, w, h = [int(v) for v in roi.split(",")]
        return x, y, w, h
    except Exception:
        logger.warning("ROI inválido, se ignora: %s", roi)
        return None


def mjpeg_generator(video_source: int, roi: Optional[str] = None):
    roi_rect = parse_roi(roi)
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        logger.error("No se pudo abrir la cámara: %s", video_source)
        return

    detect_usecase = ServiceLocator.detect_usecase()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.error("Error leyendo frame de la cámara.")
                break

            boxes = detect_usecase.detect(frame, roi=roi_rect)
            DetectObjectsUseCase = type(detect_usecase)  # for static method access
            DetectObjectsUseCase.draw_boxes(frame, boxes)

            # Encode JPEG
            ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                logger.error("Error codificando frame a JPEG.")
                continue
            frame_bytes = jpg.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
    finally:
        cap.release()


@router.get("/stream", response_class=Response)
def stream(roi: Optional[str] = None):
    """Devuelve un stream MJPEG de la cámara con detección y cajas dibujadas.

    Parámetros:
    - roi: "x,y,w,h" para limitar la detección a una región del frame.
    """
    cfg = ServiceLocator.config()
    return Response(
        content=mjpeg_generator(cfg.video_source, roi=roi),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )