from typing import Optional, Tuple, Dict, List

import cv2
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import time
import re
from urllib.parse import urljoin, urlparse, parse_qs

try:
    import requests  # type: ignore
    _requests_available = True
except Exception:
    requests = None  # type: ignore
    _requests_available = False

try:
    import urllib3  # type: ignore
    from urllib3.exceptions import InsecureRequestWarning  # type: ignore
    _urllib3_available = True
except Exception:
    urllib3 = None  # type: ignore
    InsecureRequestWarning = None  # type: ignore
    _urllib3_available = False

from app.core.di.service_locator import ServiceLocator
from app.core.utils.logger import get_logger


router = APIRouter(prefix="/api/v1/video", tags=["video"])
logger = get_logger("video")

# Simple in-memory registry to manage multiple video sources by ID
# Example: {"1": "samples/Video1.mp4", "2": "rtsp://user:pass@ip/..."}
sources_registry: Dict[str, str] = {}


def preload_default_sources():
    """Preload sample sources Video1.mp4 .. Video8.mp4 at startup if available/openable."""
    for i in range(1, 9):
        source_id = str(i)
        path = os.path.join("samples", f"Video{i}.mp4")
        lower = path.lower()
        use_ffmpeg = lower.endswith((".mp4", ".avi", ".mov", ".mkv"))
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG) if use_ffmpeg else cv2.VideoCapture(path)
        ok = cap.isOpened()
        cap.release()
        if ok:
            sources_registry[source_id] = path
            logger.info("Fuente precargada: id=%s, source=%s", source_id, path)
        else:
            logger.warning("No se pudo precargar fuente %s (%s). Archivo inexistente o no abrible.", source_id, path)


def parse_roi(roi: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if not roi:
        return None
    try:
        x, y, w, h = [int(v) for v in roi.split(",")]
        return x, y, w, h
    except Exception:
        logger.warning("ROI inválido, se ignora: %s", roi)
        return None


def mjpeg_generator(video_source, roi: Optional[str] = None, fps: Optional[float] = None, loop: bool = False):
    """Generador MJPEG con detección.

    - Control de cadencia mediante "fps": si se especifica, limita la tasa de envío.
    - Para fuentes de archivo, si loop=True, rebobina al finalizar.
    """
    roi_rect = parse_roi(roi)
    # Allow both integer webcam index (e.g., "0") and string sources (file path/RTSP)
    src = video_source
    is_str = isinstance(src, str)
    if is_str and src.isdigit():
        src = int(src)

    # Decide backend y tipo de fuente
    use_ffmpeg = False
    is_file = False
    is_url = False
    if isinstance(src, str):
        lower = src.lower()
        is_file = lower.endswith((".mp4", ".avi", ".mov", ".mkv")) or os.path.isfile(src)
        is_url = lower.startswith("rtsp://") or lower.startswith("http://") or lower.startswith("https://")
        use_ffmpeg = is_url or is_file

    # Abrir captura con backend apropiado
    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG) if use_ffmpeg else cv2.VideoCapture(src)
    if not cap.isOpened():
        logger.error("No se pudo abrir la fuente de video: %s", video_source)
        return

    # Determinar FPS efectivo
    target_fps = fps
    if target_fps is None:
        # Intentar leer FPS del archivo/cámara
        read_fps = cap.get(cv2.CAP_PROP_FPS)
        if read_fps and read_fps > 0 and read_fps < 120:
            target_fps = read_fps
        else:
            target_fps = 25.0 if is_file else None  # en archivos, por defecto 25; en tiempo real no limitar

    delay = (1.0 / target_fps) if target_fps and target_fps > 0 else 0.0

    cfg = ServiceLocator.config()
    is_segment_task = (cfg.model_task.lower() == "segment")
    detect_usecase = ServiceLocator.detect_usecase()
    segment_usecase = ServiceLocator.segment_usecase() if is_segment_task else None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if is_file:
                    logger.info("Fin de archivo o fallo de lectura. loop=%s", loop)
                    if loop:
                        # Intentar rebobinar al inicio
                        reset_ok = cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        if not reset_ok:
                            # Algunos backends no soportan rebobinado: reabrir captura
                            cap.release()
                            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG) if use_ffmpeg else cv2.VideoCapture(src)
                        # pequeña espera para evitar bucle rápido
                        time.sleep(0.05)
                        continue
                    else:
                        break
                else:
                    logger.error("Error leyendo frame de la cámara/stream.")
                    # En cámaras/RTSP: intentar un pequeño retry
                    time.sleep(0.05)
                    continue

            # Detección o segmentación y overlay
            if is_segment_task and segment_usecase is not None:
                instances = segment_usecase.segment(frame, roi=roi_rect)
                SegmentObjectsUseCase = type(segment_usecase)
                # Draw masks and optional bounding boxes around masks based on environment config
                SegmentObjectsUseCase.draw_masks(frame, instances, alpha=0.4, draw_bboxes=cfg.segment_draw_bbox)
            else:
                boxes = detect_usecase.detect(frame, roi=roi_rect)
                DetectObjectsUseCase = type(detect_usecase)  # acceso al método estático
                DetectObjectsUseCase.draw_boxes(frame, boxes)

            # Encode JPEG
            ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                logger.error("Error codificando frame a JPEG.")
                # Avanzar al siguiente frame
                continue
            frame_bytes = jpg.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

            # Control de cadencia
            if delay > 0:
                time.sleep(delay)

    finally:
        cap.release()


class FileSourceRequest(BaseModel):
    path: str


class RtspSourceRequest(BaseModel):
    url: str

class HlsSourceRequest(BaseModel):
    url: str  # Puede ser una página HTML que contenga el .m3u8 o el .m3u8 directo
    auto_find_m3u8: bool = True  # si es true y la URL no termina en .m3u8, intenta extraerlo de la página
    # verify_ssl:
    #   - None: intentar primero con verificación y, si falla por SSL, reintentar sin verificación automáticamente.
    #   - True/False: respetar el valor explícito sin fallback automático.
    verify_ssl: Optional[bool] = None

class SourceItem(BaseModel):
    id: str
    source: str  # file path or RTSP/HTTP URL or webcam index as string


class BulkSourcesRequest(BaseModel):
    items: List[SourceItem]


@router.get("/source")
def get_source():
    cfg = ServiceLocator.config()
    return {"video_source": cfg.video_source}


@router.post("/source/file")
def set_source_file(req: FileSourceRequest):
    path = (req.path or "").strip()
    if not path:
        raise HTTPException(status_code=400, detail="Ruta de archivo vacía")

    # Validación rápida: intentar abrir el archivo
    use_ffmpeg = path.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG) if use_ffmpeg else cv2.VideoCapture(path)
    ok = cap.isOpened()
    cap.release()
    if not ok:
        raise HTTPException(status_code=400, detail=f"No se pudo abrir el archivo de video: {path}")

    cfg = ServiceLocator.config()
    cfg.video_source = path
    logger.info("Fuente de video actualizada (archivo): %s", path)
    return {"message": "Fuente de video actualizada (archivo)", "video_source": cfg.video_source}


@router.post("/source/rtsp")
def set_source_rtsp(req: RtspSourceRequest):
    url = (req.url or "").strip()
    if not url or not (url.startswith("rtsp://") or url.startswith("http://") or url.startswith("https://")):
        raise HTTPException(status_code=400, detail="URL inválida. Debe comenzar con rtsp://, http:// o https://")

    # Validación rápida con backend FFMPEG
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    ok = cap.isOpened()
    cap.release()
    if not ok:
        raise HTTPException(status_code=400, detail=f"No se pudo abrir la URL RTSP/HTTP: {url}")

    cfg = ServiceLocator.config()
    cfg.video_source = url
    logger.info("Fuente de video actualizada (RTSP/HTTP): %s", url)
    return {"message": "Fuente de video actualizada (RTSP/HTTP)", "video_source": cfg.video_source}


@router.get("/sources")
def list_sources():
    """List all registered sources in the in-memory registry."""
    return {"sources": sources_registry}


@router.post("/sources")
def add_or_update_source(item: SourceItem):
    """Add or update a video source by ID.

    Validates the source by attempting to open it with OpenCV (FFMPEG for URLs/files).
    """
    src = (item.source or "").strip()
    if not item.id or not src:
        raise HTTPException(status_code=400, detail="Debe proporcionar 'id' y 'source'.")

    lower = src.lower()
    use_ffmpeg = lower.startswith("rtsp://") or lower.startswith("http://") or lower.startswith("https://") \
                 or lower.endswith((".mp4", ".avi", ".mov", ".mkv"))

    # Try open
    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG) if use_ffmpeg else cv2.VideoCapture(src)
    ok = cap.isOpened()
    cap.release()
    if not ok:
        raise HTTPException(status_code=400, detail=f"No se pudo abrir el origen de video: {src}")

    sources_registry[item.id] = src
    logger.info("Fuente registrada: id=%s, source=%s", item.id, src)
    return {"message": "Fuente registrada/actualizada", "id": item.id, "source": src}


@router.post("/sources/bulk")
def bulk_add_sources(req: BulkSourcesRequest):
    """Bulk add/update sources. Returns per-item results and errors without aborting the whole request."""
    results = []
    for item in req.items:
        try:
            src = (item.source or "").strip()
            if not item.id or not src:
                raise HTTPException(status_code=400, detail="Debe proporcionar 'id' y 'source'.")

            lower = src.lower()
            use_ffmpeg = lower.startswith("rtsp://") or lower.startswith("http://") or lower.startswith("https://") \
                         or lower.endswith((".mp4", ".avi", ".mov", ".mkv"))

            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG) if use_ffmpeg else cv2.VideoCapture(src)
            ok = cap.isOpened()
            cap.release()
            if not ok:
                results.append({"id": item.id, "source": src, "status": "error", "detail": "No se pudo abrir el origen de video"})
                continue

            sources_registry[item.id] = src
            logger.info("Fuente registrada: id=%s, source=%s", item.id, src)
            results.append({"id": item.id, "source": src, "status": "ok"})
        except HTTPException as he:
            results.append({"id": item.id, "source": item.source, "status": "error", "detail": he.detail})
        except Exception as e:
            results.append({"id": item.id, "source": item.source, "status": "error", "detail": str(e)})

    return {"results": results}


@router.delete("/sources/{source_id}")
def delete_source(source_id: str):
    if source_id in sources_registry:
        removed = sources_registry.pop(source_id)
        logger.info("Fuente eliminada: id=%s, source=%s", source_id, removed)
        return {"message": "Fuente eliminada", "id": source_id}
    else:
        raise HTTPException(status_code=404, detail="Fuente no encontrada")


def resolve_hls_url(url: str, auto_find_m3u8: bool = True, verify_ssl: Optional[bool] = None) -> str:
    """Resuelve una URL HLS. Si 'url' es una página HTML, intenta extraer la primera URL .m3u8.

    - Si la URL ya termina en .m3u8, se devuelve tal cual.
    - Si auto_find_m3u8 es True y la URL no termina en .m3u8, descarga la página y busca patrones de .m3u8.
    """
    lower = url.lower()
    if lower.endswith(".m3u8"):
        return url

    if not auto_find_m3u8:
        return url  # devolver la URL original; cv2 intentará abrirla si es un stream directo

    if not _requests_available:
        raise HTTPException(status_code=500, detail="'requests' no está disponible en el contenedor para resolver HLS")

    # Helper para peticiones con control de verificación
    def _fetch(_url: str, _verify: bool):
        if not _verify and _urllib3_available and InsecureRequestWarning is not None:
            urllib3.disable_warnings(InsecureRequestWarning)
        return requests.get(_url, timeout=10, verify=_verify)

    # Estrategia de verificación:
    # - verify_ssl is None: probar con verify=True y si falla por SSL, reintentar con verify=False.
    # - verify_ssl True/False: usar tal cual sin fallback.
    initial_verify = True if verify_ssl is None else bool(verify_ssl)

    # Heurística: si la URL parece ser una página go2rtc stream.html con parámetro src,
    # construir directamente endpoints HLS conocidos sin necesidad de parsear HTML.
    try:
        parsed = urlparse(url)
        path = parsed.path or ""
        qs = parse_qs(parsed.query or "")
        base = f"{parsed.scheme}://{parsed.netloc}"
        if "stream.html" in path and "src" in qs and qs["src"]:
            stream_id = qs["src"][0]
            candidates = [
                f"{base}/api/stream.m3u8?src={stream_id}",
                f"{base}/stream.m3u8?src={stream_id}",
            ]
            if "/liveviews/" in path:
                candidates.append(f"{base}/liveviews/api/stream.m3u8?src={stream_id}")
            for cand in candidates:
                try:
                    resp = _fetch(cand, initial_verify)
                    if resp.status_code in (200, 206):
                        return cand
                    if resp.status_code in (301, 302, 303, 307, 308):
                        loc = resp.headers.get("Location")
                        if loc and loc.lower().endswith(".m3u8"):
                            resolved = urljoin(cand, loc)
                            resp2 = _fetch(resolved, initial_verify)
                            if resp2.status_code in (200, 206):
                                return resolved
                except Exception:
                    if verify_ssl is None:
                        try:
                            resp = _fetch(cand, False)
                            if resp.status_code in (200, 206):
                                return cand
                        except Exception:
                            pass
            # Si ningún candidato funciona, continuar con la resolución estándar
    except Exception:
        pass
    try:
        resp = _fetch(url, initial_verify)
        resp.raise_for_status()
        html = resp.text
        # Buscar la primera coincidencia .m3u8 en la página
        # Captura URLs absolutas o relativas dentro de comillas
        matches = re.findall(r"([\"\'])(?P<u>[^\"\']+\.m3u8)(?:\?[^\"\']*)?\1", html)
        candidate = None
        for m in matches:
            u = m[1]
            candidate = u
            break
        if not candidate:
            # Buscar sin comillas como fallback
            plain = re.findall(r"(?P<u>https?://[^\s'\"]+\.m3u8(?:\?[^\s'\"]*)?)", html)
            if plain:
                candidate = plain[0]
        if not candidate:
            raise HTTPException(status_code=404, detail="No se encontró ninguna URL .m3u8 en la página proporcionada")
        # Resolver si es relativa
        resolved = urljoin(url, candidate)
        return resolved
    except HTTPException:
        raise
    except Exception as e:
        # Si no se indicó verify_ssl y el error sugiere problema de certificado, reintentar sin verificación
        if verify_ssl is None:
            try:
                from requests.exceptions import SSLError as RequestsSSLError  # type: ignore
                is_ssl_error = isinstance(e, RequestsSSLError)
            except Exception:
                is_ssl_error = "CERTIFICATE_VERIFY_FAILED" in str(e) or "SSLCertVerificationError" in str(e)

            if is_ssl_error:
                try:
                    resp = _fetch(url, False)
                    resp.raise_for_status()
                    html = resp.text
                    matches = re.findall(r"([\"\'])(?P<u>[^\"\']+\.m3u8)(?:\?[^\"\']*)?\1", html)
                    candidate = None
                    for m in matches:
                        u = m[1]
                        candidate = u
                        break
                    if not candidate:
                        plain = re.findall(r"(?P<u>https?://[^\s'\"]+\.m3u8(?:\?[^\s'\"]*)?)", html)
                        if plain:
                            candidate = plain[0]
                    if not candidate:
                        raise HTTPException(status_code=404, detail="No se encontró ninguna URL .m3u8 en la página proporcionada (retry insecure)")
                    resolved = urljoin(url, candidate)
                    return resolved
                except HTTPException:
                    raise
                except Exception as e2:
                    raise HTTPException(status_code=500, detail=f"Error resolviendo HLS (retry insecure): {e2}")
        # Sin fallback: devolver error original
        raise HTTPException(status_code=500, detail=f"Error resolviendo HLS: {e}")


@router.post("/source/hls")
def set_source_hls(req: HlsSourceRequest):
    """Establece la fuente de video desde una URL HLS (.m3u8) o una página que contenga HLS.

    - Si 'url' es una página HTML (por ejemplo, stream.html?mode=hls), intenta extraer la primera URL .m3u8.
    - Valida la captura con OpenCV FFMPEG.
    """
    if not req.url:
        raise HTTPException(status_code=400, detail="URL vacía")

    final_url = resolve_hls_url(req.url, auto_find_m3u8=req.auto_find_m3u8, verify_ssl=req.verify_ssl)

    cap = cv2.VideoCapture(final_url, cv2.CAP_FFMPEG)
    ok = cap.isOpened()
    cap.release()
    if not ok:
        raise HTTPException(status_code=400, detail=f"No se pudo abrir la URL HLS: {final_url}")

    cfg = ServiceLocator.config()
    cfg.video_source = final_url
    logger.info("Fuente HLS establecida: %s -> %s", req.url, final_url)
    return {"message": "Fuente HLS actualizada", "video_source": cfg.video_source}


@router.get("/hls/stream")
def stream_hls(url: str, roi: Optional[str] = None, fps: Optional[float] = None, verify_ssl: Optional[bool] = None):
    """Stream MJPEG directamente desde una URL HLS o una página que contenga .m3u8.

    - No modifica la fuente global; usa la URL proporcionada.
    - Si 'url' no termina en .m3u8, intenta extraerla del HTML.
    """
    final_url = resolve_hls_url(url, auto_find_m3u8=True, verify_ssl=verify_ssl)
    # Validación rápida
    cap = cv2.VideoCapture(final_url, cv2.CAP_FFMPEG)
    ok = cap.isOpened()
    cap.release()
    if not ok:
        raise HTTPException(status_code=400, detail=f"No se pudo abrir la URL HLS: {final_url}")

    return StreamingResponse(
        mjpeg_generator(final_url, roi=roi, fps=fps, loop=False),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/stream")
def stream(roi: Optional[str] = None, fps: Optional[float] = None, loop: bool = False):
    """Devuelve un stream MJPEG de la cámara con detección y cajas dibujadas.

    Parámetros:
    - roi: "x,y,w,h" para limitar la detección a una región del frame.
    - fps: límite de frames por segundo (solo aplicable si se especifica o si la fuente es archivo).
    - loop: si la fuente es un archivo, rebobina al finalizar.
    """
    cfg = ServiceLocator.config()
    return StreamingResponse(
        mjpeg_generator(cfg.video_source, roi=roi, fps=fps, loop=loop),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/{source_id}/stream")
def stream_by_id(source_id: str, roi: Optional[str] = None, fps: Optional[float] = None, loop: bool = False):
    """Stream MJPEG for a specific registered source ID."""
    src = sources_registry.get(source_id)
    if not src:
        # Fallback: if source_id is numeric and a samples/Video{N}.mp4 exists, use it automatically
        if source_id.isdigit():
            fallback_path = os.path.join("samples", f"Video{int(source_id)}.mp4")
            lower = fallback_path.lower()
            use_ffmpeg = lower.endswith((".mp4", ".avi", ".mov", ".mkv"))
            cap = cv2.VideoCapture(fallback_path, cv2.CAP_FFMPEG) if use_ffmpeg else cv2.VideoCapture(fallback_path)
            ok = cap.isOpened()
            cap.release()
            if ok:
                logger.info("Usando fallback para id=%s -> %s", source_id, fallback_path)
                sources_registry[source_id] = fallback_path
                src = fallback_path
        if not src:
            raise HTTPException(status_code=404, detail=f"Fuente no encontrada: {source_id}")
    return StreamingResponse(
        mjpeg_generator(src, roi=roi, fps=fps, loop=loop),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )