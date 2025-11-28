# Pipeline de Streaming de Video (MP4, RTSP y HLS)

Este documento describe de forma detallada cómo el backend "Vision CEX" captura, procesa y sirve video en tiempo real desde archivos locales (`.mp4`) y fuentes de red (`rtsp://`, `http://`, `https://`, HLS `.m3u8`). Incluye el flujo por capas, los endpoints disponibles, el modelo de segmentación y los parámetros clave.

## Resumen del Flujo
- El cliente HTTP invoca endpoints bajo ` /api/v1/video`.
- La capa de Presentación (FastAPI) gestiona la fuente de video y expone un **stream MJPEG** con detección/segmentación y superposición visual.
- La capa de Dominio orquesta la inferencia: detección o segmentación según configuración (`model_task`).
- La capa de Datos provee el adaptador de **Ultralytics YOLO** para correr inferencia en CPU/GPU.
- La capa Core/Infra (DI y Config) entrega `ServiceLocator` y la configuración (`.env`).

## Arquitectura por Capas

- Presentación (FastAPI)
  - Archivo: `backend/app/presentation/api/main.py`
  - Router: `backend/app/presentation/api/v1/video_router.py`
  - Responsabilidades:
    - Gestión de fuentes de video (archivo local, RTSP/HTTP, HLS).
    - Endpoints `GET /stream` y `GET /hls/stream` que devuelven MJPEG.
    - Conversión y validación de parámetros (`roi`, `fps`, `loop`).
    - Resolución de HLS desde páginas HTML con extracción de `.m3u8`.

- Dominio (Use Cases y Entidades)
  - `backend/app/domain/usecases/segment_objects_usecase.py`
    - Método `segment(frame, roi)` ejecuta segmentación y traslada coordenadas desde `roi` al espacio global.
    - `draw_masks(frame, instances, alpha, draw_bboxes, ...)` superpone polígonos y etiquetas; opcionalmente dibuja bounding boxes.
  - Entidad `BoundingBox`: `backend/app/domain/entities/bbox_entity.py`.
  - Detect/Segment use cases se obtienen vía DI (`ServiceLocator.detect_usecase()` y `ServiceLocator.segment_usecase()`).

- Datos (Adapters)
  - `backend/app/data/adapters/yolo_ultralytics_adapter.py`
    - Carga `ultralytics.YOLO` con pesos configurables.
    - `detect(frame)` devuelve `List[BoundingBox]` con `xyxy`, `cls`, `conf`.
    - `segment(frame)` devuelve polígonos (`r.masks.xy`) y, si existen, bboxes y confidencias por instancia.

- Core / Infraestructura
  - `ServiceLocator` (en `app/core/di/service_locator.py`) provee adaptadores y configuración.
  - Logger (`app/core/utils/logger.py`).
  - `.env` define parámetros como `MODEL_TASK`, ruta de pesos, tamaño de entrada, etc.

## Fuentes de Video Soportadas

- Archivos locales
  - Extensiones: `.mp4`, `.avi`, `.mov`, `.mkv`.
  - Se abren con `cv2.VideoCapture(path, cv2.CAP_FFMPEG)` si son video de archivo.

- RTSP / HTTP
  - URLs que comienzan con `rtsp://`, `http://`, `https://` se abren con backend FFMPEG.

- HLS (`.m3u8`)
  - Si la URL termina en `.m3u8`, se usa directamente.
  - Si la URL apunta a una página HTML, `video_router.resolve_hls_url()` intenta extraer la primera URL `.m3u8`.
  - Comportamiento de verificación SSL configurable: `verify_ssl` admite `True`, `False` o `None` (estrategia con fallback).

## Endpoints Principales

- `GET /api/v1/video/source`
  - Devuelve la fuente de video actual.

- `POST /api/v1/video/source/file`
  - Body: `{ "path": "samples/Video1.mp4" }`
  - Valida que el archivo se pueda abrir; actualiza la fuente de video global.

- `POST /api/v1/video/source/rtsp`
  - Body: `{ "url": "rtsp://user:pass@host:port/path" }`
  - Valida apertura con FFMPEG; actualiza fuente de video global.

- `POST /api/v1/video/source/hls`
  - Body: `{ "url": "https://.../stream.html", "auto_find_m3u8": true, "verify_ssl": null }`
  - Resuelve `.m3u8` y valida. Actualiza fuente de video.

- `GET /api/v1/video/stream`
  - Stream MJPEG de la fuente global.
  - Parámetros:
    - `roi`: `"x,y,w,h"` para limitar inferencia a una región.
    - `fps`: número para limitar tasa de cuadros. Si no se especifica, para archivos se usa FPS del video (si disponible) o 25 por defecto; para tiempo real no se limita.
    - `loop`: en archivos, si `true`, rebobina al finalizar.

- `GET /api/v1/video/hls/stream`
  - Igual que `stream`, pero tomando HLS directo/HTML sin modificar la fuente global.
  - Optimización integrada: gating por movimiento usando `cv2.BackgroundSubtractorMOG2`.
    - Si la variación de píxeles excede 35% en el frame (o ROI), se activa YOLO para detección.
    - Si se detecta la clase `bus` con confianza > 0.5, se capturan snapshots parametrizables (cantidad e intervalo) y se guardan en el directorio configurado.

### Snapshots en cualquier fuente

Por defecto los snapshots solo se activan en streams HLS para proteger el rendimiento. Si necesitas capturas en cualquier fuente (RTSP, archivo MP4, URLs que no terminan en `.m3u8`), habilita la bandera de entorno:

```
SNAPSHOT_ENABLE_ALL_SOURCES=true
```

Con esta bandera, el `SnapshotScheduler` se creará para cualquier tipo de fuente y se disparará cuando se detecte `bus` con la confianza mínima configurada. El gating de movimiento sigue aplicándose únicamente a HLS.

- Gestión de múltiples fuentes en memoria
  - `GET /api/v1/video/sources` lista fuentes registradas.
  - `POST /api/v1/video/sources` agrega/actualiza una fuente por `id`.
  - `POST /api/v1/video/sources/bulk` alta masiva con resultados por ítem.
  - `DELETE /api/v1/video/sources/{id}` elimina fuente.
  - `GET /api/v1/video/{id}/stream` sirve MJPEG para una fuente específica.

## Generador MJPEG y Procesamiento de Frames

- `video_router.mjpeg_generator(video_source, roi, fps, loop)`:
  - Selección de backend: FFMPEG para URLs/HLS/archivos; `cv2.VideoCapture` estándar para cámaras locales.
  - FPS efectivo:
    - Si `fps` se especifica, se usa para limitar tasa.
    - Si no, intenta leer FPS de la captura; en archivos usa 25 si no disponible.
  - ROI: si se especifica, el frame se recorta para inferencia y luego se **trasladan** las coordenadas de salida al sistema global.
  - Inferencia:
    - Según `cfg.model_task`:
      - `segment`: usa `SegmentObjectsUseCase.segment(...)` y `draw_masks(...)`.
      - `detect`: usa `DetectObjectsUseCase.detect(...)` y `draw_boxes(...)`.
      - En HLS (`.m3u8`): para mejorar rendimiento, se aplica **detección condicionada por movimiento** (gating):
        - Se evalúa el porcentaje de píxeles en movimiento con `BackgroundSubtractorMOG2`.
        - Solo si el cambio supera 35% se activa la detección.
        - Ante detección de `bus` con `conf > 0.5`, se inicia un scheduler de snapshots (10 capturas, cada 0.3s) en `/app/samples/snapshots`.
  - Codificación:
    - JPEG con calidad 80 (`cv2.imencode`) y envío multipart `multipart/x-mixed-replace; boundary=frame`.

## Segmentación y Detección (Ultralytics YOLO)

- Adaptador: `YoloUltralyticsAdapter`
  - Inicialización:
    - Pesos: `YOLO(weights_path)`.
    - Device: `auto` selecciona GPU si disponible (`torch.cuda.is_available()`), si no CPU.
    - Parámetros: `imgsz` y `conf` se definen según configuración.
  - Detección (`detect`):
    - `results = model.predict(source=frame, imgsz, conf, device, half)`.
    - Extrae `xyxy`, `conf`, `cls` y retorna `BoundingBox` por instancia.
  - Segmentación (`segment`):
    - Usa el mismo `predict` y obtiene `r.masks.xy` (lista de puntos por instancia).
    - Asocia polígonos con cajas y confidencias cuando están disponibles.

- UseCase de segmentación: `SegmentObjectsUseCase`
  - Si se pasó `roi`, traduce polígonos y bboxes sumando el `x,y` del ROI al espacio global.
  - `draw_masks` crea un overlay semitransparente (`alpha=0.4`) y etiqueta cada máscara con la clase y confianza. Puede dibujar bounding boxes si `cfg.segment_draw_bbox` está habilitado.

## Configuración por `.env` (ejemplos útiles)

- `MODEL_TASK`: `detect` o `segment`.
- `YOLO_WEIGHTS`: ruta interna de pesos YOLO (p. ej. `/app/backend/weights/yolov8n.pt` o un `.pt` de segmentación).
- `MODEL_INPUT_SIZE`: tamaño de entrada para `imgsz` (p. ej. `640`, `960`, `1280`).
- `CONF_THRESHOLD`: umbral de confianza por defecto.
- `segment_draw_bbox`: `true/false` para dibujar cajas alrededor de las máscaras.
- `SEGMENT_ALLOWED_CLASSES`: lista separada por comas de clases permitidas en segmentación. Por ejemplo `bus` para que solo se muestren máscaras de buses. Por defecto: `bus`.
- `DETECT_ALLOWED_CLASSES`: lista separada por comas de clases permitidas en detección. Útil cuando `MODEL_TASK=detect`. Por ejemplo `bus` para que solo se dibujen cajas de buses. Por defecto: `bus`.
- `video_source`: fuente inicial (p. ej. `samples/Video1.mp4`).

> Notas de optimización HLS:
> - El gating por movimiento está habilitado automáticamente cuando la fuente es HLS (`.m3u8`).
> - Variables `.env` para parametrizar:
>   - `MOTION_CHANGE_THRESHOLD` (por defecto `0.35`).
>   - `SNAPSHOT_INTERVAL_SECONDS` (por defecto `0.3`).
>   - `SNAPSHOT_MAX_COUNT` (por defecto `10`).
>   - `SNAPSHOT_SAVE_DIR` (por defecto `/app/samples/snapshots`).
>   - `SNAPSHOT_DETECT_MIN_CONF` (por defecto `0.5`): confianza mínima para disparar snapshots.
>   - `SNAPSHOT_ENABLE_ALL_SOURCES` (por defecto `false`): habilita snapshots también en RTSP/archivo/URLs que no terminen en `.m3u8`.
> - Las capturas se guardan en `SNAPSHOT_SAVE_DIR`; asegúrate de montar `/app` como volumen en Docker para persistencia.

> Nota: La apertura de URLs y archivos se realiza con FFMPEG (`cv2.CAP_FFMPEG`) cuando corresponde.

## Ejemplos de Uso (curl)

- Establecer archivo local y hacer streaming:

```bash
curl -X POST http://localhost:8000/api/v1/video/source/file \
  -H "Content-Type: application/json" \
  -d '{"path":"samples/Video1.mp4"}'

curl -X GET "http://localhost:8000/api/v1/video/stream?fps=25&loop=true" \
  -H "Accept: multipart/x-mixed-replace"
```

- Establecer RTSP y hacer streaming:

```bash
curl -X POST http://localhost:8000/api/v1/video/source/rtsp \
  -H "Content-Type: application/json" \
  -d '{"url":"rtsp://user:pass@host:port/path"}'

curl -X GET "http://localhost:8000/api/v1/video/stream?fps=15" \
  -H "Accept: multipart/x-mixed-replace"
```

- HLS desde página HTML (auto-resolución .m3u8) y streaming directo sin cambiar fuente global:

```bash
curl -X GET "http://localhost:8000/api/v1/video/hls/stream?url=https://ejemplo.com/stream.html&fps=25" \
  -H "Accept: multipart/x-mixed-replace"
```

- Usar ROI para concentrar la inferencia en una región del frame:

```bash
curl -X GET "http://localhost:8000/api/v1/video/stream?roi=320,200,640,480&fps=20" \
  -H "Accept: multipart/x-mixed-replace"
```

- Limitar segmentación a la clase "bus" mediante `.env`:

```bash
# .env
SEGMENT_ALLOWED_CLASSES=bus
SEGMENT_MIN_CONF=0.5
MODEL_TASK=segment
MODEL_BACKEND=ultralytics
YOLO_WEIGHTS=/app/backend/weights/yolov8n-seg.pt
MODEL_INPUT_SIZE=960
```

- Limitar detección a la clase "bus" (si `MODEL_TASK=detect`):

```bash
# .env
DETECT_ALLOWED_CLASSES=bus
MODEL_TASK=detect
MODEL_BACKEND=onnx   # o ultralytics, según tus pesos
YOLO_WEIGHTS=/app/backend/weights/yolov8n.onnx
MODEL_INPUT_SIZE=960
```

## Consideraciones de Seguridad y Robustez

- Verificación SSL en HLS:
  - `verify_ssl` controla si se valida certificado. Si es `None`, se intenta primero verificado y se reintenta sin verificación si el error sugiere problema de certificado.
- CORS: habilitado de forma permisiva en desarrollo; endurecer en producción.
- Credenciales RTSP/HTTP: evitar exponer usuarios/contraseñas; usar redes seguras y/o proxys.
- Manejo de FPS: limitarlo reduce carga del servidor y ancho de banda.
- Validación de fuentes: cada alta de fuente prueba apertura con OpenCV antes de aceptar.

## Errores Comunes y Soluciones

- "No se pudo abrir la fuente de video":
  - Verificar ruta/URL y accesibilidad; para HLS, confirmar que la URL final es `.m3u8` válido.
- "No se encontró URL .m3u8 en la página":
  - Usar directamente la `.m3u8` o ajustar `auto_find_m3u8`.
- Latencia alta o caídas en RTSP/HLS:
  - Ajustar `fps`, usar un backend FFMPEG actualizado, mejorar red.
- Sin resultados de segmentación/detección:
  - Revisar `MODEL_TASK`, pesos (`YOLO_WEIGHTS`) compatibles y `MODEL_INPUT_SIZE`.

## Alineación con Arquitectura

- Se emplea una arquitectura por capas (Presentación, Dominio, Datos, Core/Infra) en línea con principios de Clean Architecture y SOLID.
- Los endpoints no dependen directamente de librerías de inferencia; se delega a use cases y adaptadores inyectados por DI.
- La configuración está centralizada y desacoplada del flujo HTTP.

## Referencias de Código

- Presentación
  - `backend/app/presentation/api/main.py`
  - `backend/app/presentation/api/v1/video_router.py`

- Dominio
  - `backend/app/domain/usecases/segment_objects_usecase.py`
  - `backend/app/domain/entities/bbox_entity.py`

- Datos
  - `backend/app/data/adapters/yolo_ultralytics_adapter.py`

- Core/Infra
  - `app/core/di/service_locator.py`
  - `app/core/utils/logger.py`

---

Si necesitas incluir diagramas o ejemplos adicionales (p. ej. configuración de `.env` específica para segmentación), indícame el contexto y los agrego.
