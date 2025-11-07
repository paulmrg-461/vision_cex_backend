# Vision CEX Backend

FastAPI backend for real-time object detection and MJPEG video streaming. It supports two detection backends:

- Ultralytics YOLO (PyTorch) — supports YOLOv8 and YOLOv11 weights (CPU or CUDA if available)
- ONNX Runtime — lightweight inference with pre-exported YOLO ONNX models

The service provides endpoints to select the video source (file path or RTSP/HTTP URL) and a streaming endpoint that overlays bounding boxes and labels on each frame.

## Features

- FastAPI + Uvicorn server (port 8000)
- MJPEG streaming endpoint with bounding boxes and labels
- Switchable detection backend: Ultralytics (PT) or ONNX Runtime
- Video source management endpoints (file path or RTSP/HTTP)
- FPS throttling and loop support for video files
- Optional ROI (region of interest) detection

## Project Structure

```
vision_cex_backend/
├── backend/
│   └── app/
│       ├── core/                # config, DI, utils
│       ├── data/                # detector adapters (ONNX, Ultralytics)
│       ├── domain/              # entities & use cases
│       └── presentation/        # FastAPI routes
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── samples/                     # place your .mp4 files here
├── requirements.txt
├── .env                         # runtime configuration (loaded by docker-compose)
├── .env.example                 # example configuration
└── README.md
```

Note: `.gitignore` excludes `*.mp4`. Place test videos under `samples/` so they are mounted into the container but not committed to Git.

## Prerequisites

- Docker and Docker Compose
- Optional: NVIDIA GPU drivers and Docker GPU support (for CUDA acceleration)
  - docker-compose uses `runtime: nvidia` in the `fastapi` service. If your host does not expose a CUDA device to containers, the app will fall back to CPU.

## Configuration (.env)

Create a `.env` file at the repository root. Example configurations:

Ultralytics (recommended for immediate boxes):

```
MODEL_BACKEND=ultralytics
YOLO_WEIGHTS=yolov8n.pt
DEVICE=cpu
VIDEO_SOURCE=0
MODEL_INPUT_SIZE=640
```

ONNX Runtime (lightweight inference):

```
MODEL_BACKEND=onnx
YOLO_WEIGHTS=yolov8n.onnx
DEVICE=cpu
VIDEO_SOURCE=0
MODEL_INPUT_SIZE=640
```

Notes:
- `VIDEO_SOURCE` can be a webcam index (e.g., `0`), a relative file path (e.g., `samples/Video2.mp4`), or an RTSP/HTTP URL.
- For Windows hosts running Linux containers, host webcams are not accessible via index `0`. Use a video file or an IP/RTSP camera.
- For Ultralytics with GPU, set `DEVICE=0` (first CUDA GPU) if the container sees CUDA.

Segmentation vs Detection:
- Set `MODEL_TASK=detect` to draw bounding boxes.
- Set `MODEL_TASK=segment` to draw semi-transparent masks with labels (Ultralytics segmentation models only).
- Ensure `YOLO_WEIGHTS` matches the task (e.g., `yolov8n.pt` for detect, `yolov8n-seg.pt` or custom `best.pt` for segment).
 - Optional: set `SEGMENT_DRAW_BBOX=true` to also draw bounding boxes around segmentation masks.

## Build and Run

From the repository root:

```
docker compose -f docker/docker-compose.yml up --build
```

Run detached:

```
docker compose -f docker/docker-compose.yml up --build -d
```

The server starts at:

```
http://localhost:8000/
```

API docs (Swagger UI):

```
http://localhost:8000/docs
```

If you edit code or `.env`, restart the service:

```
docker compose -f docker/docker-compose.yml restart fastapi
```

## Endpoints

Base path: `/api/v1/video`

- GET `/source`
  - Returns the current video source.
  - Response: `{ "video_source": "samples/Video2.mp4" }`

- POST `/source/file`
  - Set video source to a local file path inside the container.
  - Use paths relative to the repo mount (e.g., `samples/Video2.mp4`). Do not use Windows paths like `C:\\...`.
  - Body (JSON): `{ "path": "samples/Video2.mp4" }`

- POST `/source/rtsp`
  - Set video source to an RTSP or HTTP URL.
  - Body (JSON): `{ "url": "rtsp://user:pass@ip:554/Streaming/Channels/101?transport=tcp" }`

- GET `/stream`
  - Returns an MJPEG stream with detections (open in a browser).
  - Query params:
    - `fps`: target frames per second (e.g., `10`, `25`, `30`). If file source, limits playback speed.
    - `loop`: `true` to loop a file when it reaches the end.
    - `roi`: region of interest as `x,y,w,h` (e.g., `100,100,400,400`). Detection is applied only inside this rectangle.
  - Example:
    - `http://localhost:8000/api/v1/video/stream?fps=25&loop=true`
    - `http://localhost:8000/api/v1/video/stream?fps=10&loop=true&roi=100,100,400,400`

Note: If your router version supports source indexing, you may also use:
`http://localhost:8000/api/v1/video/2/stream?fps=30&loop=true`

### cURL Examples

Set a file source:

```
curl -X POST "http://localhost:8000/api/v1/video/source/file" \
     -H "Content-Type: application/json" \
     -d '{"path":"samples/Video2.mp4"}'
```

Set an RTSP source:

```
curl -X POST "http://localhost:8000/api/v1/video/source/rtsp" \
     -H "Content-Type: application/json" \
     -d '{"url":"rtsp://user:pass@ip:554/Streaming/Channels/101?transport=tcp"}'
```

Open the MJPEG stream in your browser:

```
http://localhost:8000/api/v1/video/stream?fps=25&loop=true
```

## Models

- Ultralytics:
  - Default weights: `yolov8n.pt` (place at repo root or provide a path in `.env`).
  - The service auto-labels classes from the model.
  - For segmentation set `MODEL_TASK=segment` and use `yolov8n-seg.pt` or a custom trained `best.pt`.
- ONNX Runtime:
  - Use a pre-exported ONNX model (e.g., `yolov8n.onnx`). Place it at the repo root or set a path.
  - To export from Ultralytics locally: `yolo export model=yolov8n.pt format=onnx`
  - COCO class names are included by default; adjust if your model uses a different dataset.

## Training a car-parts segmentation model (Ultralytics)
This backend includes a script and dataset template to train segmentation for car parts like `llanta`, `puerta`, `ventana`, `parachoques`, `faro`.

1) Prepare dataset (YOLO segmentation format):
```
backend/datasets/carparts-seg/
  images/train/*.jpg
  images/val/*.jpg
  labels/train/*.txt
  labels/val/*.txt
  data.yaml
```
An example `data.yaml` is provided at `backend/datasets/carparts-seg/data.yaml`.

2) Start Docker:
```
docker compose -f docker/docker-compose.yml up --build -d
```

3) Train inside the container:
```
docker compose -f docker/docker-compose.yml exec fastapi \
  python backend/scripts/train_carparts_seg.py \
  --data /app/backend/datasets/carparts-seg/data.yaml \
  --model yolov8n-seg.pt \
  --epochs 100 \
  --imgsz 640 \
  --device auto
```

Alternatively, to use Ultralytics' public Carparts-Seg dataset and YOLO11:
```
docker compose -f docker/docker-compose.yml exec fastapi \
  python -c "from ultralytics import YOLO; m=YOLO('yolo11n-seg.pt'); m.train(data='carparts-seg.yaml', epochs=100, imgsz=640)"
```

4) Use the trained weights:
Update `.env` with:
```
MODEL_BACKEND=ultralytics
MODEL_TASK=segment
YOLO_WEIGHTS=/app/runs/segment/train/weights/best.pt  # ajusta a la ruta real impresa por el script
DEVICE=auto
MODEL_INPUT_SIZE=640
YOLO_CONF=0.25
SEGMENT_DRAW_BBOX=true
```

Restart the service:
```
docker compose -f docker/docker-compose.yml restart fastapi
```

## Training a car-parts detection model (Ultralytics)
If you want bounding boxes for `llanta`, `puerta`, `ventana` (no masks):

1) Prepare dataset (YOLO detection format):
```
backend/datasets/carparts-det/
  images/train/*.jpg
  images/val/*.jpg
  labels/train/*.txt   # lines: <class_id> <cx> <cy> <w> <h> (normalized)
  labels/val/*.txt
  data.yaml
```
An example `data.yaml` is provided at `backend/datasets/carparts-det/data.yaml`.

2) Start Docker (if not already running):
```
docker compose -f docker/docker-compose.yml up --build -d
```

3) Train inside the container:
```
docker compose -f docker/docker-compose.yml exec fastapi \
  python backend/scripts/train_carparts_det.py \
  --data /app/backend/datasets/carparts-det/data.yaml \
  --model yolov8n.pt \
  --epochs 100 \
  --imgsz 640 \
  --device auto
```

4) Use the trained weights:
Update `.env` with:
```
MODEL_BACKEND=ultralytics
MODEL_TASK=detect
YOLO_WEIGHTS=/app/runs/detect/train/weights/best.pt  # ajusta a la ruta real impresa por el script
DEVICE=auto
MODEL_INPUT_SIZE=640
YOLO_CONF=0.25
```

Restart the service:
```
docker compose -f docker/docker-compose.yml restart fastapi
```

Open the MJPEG stream in your browser and you should see boxes for the requested classes.

## Troubleshooting

- "Error reading frame" when using a file:
  - The stream now supports `loop=true` to restart when the file reaches the end.
  - If errors persist, verify the file is accessible inside the container and readable via FFMPEG.

- No bounding boxes visible:
  - Check that the model backend and weights match (`MODEL_BACKEND=ultralytics` + `yolov8n.pt` OR `MODEL_BACKEND=onnx` + `yolov8n.onnx`).
  - Watch container logs for model loading messages.

- Windows host webcam (index `0`) not accessible:
  - Linux containers on Windows cannot access host webcams directly. Use a video file (`samples/*.mp4`) or an IP/RTSP camera.

- GPU not used:
  - Ensure Docker has GPU support and the container sees CUDA devices. Otherwise set `DEVICE=cpu`.

## Development Notes

- The code is mounted into the container (`volumes: ../:/app`), so changes are reflected immediately for most assets.
- Uvicorn is started without `--reload`; restart the service after code changes:
  - `docker compose -f docker/docker-compose.yml restart fastapi`

## License

Copyright © Your Organization. All rights reserved.