"""
Train an Ultralytics YOLO segmentation model for car parts.

Usage (inside Docker container):
  python backend/scripts/train_carparts_seg.py \
    --data /app/backend/datasets/carparts-seg/data.yaml \
    --model yolov8n-seg.pt \
    --epochs 100 \
    --imgsz 640 \
    --device auto

If --device is 'auto', this script will choose GPU '0' if available, otherwise 'cpu'.
"""

import argparse
import os

try:
    from ultralytics import YOLO  # type: ignore
    _ultralytics_available = True
except Exception:
    YOLO = None  # type: ignore
    _ultralytics_available = False

try:
    import torch  # type: ignore
    _torch_available = True
except Exception:
    torch = None  # type: ignore
    _torch_available = False


def auto_device(dev_opt: str) -> str:
    if dev_opt.lower() == "auto":
        if _torch_available and torch.cuda.is_available():
            return "0"  # first GPU
        return "cpu"
    return dev_opt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/app/backend/datasets/carparts-seg/data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n-seg.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device_final = auto_device(args.device)

    if not _ultralytics_available:
        print("ERROR: Ultralytics no está disponible en el contenedor.")
        return 1

    print(f"Entrenando segmentación: data={args.data}, model={args.model}, epochs={args.epochs}, imgsz={args.imgsz}, device={device_final}")
    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device_final,
        verbose=True,
    )
    # Ultralytics returns a dict-like results; best weights path is under 'save_dir/weights/best.pt'
    save_dir = getattr(results, 'save_dir', None)
    if save_dir is None:
        print("Entrenamiento finalizado, no se pudo determinar save_dir.")
    else:
        best_path = os.path.join(str(save_dir), "weights", "best.pt")
        print(f"Entrenamiento finalizado. Pesos óptimos: {best_path}")
        print("Para usar en el backend, establece en .env:")
        print(f"  MODEL_BACKEND=ultralytics\n  MODEL_TASK=segment\n  YOLO_WEIGHTS={best_path}\n  DEVICE=auto\n  MODEL_INPUT_SIZE=640")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())