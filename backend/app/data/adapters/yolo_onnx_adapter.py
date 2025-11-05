from typing import List, Tuple, Optional

import numpy as np
import cv2

try:
    import onnxruntime as ort  # type: ignore
    _ort_available = True
except Exception:
    ort = None  # type: ignore
    _ort_available = False

from app.domain.entities.bbox_entity import BoundingBox
from app.core.utils.logger import get_logger

_logger = get_logger("yolo_onnx")


def letterbox(image: np.ndarray, new_shape: int = 640, color: Tuple[int, int, int] = (114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize and pad image to meet stride-multiple constraints.

    Returns resized image, gain, and padding (pad_w, pad_h) applied.
    """
    shape = image.shape[:2]  # (h, w)
    if isinstance(new_shape, (list, tuple)):
        new_h, new_w = new_shape
    else:
        new_h, new_w = new_shape, new_shape

    # Scale ratio (new / old) and compute padding
    r = min(new_w / shape[1], new_h / shape[0])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_w - new_unpad[0], new_h - new_unpad[1]
    dw /= 2
    dh /= 2

    # Resize
    img = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)


class YoloOnnxAdapter:
    """Adapter de detección usando ONNX Runtime.

    Asume un modelo estilo YOLOv8 exportado a ONNX con salida [batch, num, classes+5] o similar.
    """

    def __init__(self, onnx_path: str, device: str = "cpu", input_size: int = 640, conf_thres: float = 0.25, iou_thres: float = 0.45):
        self._enabled = _ort_available
        self._session = None
        self._input_name = None
        self._providers = ["CPUExecutionProvider"]
        if device.lower() in ("cuda", "gpu"):
            # Will only work if onnxruntime-gpu is installed inside container
            self._providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._img_size = input_size
        self._conf_thres = conf_thres
        self._iou_thres = iou_thres
        # COCO classes (80). If your model uses a different set, adjust as needed.
        self._names: Optional[dict] = {i: name for i, name in enumerate([
            "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
            "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
            "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
            "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
            "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
            "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
            "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
            "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
            "hair drier","toothbrush"
        ])}

        if not self._enabled:
            _logger.error("ONNX Runtime no disponible. ¿Está instalado onnxruntime en el contenedor?")
        else:
            try:
                self._session = ort.InferenceSession(onnx_path, providers=self._providers)
                self._input_name = self._session.get_inputs()[0].name
                _logger.info("Modelo ONNX cargado: %s (providers=%s)", onnx_path, self._providers)
            except Exception as e:
                self._enabled = False
                _logger.error("No se pudo cargar el modelo ONNX en '%s': %s", onnx_path, e)

    def _nms(self, boxes: List[Tuple[int, int, int, int]], scores: List[float], iou_thres: float) -> List[int]:
        idxs = cv2.dnn.NMSBoxes(bboxes=boxes, scores=scores, score_threshold=self._conf_thres, nms_threshold=iou_thres)
        if idxs is None or len(idxs) == 0:
            return []
        # idxs may be a list of lists [[i], [j], ...]
        return [int(i[0]) if not isinstance(i, int) else int(i) for i in idxs]

    def detect(self, frame: np.ndarray) -> List[BoundingBox]:
        if not self._enabled or self._session is None:
            # Si el adaptador no está habilitado, devolver vacío para no romper el flujo.
            return []

        img, gain, pad = letterbox(frame, self._img_size)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        img_tensor = np.transpose(img_norm, (2, 0, 1))  # HWC->CHW
        img_tensor = np.expand_dims(img_tensor, 0)  # add batch

        outputs = self._session.run(None, {self._input_name: img_tensor})
        if outputs is None or len(outputs) == 0:
            return []

        out = outputs[0]
        # Normalize output shape to (num, attributes)
        if out.ndim == 3:
            # (batch, num, attrs) or (batch, attrs, num)
            if out.shape[1] < out.shape[2]:
                out = out[0]  # (num, attrs)
            else:
                out = out[0].transpose(1, 0)  # (num, attrs)
        elif out.ndim == 2:
            # (num, attrs)
            pass
        else:
            return []

        boxes_xywh = out[:, :4]
        objectness = out[:, 4]
        class_scores = out[:, 5:]

        # Combine objectness with class scores to get per-class confidences, then pick best class
        if class_scores.size == 0:
            # No class scores; treat all as single-class objects
            scores = objectness.tolist()
            classes = [None] * len(scores)
        else:
            class_scores = class_scores * objectness[:, None]
            classes = np.argmax(class_scores, axis=1)
            scores = np.max(class_scores, axis=1)

        sel = scores >= self._conf_thres
        boxes_xywh = boxes_xywh[sel]
        scores = np.array(scores)[sel]
        classes = np.array(classes)[sel] if len(classes) and classes[0] is not None else classes

        # Convert xywh (center x,y,width,height) to xyxy in original image coordinates
        boxes_xyxy = []
        for b in boxes_xywh:
            cx, cy, w, h = b
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            # scale back using gain and pad
            x1 = (x1 - pad[0]) / gain
            y1 = (y1 - pad[1]) / gain
            x2 = (x2 - pad[0]) / gain
            y2 = (y2 - pad[1]) / gain
            boxes_xyxy.append((int(max(0, x1)), int(max(0, y1)), int(min(frame.shape[1] - 1, x2)), int(min(frame.shape[0] - 1, y2))))

        # NMS
        keep = self._nms(boxes_xyxy, scores.tolist(), self._iou_thres)
        boxes: List[BoundingBox] = []
        for i in keep:
            cls_id = int(classes[i]) if isinstance(classes, np.ndarray) and classes.size else None
            cls_name = None
            if cls_id is not None and isinstance(self._names, dict):
                cls_name = self._names.get(cls_id, str(cls_id))
            boxes.append(BoundingBox(x1=boxes_xyxy[i][0], y1=boxes_xyxy[i][1], x2=boxes_xyxy[i][2], y2=boxes_xyxy[i][3], cls=cls_name, conf=float(scores[i])))
        return boxes