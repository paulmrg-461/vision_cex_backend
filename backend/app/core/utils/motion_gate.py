import os
from typing import Optional, Tuple

import cv2
import numpy as np


class MotionGate:
    """
    Simple motion detector using OpenCV BackgroundSubtractor (MOG2).

    - Maintains an internal background model across frames.
    - Computes foreground mask and returns ratio of changed pixels.
    - Supports optional ROI in format (x, y, w, h).
    """

    def __init__(self, history: int = 200, var_threshold: float = 16.0, detect_shadows: bool = True):
        self.back_sub = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=var_threshold, detectShadows=detect_shadows)
        # Kernel for noise reduction in mask
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def _extract_roi(self, frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
        if roi is None:
            return frame
        x, y, w, h = roi
        h_frame, w_frame = frame.shape[:2]
        # Clamp ROI to frame bounds for safety
        x = max(0, min(x, w_frame - 1))
        y = max(0, min(y, h_frame - 1))
        w = max(1, min(w, w_frame - x))
        h = max(1, min(h, h_frame - y))
        return frame[y : y + h, x : x + w]

    def compute_motion_ratio(self, frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> float:
        """
        Updates the background model with the given frame and returns the ratio
        of foreground (changed) pixels in [0.0, 1.0].
        """
        if frame is None or frame.size == 0:
            return 0.0

        region = self._extract_roi(frame, roi)
        # Apply background subtraction
        fg_mask = self.back_sub.apply(region)
        if fg_mask is None:
            return 0.0

        # Remove shadows and small noise
        # MOG2 marks shadows as 127; threshold to binary and morph close/open
        _, bin_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, self.kernel, iterations=1)

        # Compute ratio of white pixels
        white = float(cv2.countNonZero(bin_mask))
        total = float(bin_mask.shape[0] * bin_mask.shape[1])
        if total <= 0.0:
            return 0.0
        return white / total

    def should_trigger(self, frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None, threshold: float = 0.35) -> bool:
        ratio = self.compute_motion_ratio(frame, roi)
        return ratio >= threshold


def ensure_directory(path: str) -> None:
    """Create directory tree if it doesn't exist (safe in concurrent contexts)."""
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        # Best-effort; caller can handle failures when writing files
        pass

