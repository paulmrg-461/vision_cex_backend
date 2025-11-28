import os
import time
from typing import Optional, Callable

import cv2


class SnapshotScheduler:
    """
    A simple time-based scheduler to save a fixed number of snapshots
    at a given interval to a target directory.
    """

    def __init__(self, save_dir: str, interval_seconds: float = 0.3, max_count: int = 10, prefix: str = "bus", on_snapshot: Optional[Callable[[str], None]] = None):
        self.save_dir = save_dir
        self.interval = max(0.01, float(interval_seconds))
        self.max_count = max(1, int(max_count))
        self.prefix = prefix
        self.on_snapshot = on_snapshot

        self._active = False
        self._next_ts: float = 0.0
        self._captured = 0

    def trigger(self, now: Optional[float] = None, interval_seconds: Optional[float] = None, max_count: Optional[int] = None, prefix: Optional[str] = None):
        if interval_seconds is not None:
            self.interval = max(0.01, float(interval_seconds))
        if max_count is not None:
            self.max_count = max(1, int(max_count))
        if prefix is not None:
            self.prefix = prefix

        ts = time.time() if now is None else float(now)
        self._active = True
        self._next_ts = ts  # allow immediate capture on next process call
        self._captured = 0

    @property
    def active(self) -> bool:
        return self._active

    def process(self, frame, now: Optional[float] = None) -> bool:
        """
        If active and due, saves a snapshot of the given frame.
        Returns True if a snapshot was saved.
        """
        if not self._active or frame is None:
            return False
        ts = time.time() if now is None else float(now)
        if ts < self._next_ts:
            return False

        try:
            os.makedirs(self.save_dir, exist_ok=True)
            # Compose filename with millisecond resolution
            millis = int(ts * 1000)
            name = f"{self.prefix}_{millis}_{self._captured + 1:02d}.jpg"
            path = os.path.join(self.save_dir, name)
            cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1].tofile(path)
            self._captured += 1
            # Callback de notificación
            try:
                if self.on_snapshot:
                    self.on_snapshot(path)
            except Exception:
                # no bloquear por fallos en callback
                pass
        except Exception:
            # Non-fatal: keep schedule running
            pass

        # Schedule next capture or deactivate
        if self._captured >= self.max_count:
            self._active = False
        else:
            self._next_ts = ts + self.interval

        return True
