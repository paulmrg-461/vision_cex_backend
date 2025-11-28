import os
import time
from pathlib import Path

import numpy as np

from app.core.utils.snapshot_scheduler import SnapshotScheduler


def test_snapshot_scheduler_invokes_callback(tmp_path: Path):
    saved = []

    def on_snap(path: str):
        saved.append(path)

    scheduler = SnapshotScheduler(save_dir=str(tmp_path), interval_seconds=0.01, max_count=3, prefix="bus", on_snapshot=on_snap)
    scheduler.trigger()

    # frame dummy
    frame = (np.ones((64, 64, 3)) * 255).astype("uint8")

    # loop until complete
    start = time.time()
    while scheduler.active and time.time() - start < 2.0:
        scheduler.process(frame)
        time.sleep(0.01)

    assert len(saved) == 3
    for p in saved:
        assert os.path.exists(p)

