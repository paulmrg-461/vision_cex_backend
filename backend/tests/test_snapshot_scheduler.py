import os
import numpy as np
import cv2

from app.core.utils.snapshot_scheduler import SnapshotScheduler


def test_snapshot_scheduler_saves_expected_count(tmp_path):
    save_dir = tmp_path / "snapshots"
    scheduler = SnapshotScheduler(save_dir=str(save_dir), interval_seconds=0.05, max_count=5, prefix="bus")

    # Simple frame
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    frame[10:40, 10:40] = 255

    # Trigger and advance time artificially
    scheduler.trigger(now=0.0)

    ts = 0.0
    saved = 0
    for _ in range(100):
        if scheduler.process(frame, now=ts):
            saved += 1
        ts += 0.01
        if not scheduler.active:
            break

    assert saved == 5
    files = list(os.listdir(save_dir))
    assert len(files) == 5
    assert all(name.endswith('.jpg') for name in files)


def test_snapshot_scheduler_inactive_no_save(tmp_path):
    save_dir = tmp_path / "snapshots2"
    scheduler = SnapshotScheduler(save_dir=str(save_dir), interval_seconds=0.05, max_count=3)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    # Not triggered yet
    for _ in range(10):
        assert scheduler.process(frame, now=0.0) is False
    assert os.path.isdir(save_dir) is False or len(os.listdir(save_dir)) == 0

