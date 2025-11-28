import numpy as np

from app.core.utils.motion_gate import MotionGate


def test_motion_gate_detects_change_full_frame():
    gate = MotionGate(history=10)
    # Create two frames: one black, one with a white rectangle
    frame1 = np.zeros((200, 200, 3), dtype=np.uint8)
    frame2 = frame1.copy()
    frame2[50:150, 50:150] = 255

    # Warm-up background with first frame
    _ = gate.compute_motion_ratio(frame1)
    # Apply second frame
    ratio = gate.compute_motion_ratio(frame2)
    assert 0.1 <= ratio <= 0.5  # changed area ~ (100x100)/(200x200) = 0.25


def test_motion_gate_roi_limits_and_trigger():
    gate = MotionGate(history=10)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # Change only inside ROI
    frame[10:40, 10:40] = 255
    roi = (5, 5, 50, 50)

    # First compute should set background and detect change
    ratio = gate.compute_motion_ratio(frame, roi)
    assert ratio > 0.1
    assert gate.should_trigger(frame, roi, threshold=0.35) is (ratio >= 0.35)


def test_motion_gate_handles_invalid_input():
    gate = MotionGate(history=10)
    # None frame or empty frame should return 0 ratio and not trigger
    assert gate.compute_motion_ratio(None) == 0.0  # type: ignore
    empty = np.array([], dtype=np.uint8)
    assert gate.compute_motion_ratio(empty) == 0.0
    assert gate.should_trigger(empty, threshold=0.35) is False

