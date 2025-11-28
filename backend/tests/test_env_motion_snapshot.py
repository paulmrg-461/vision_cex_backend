import os

from app.core.config.environment_config import EnvironmentConfig


def test_env_defaults_for_motion_and_snapshots(monkeypatch):
    # Ensure no custom values in environment
    for key in [
        "MOTION_CHANGE_THRESHOLD",
        "SNAPSHOT_INTERVAL_SECONDS",
        "SNAPSHOT_MAX_COUNT",
        "SNAPSHOT_SAVE_DIR",
        "SNAPSHOT_DETECT_MIN_CONF",
    ]:
        monkeypatch.delenv(key, raising=False)

    cfg = EnvironmentConfig()
    assert abs(cfg.motion_change_threshold - 0.35) < 1e-6
    assert abs(cfg.snapshot_interval_seconds - 0.3) < 1e-6
    assert cfg.snapshot_max_count == 10
    assert cfg.snapshot_save_dir == "/app/samples/snapshots"
    assert abs(cfg.snapshot_detect_min_conf - 0.5) < 1e-6


def test_env_overrides_for_motion_and_snapshots(monkeypatch):
    monkeypatch.setenv("MOTION_CHANGE_THRESHOLD", "0.5")
    monkeypatch.setenv("SNAPSHOT_INTERVAL_SECONDS", "0.1")
    monkeypatch.setenv("SNAPSHOT_MAX_COUNT", "7")
    monkeypatch.setenv("SNAPSHOT_SAVE_DIR", "/tmp/snaps")
    monkeypatch.setenv("SNAPSHOT_DETECT_MIN_CONF", "0.65")

    cfg = EnvironmentConfig()
    assert abs(cfg.motion_change_threshold - 0.5) < 1e-6
    assert abs(cfg.snapshot_interval_seconds - 0.1) < 1e-6
    assert cfg.snapshot_max_count == 7
    assert cfg.snapshot_save_dir == "/tmp/snaps"
    assert abs(cfg.snapshot_detect_min_conf - 0.65) < 1e-6
