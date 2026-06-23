from ac_rl.config import Config
from ac_rl.termination import TerminationTracker

def test_offtrack_terminates_after_duration():
    cfg = Config()
    t = TerminationTracker(cfg)
    reason = None
    for _ in range(int(cfg.offtrack_duration_s * cfg.control_hz) + 1):
        reason = t.update(offtrack=True, lateral_norm=0.5, speed_kmh=100.0, spline_pos=0.5)
    assert reason == "offtrack"

def test_stopped_terminates_after_duration():
    cfg = Config()
    t = TerminationTracker(cfg)
    reason = None
    for _ in range(int(cfg.stopped_duration_s * cfg.control_hz) + 1):
        reason = t.update(offtrack=False, lateral_norm=0.0, speed_kmh=0.0, spline_pos=0.1)
    assert reason == "stopped"

def test_no_progress_terminates():
    cfg = Config()
    t = TerminationTracker(cfg)
    reason = None
    for _ in range(int(cfg.no_progress_duration_s * cfg.control_hz) + 5):
        reason = t.update(offtrack=False, lateral_norm=0.0, speed_kmh=50.0, spline_pos=0.5)
    assert reason == "no_progress"

def test_lap_completion_does_not_terminate():
    """Crossing the S/F line counts a lap but the episode continues."""
    cfg = Config()
    t = TerminationTracker(cfg)
    t.update(offtrack=False, lateral_norm=0.0, speed_kmh=100.0, spline_pos=0.99)
    reason = t.update(offtrack=False, lateral_norm=0.0, speed_kmh=100.0, spline_pos=0.01)
    assert reason is None
    assert t.lap_completed_now is True
    assert t.laps_completed == 1
    # Next step does NOT re-flag the same crossing.
    reason = t.update(offtrack=False, lateral_norm=0.0, speed_kmh=100.0, spline_pos=0.02)
    assert t.lap_completed_now is False
    assert t.laps_completed == 1


def test_multiple_laps_counted():
    cfg = Config()
    t = TerminationTracker(cfg)
    # Walk spline_pos around the lap twice.
    for sp in [0.5, 0.95, 0.05, 0.5, 0.95, 0.05]:
        t.update(offtrack=False, lateral_norm=0.0, speed_kmh=100.0, spline_pos=sp)
    assert t.laps_completed == 2

def test_lateral_clip_terminates():
    cfg = Config()
    t = TerminationTracker(cfg)
    reason = t.update(offtrack=False, lateral_norm=2.5, speed_kmh=100.0, spline_pos=0.5)
    assert reason == "lateral_excess"
