import pytest
from ac_rl.config import Config
from ac_rl.reward import compute_reward, RewardInputs

def make_inputs(**kwargs):
    defaults = dict(
        progress_delta_m=0.0,
        lateral_error_m=0.0,
        heading_error_rad=0.0,
        delta_steer=0.0,
        delta_throttle=0.0,
        offtrack_terminal=False,
        speed_kmh=0.0,
        lap_completed_now=False,
    )
    defaults.update(kwargs)
    return RewardInputs(**defaults)

def test_progress_positive():
    cfg = Config()
    # speed above the anti-stop threshold so only progress (+ speed_bonus) is in play
    r = compute_reward(make_inputs(progress_delta_m=1.0, speed_kmh=30.0), cfg)
    assert r > 0

def test_lateral_penalty_quadratic():
    cfg = Config()
    # isolate the lateral term by differencing at a fixed speed (other terms cancel)
    base = compute_reward(make_inputs(lateral_error_m=0.0, speed_kmh=30.0), cfg)
    p1 = base - compute_reward(make_inputs(lateral_error_m=1.0, speed_kmh=30.0), cfg)
    p2 = base - compute_reward(make_inputs(lateral_error_m=2.0, speed_kmh=30.0), cfg)
    assert p1 == pytest.approx(cfg.reward.w_lateral * 1.0)
    assert p2 == pytest.approx(cfg.reward.w_lateral * 4.0)   # quadratic: 4x p1

def test_offtrack_one_shot_penalty():
    cfg = Config()
    r = compute_reward(make_inputs(offtrack_terminal=True), cfg)
    assert r <= -cfg.reward.w_offtrack

def test_speed_bonus_present():
    cfg = Config()
    r0 = compute_reward(make_inputs(speed_kmh=0.0), cfg)
    r1 = compute_reward(make_inputs(speed_kmh=100.0), cfg)
    assert r1 > r0


def test_lap_bonus_only_on_crossing_step():
    cfg = Config()
    r0 = compute_reward(make_inputs(), cfg)
    r1 = compute_reward(make_inputs(lap_completed_now=True), cfg)
    assert r1 == pytest.approx(r0 + cfg.reward.w_lap_bonus)


def test_steer_straight_penalised_on_straight():
    """On a straight (curvature 0) |delta_steer| is penalised by w_steer_straight."""
    cfg = Config()
    base = compute_reward(make_inputs(speed_kmh=100.0, delta_steer=0.0, curvature=0.0), cfg)
    r = compute_reward(make_inputs(speed_kmh=100.0, delta_steer=0.3, curvature=0.0), cfg)
    # the only delta_steer terms are w_jerk (ungated) and w_steer_straight (gate=1 here)
    expected = (cfg.reward.w_jerk + cfg.reward.w_steer_straight) * 0.3
    assert (base - r) == pytest.approx(expected)


def test_steer_straight_free_in_corner():
    """In a corner (curvature >= clamp_curv_full) the straight-only penalty is gated OFF;
    only the tiny ungated w_jerk remains -> fast steering is free for the chicane."""
    cfg = Config()
    c = cfg.clamp_curv_full
    base = compute_reward(make_inputs(speed_kmh=100.0, delta_steer=0.0, curvature=c), cfg)
    r = compute_reward(make_inputs(speed_kmh=100.0, delta_steer=0.3, curvature=c), cfg)
    assert (base - r) == pytest.approx(cfg.reward.w_jerk * 0.3)  # w_steer_straight gated to 0


def test_antistop_zero_above_threshold():
    """No anti-stop penalty at/above the threshold (legit slow corners untouched)."""
    cfg = Config()
    thr = cfg.low_speed_threshold_kmh
    r_thr = compute_reward(make_inputs(speed_kmh=thr), cfg)
    r_fast = compute_reward(make_inputs(speed_kmh=thr + 30), cfg)
    # the only speed-dependent terms are speed_bonus (linear) and anti-stop; above threshold
    # anti-stop is 0, so the difference is exactly the speed_bonus over the 30 km/h gap.
    assert (r_fast - r_thr) == pytest.approx(cfg.reward.w_speed_bonus * 30)


def test_antistop_standing_worse_than_crawling():
    """Stopped scores strictly less than crawling, which scores less than at threshold."""
    cfg = Config()
    r_stop = compute_reward(make_inputs(speed_kmh=0.0), cfg)
    r_crawl = compute_reward(make_inputs(speed_kmh=10.0), cfg)
    r_thr = compute_reward(make_inputs(speed_kmh=cfg.low_speed_threshold_kmh), cfg)
    assert r_stop < r_crawl < r_thr


def test_antistop_linear_below_threshold():
    """Below threshold the anti-stop penalty is linear: v=0 costs w_low*threshold."""
    cfg = Config()
    thr = cfg.low_speed_threshold_kmh
    speed_only = cfg.reward.w_speed_bonus * 0.0
    full = compute_reward(make_inputs(speed_kmh=0.0), cfg)
    assert (speed_only - full) == pytest.approx(cfg.reward.w_low * thr)
