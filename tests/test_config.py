from ac_rl.config import Config

def test_default_config_loads():
    cfg = Config()
    assert cfg.control_hz == 20
    assert cfg.obs_dim == 20
    assert cfg.action_dim == 2
    assert cfg.reward.w_progress == 1.0
    assert cfg.reward.w_lateral == 2.0
    assert cfg.reward.w_low == 0.2
    assert cfg.reward.w_steer_straight == 0.5
    assert cfg.clamp_straight == 2.5
    assert cfg.clamp_corner == 4.0
    assert cfg.clamp_corner > cfg.clamp_straight   # corners looser than straights
    assert cfg.clamp_curv_full == 0.015
    assert cfg.clamp_behind_m == 15.0
    assert cfg.low_speed_threshold_kmh == 20.0
    assert len(cfg.lookahead_distances_m) == 6

def test_lookahead_distances_monotonic():
    cfg = Config()
    d = cfg.lookahead_distances_m
    assert all(d[i] < d[i+1] for i in range(len(d) - 1))
