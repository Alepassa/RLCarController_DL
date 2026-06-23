import numpy as np
from ac_rl.config import Config
from ac_rl.ai_line import AILine
from ac_rl.observation import CarState, build_observation
from tests.fixtures.ai_line_factory import straight_line

def test_observation_shape_and_normalization():
    cfg = Config()
    line = AILine(straight_line())
    state = CarState(
        position=np.array([500.0, 0.0, 0.0]),
        velocity=np.array([50.0, 0.0, 0.0]),
        heading_rad=0.0,
        yaw_rate=0.0,
        num_tyres_out=0,
    )
    obs = build_observation(state, line, prev_steer=0.0, prev_throttle=0.0, cfg=cfg)
    assert obs.shape == (cfg.obs_dim,)
    assert np.all(np.isfinite(obs))

def test_observation_lateral_error_normalized():
    cfg = Config()
    line = AILine(straight_line())
    # Track plane is X-Z; lateral offset is along Z, not Y.
    state = CarState(
        position=np.array([500.0, 0.0, cfg.track_half_width_m]),
        velocity=np.array([0.0, 0.0, 0.0]),
        heading_rad=0.0,
        yaw_rate=0.0,
        num_tyres_out=0,
    )
    obs = build_observation(state, line, 0.0, 0.0, cfg)
    assert abs(obs[0] - 1.0) < 0.05

def test_offtrack_flag():
    cfg = Config()
    line = AILine(straight_line())
    state = CarState(
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        heading_rad=0.0,
        yaw_rate=0.0,
        num_tyres_out=3,
    )
    obs = build_observation(state, line, 0.0, 0.0, cfg)
    assert obs[-1] == 1.0
