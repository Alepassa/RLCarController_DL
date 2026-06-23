from dataclasses import dataclass
import numpy as np
from ac_rl.config import Config
from ac_rl.ai_line import AILine

# AC track plane is X-Z; Y is vertical.
_HORIZ = [0, 2]


@dataclass
class CarState:
    position: np.ndarray       # (3,) world XYZ
    velocity: np.ndarray       # (3,) world XYZ
    heading_rad: float         # AC yaw: atan2(forward_z, forward_x) (same convention as AILine.tangent_heading)
    yaw_rate: float            # rad/s
    num_tyres_out: int         # from shared memory


def _wrap_angle(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


def build_observation(
    state: CarState,
    line: AILine,
    prev_steer: float,
    prev_throttle: float,
    cfg: Config,
) -> np.ndarray:
    pos = state.position
    lateral = line.signed_lateral_error(pos)
    tangent_heading = line.tangent_heading(pos)
    heading_err = _wrap_angle(state.heading_rad - tangent_heading)

    idx, _ = line.nearest(pos)
    t = line.tangent_xz(idx)
    normal = np.array([-t[1], t[0]])
    vel_xz = state.velocity[_HORIZ]
    lat_vel = float(np.dot(vel_xz, normal))

    speed = float(np.linalg.norm(state.velocity)) * 3.6  # m/s -> km/h

    lookahead = line.lookahead_points(pos, cfg.lookahead_distances_m)
    # Rotate lookahead points into the car's local (forward, left) frame in X-Z.
    # AC heading convention: h = atan2(forward_x, forward_z), so
    # forward_world = (sin h, cos h), left_world = (-cos h, sin h).
    cos_h, sin_h = np.cos(state.heading_rad), np.sin(state.heading_rad)
    rel = lookahead[:, _HORIZ] - pos[_HORIZ]
    local = np.empty_like(rel)
    local[:, 0] = sin_h * rel[:, 0] + cos_h * rel[:, 1]   # forward
    local[:, 1] = -cos_h * rel[:, 0] + sin_h * rel[:, 1]  # left

    norm_dist = float(cfg.lookahead_distances_m[-1])
    obs = np.zeros(cfg.obs_dim, dtype=np.float32)
    obs[0] = lateral / cfg.track_half_width_m
    obs[1] = heading_err / np.pi
    obs[2] = lat_vel / 30.0
    obs[3] = speed / cfg.speed_norm_kmh
    obs[4:16] = (local / norm_dist).flatten()
    obs[16] = prev_steer
    obs[17] = prev_throttle
    obs[18] = state.yaw_rate / 3.0
    obs[19] = 1.0 if state.num_tyres_out >= 3 else 0.0
    return obs
