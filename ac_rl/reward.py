from dataclasses import dataclass
from ac_rl.config import Config


@dataclass
class RewardInputs:
    progress_delta_m: float
    lateral_error_m: float
    heading_error_rad: float
    delta_steer: float
    delta_throttle: float
    offtrack_terminal: bool
    speed_kmh: float
    curvature: float = 0.0  # ai_line curvature ahead, gates the straight-only steer penalty
    lap_completed_now: bool = False  # True only on the step where S/F line is crossed


def compute_reward(inp: RewardInputs, cfg: Config) -> float:
    w = cfg.reward
    r = 0.0
    r += w.w_progress * inp.progress_delta_m
    r -= w.w_lateral * (inp.lateral_error_m ** 2)
    r -= w.w_heading * (inp.heading_error_rad ** 2)
    r -= w.w_low * max(0.0, cfg.low_speed_threshold_kmh - inp.speed_kmh)  # anti-stop
    r -= w.w_jerk * (abs(inp.delta_steer) + abs(inp.delta_throttle))
    # Straight-only steering-smoothness: penalise |delta_steer| where the line is straight
    # (gate=1) and fade to 0 as a corner approaches (curv -> clamp_curv_full). Kills the saw at
    # its SOURCE (the policy) on straights, while leaving the chicane's fast steering free. The
    # spatial gate is what the ungated steersmooth L1 lacked (it hurt cornering).
    straight_gate = max(0.0, 1.0 - inp.curvature / cfg.clamp_curv_full) if cfg.clamp_curv_full > 0 else 1.0
    r -= w.w_steer_straight * abs(inp.delta_steer) * straight_gate
    r += w.w_speed_bonus * inp.speed_kmh
    if inp.offtrack_terminal:
        r -= w.w_offtrack
    if inp.lap_completed_now:
        r += w.w_lap_bonus
    return float(r)
