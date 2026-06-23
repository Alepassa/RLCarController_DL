from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class RewardWeights:
    w_progress: float = 1.0
    w_lateral: float = 2.0         # baseline glue-to-line (quadratic). Kept AS-IS: the corridor
                                   # relaxation (deosc_v1b) failed -- it neither cut the saw (46%
                                   # vs 50%) nor kept laps (5 vs 466; car learned to stop). The saw
                                   # is now attacked structurally by the steer clamp, not via reward.
    w_heading: float = 0.5
    w_jerk: float = 0.05
    # Reduced from 10.0 after sanity_v1: the one-shot offtrack penalty was
    # dominating the critic loss target and "stopped" had become the dominant
    # termination because the agent learned that staying still avoided it.
    w_offtrack: float = 5.0
    # Raised from 0.01 after sanity_v1: the bonus was too weak to overcome
    # the "stay still" attractor; the agent reached only ~95 km/h top speed
    # at the end of 30k steps.
    w_speed_bonus: float = 0.05
    # One-shot reward at the step the car crosses the start/finish line.
    # Episode does NOT terminate on lap completion (the agent keeps going).
    w_lap_bonus: float = 50.0
    # Anti-stop (clamp_v2): penalty = w_low * max(0, low_speed_threshold - speed_kmh).
    # 0 above the threshold (legit slow corners untouched), grows as the car creeps below
    # it -> standing still is strictly worse than crawling. The baseline reward had no
    # anti-stop; with the slower clamped wheel the policy fell into the "stop" attractor
    # (stopped 72->100% in clamp_v1). This unblocks the stall without moving the objective.
    # Straight-only steering smoothness (steer_straight): penalty = w_steer_straight *
    # |delta_steer| * gate, gate = max(0, 1 - curvature/clamp_curv_full). Active on straights
    # (kills the saw at its source -> no ondeggio), 0 in corners (chicane stays free). Gated, so
    # unlike the ungated steersmooth L1 it does not fight the corners. Paired with the corner-aware
    # clamp (which protects the chicane during training). Tuning knob: ondeggio persists -> raise;
    # straights go flat-steer/understeer -> lower.
    w_steer_straight: float = 0.5
    w_low: float = 0.2

@dataclass
class Config:
    control_hz: int = 20
    obs_dim: int = 20
    action_dim: int = 2

    lookahead_distances_m: tuple = (5.0, 10.0, 20.0, 40.0, 80.0, 160.0)

    track_half_width_m: float = 6.0
    speed_norm_kmh: float = 300.0

    # de-oscillation via the actuator (NOT the reward): rate-limit the steering so the wheel
    # cannot move more than steer_rate_limit/control_hz per step. Models a real steering actuator's
    # finite slew rate and low-pass-filters the policy's left-right saw into its smooth average.
    # CORNER-AWARE steering clamp. A uniform clamp can't win on Monza: the rate that kills the
    # straight-line saw/ondeggio (~2.5) is too slow for the 90 deg first chicane, and the rate
    # that takes the chicane (~3.5+) lets the ondeggio back. So scale the rate-limit by the
    # ai_line curvature ahead: TIGHT on straights (kill the ondeggio), LOOSE in corners (fast
    # turn-in). rate = clamp_straight + (clamp_corner - clamp_straight) * min(1, curv/clamp_curv_full).
    # Applied to the baseline at execution -> no training needed, can't break the chicane.
    clamp_straight: float = 2.5     # rate where the line is straight (curvature ~0) -> smooth
    clamp_corner: float = 4.0       # rate where the line bends hard -> chicane-capable
    clamp_curv_full: float = 0.015  # curvature (1/m) at which the clamp is fully loosened
    clamp_ahead_m: float = 30.0     # look-ahead so the clamp loosens BEFORE the corner
    clamp_behind_m: float = 15.0    # keep clamp loose 15 m after corner exit (reactive on exit)

    # anti-stop: speed (km/h) below which the w_low penalty starts. Low (20) so legit slow
    # corners (~40-50 km/h) are NOT penalised -- only genuine stalling/creeping.
    low_speed_threshold_kmh: float = 20.0

    offtrack_duration_s: float = 1.0
    stopped_speed_kmh: float = 1.0
    stopped_duration_s: float = 3.0
    no_progress_duration_s: float = 10.0
    max_episode_seconds: float = 300.0
    lateral_error_terminate_norm: float = 2.0

    ipc_host: str = "127.0.0.1"
    ipc_port: int = 50555

    runs_dir: Path = field(default_factory=lambda: Path("runs"))
    step_sample_every: int = 100

    reward: RewardWeights = field(default_factory=RewardWeights)
