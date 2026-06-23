"""Structural steering rate-limit (actuator model) — pure, hardware-free, unit-testable.

Used by env.step() between the policy's raw steering output and vJoy. Limiting the
per-step change models the finite slew rate of a real steering actuator and low-pass-
filters the policy's high-frequency left-right saw into its smooth average, without
touching the reward/objective.
"""


def apply_steer_rate_limit(prev_cmd: float, target: float, max_delta: float) -> float:
    """Clamp `target` to within `max_delta` of `prev_cmd`, then to [-1, 1].

    prev_cmd : last applied steering command
    target   : raw steering the policy wants (already clipped to [-1, 1])
    max_delta : max change allowed this step (= steer_rate_limit * dt)
    """
    lo = prev_cmd - max_delta
    hi = prev_cmd + max_delta
    cmd = min(max(target, lo), hi)
    return float(min(1.0, max(-1.0, cmd)))
