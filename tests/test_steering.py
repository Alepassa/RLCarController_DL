from ac_rl.steering import apply_steer_rate_limit


def test_clamp_limits_change():
    """A big jump is limited to max_delta from the previous command."""
    assert apply_steer_rate_limit(0.0, 1.0, max_delta=0.1) == 0.1
    assert apply_steer_rate_limit(0.0, -1.0, max_delta=0.1) == -0.1


def test_clamp_passes_small_change():
    """A change within max_delta passes through unchanged."""
    assert apply_steer_rate_limit(0.2, 0.25, max_delta=0.1) == 0.25


def test_clamp_bounds_to_unit_range():
    assert apply_steer_rate_limit(0.95, 1.0, max_delta=0.5) == 1.0
    assert apply_steer_rate_limit(-0.95, -1.0, max_delta=0.5) == -1.0


def test_clamp_slower_limit_smaller_step():
    """A slower rate-limit produces a smaller step toward the same target (less saw)."""
    fast = apply_steer_rate_limit(0.0, 1.0, max_delta=0.2)   # steer_rate_limit 4.0 @ 20Hz
    slow = apply_steer_rate_limit(0.0, 1.0, max_delta=0.1)   # steer_rate_limit 2.0 @ 20Hz
    assert slow < fast
