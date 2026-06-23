import numpy as np
import pytest
from ac_rl.ai_line import AILine
from tests.fixtures.ai_line_factory import straight_line, circle


def test_curvature_ahead_straight_is_near_zero():
    line = AILine(straight_line())
    assert line.curvature_ahead(np.array([500.0, 0.0, 0.0]), ahead_m=50) < 1e-3


def test_curvature_ahead_circle_matches_inverse_radius():
    """A circle of radius R has curvature ~ 1/R everywhere."""
    R = 100.0
    line = AILine(circle(radius_m=R))
    k = line.curvature_ahead(np.array([R, 0.0, 0.0]), ahead_m=20)
    assert k == pytest.approx(1.0 / R, rel=0.2)


def test_nearest_point_on_straight_line():
    # Straight line along +X, lateral offset is along Z.
    pts = straight_line()
    line = AILine(pts)
    idx, dist = line.nearest(np.array([500.0, 0.0, 3.0]))
    assert idx == 500
    assert dist == pytest.approx(3.0, abs=0.01)


def test_signed_lateral_error_left_is_positive():
    pts = straight_line()
    line = AILine(pts)
    # Tangent is +X. Left-hand normal in (x, z) is (-tz, tx) = (0, 1) → +Z is "left".
    err = line.signed_lateral_error(np.array([500.0, 0.0, 5.0]))
    assert err == pytest.approx(5.0, abs=0.01)
    err = line.signed_lateral_error(np.array([500.0, 0.0, -5.0]))
    assert err == pytest.approx(-5.0, abs=0.01)


def test_progress_along_line():
    pts = straight_line()
    line = AILine(pts)
    p0 = line.progress_m(np.array([100.0, 0.0, 0.0]))
    p1 = line.progress_m(np.array([150.0, 0.0, 0.0]))
    assert p1 - p0 == pytest.approx(50.0, abs=0.1)


def test_tangent_heading():
    # Circle in X-Z plane parametrized by theta. At (R, 0, 0) (theta=0) the
    # tangent (direction of increasing theta) is along +Z. AC convention is
    # heading = atan2(forward_x, forward_z) = atan2(0, 1) = 0.
    pts = circle()
    line = AILine(pts)
    h = line.tangent_heading(np.array([100.0, 0.0, 0.0]))
    assert abs(h) < 0.1
