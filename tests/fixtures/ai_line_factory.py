import numpy as np

# Track plane in AC is X-Z (Y is vertical). Synthetic fixtures put data in
# the X-Z plane so they match the geometry AILine expects.


def straight_line(length_m=1000.0, n_points=1001):
    """Straight line along +X axis, lying flat (y = 0, z = 0)."""
    s = np.linspace(0.0, length_m, n_points)
    xs = s
    ys = np.zeros_like(s)
    zs = np.zeros_like(s)
    return np.stack([xs, ys, zs], axis=1)


def circle(radius_m=100.0, n_points=628):
    """Circle in the X-Z plane (y = 0)."""
    theta = np.linspace(0.0, 2 * np.pi, n_points, endpoint=False)
    xs = radius_m * np.cos(theta)
    ys = np.zeros_like(theta)
    zs = radius_m * np.sin(theta)
    return np.stack([xs, ys, zs], axis=1)
