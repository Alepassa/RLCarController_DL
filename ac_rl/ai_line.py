import numpy as np

# In Assetto Corsa world coordinates: X and Z are horizontal, Y is vertical.
# All planar geometry (nearest, tangent, lateral error) projects to the X-Z plane.
_HORIZ = [0, 2]


class AILine:
    """Polyline representation of an AC ai_line with geometric queries.

    Points are (N, 3) XYZ in world coords. Index 0 is the start/finish line;
    consecutive indices advance along the racing direction. The planar
    queries (nearest, tangent_heading, signed_lateral_error, ...) operate on
    the X-Z projection — Y is vertical in AC and is ignored for lateral
    geometry.
    """

    def __init__(self, points: np.ndarray):
        assert points.ndim == 2 and points.shape[1] == 3
        self.points = points.astype(np.float64)
        self._xz = self.points[:, _HORIZ]
        self.n = len(points)
        segs = np.diff(self._xz, axis=0)
        self.seg_lengths = np.linalg.norm(segs, axis=1)
        self.cum_length = np.concatenate([[0.0], np.cumsum(self.seg_lengths)])
        self.total_length = float(self.cum_length[-1])
        # Unsigned curvature (1/m) per waypoint via the cross-product form:
        # k_i = 2*|a x b| / (|a|*|b|*|a+b|), a = p_i - p_{i-1}, b = p_{i+1} - p_i.
        # ~0.02 for a 50 m radius corner. Used by the corner-aware steering clamp.
        self.curvature = self._compute_curvature()

    def _compute_curvature(self) -> np.ndarray:
        a = self._xz - np.roll(self._xz, 1, axis=0)
        b = np.roll(self._xz, -1, axis=0) - self._xz
        cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) * np.linalg.norm(a + b, axis=1)
        k = np.zeros(self.n)
        m = denom > 1e-9
        k[m] = 2.0 * np.abs(cross[m]) / denom[m]
        return k

    def curvature_ahead(self, pos: np.ndarray, ahead_m: float = 30.0, behind_m: float = 5.0) -> float:
        """Max unsigned curvature (1/m) in an arc window from `behind_m` behind to
        `ahead_m` ahead of `pos`. Looking ahead lets the clamp loosen BEFORE a corner
        so the wheel is ready for turn-in. ~0 on a true straight."""
        base = self.progress_m(pos)
        cl = self.cum_length
        start, end = base - behind_m, base + ahead_m
        if start >= 0.0 and end <= self.total_length:
            mask = (cl >= start) & (cl <= end)
        else:
            lo, hi = start % self.total_length, end % self.total_length
            mask = (cl >= lo) | (cl <= hi)
        if not mask.any():
            idx, _ = self.nearest(pos)
            return float(self.curvature[idx])
        return float(self.curvature[mask].max())

    def nearest(self, pos: np.ndarray) -> tuple[int, float]:
        q = pos[_HORIZ]
        d = np.linalg.norm(self._xz - q, axis=1)
        idx = int(np.argmin(d))
        return idx, float(d[idx])

    def tangent_xz(self, idx: int) -> np.ndarray:
        """Unit tangent in the X-Z plane at the given waypoint index."""
        j = (idx + 1) % self.n
        v = self._xz[j] - self._xz[idx]
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else np.array([1.0, 0.0])

    def tangent_heading(self, pos: np.ndarray) -> float:
        """Heading angle of the tangent in AC's convention: atan2(forward_x, forward_z).

        AC's shared-memory `heading` field uses this convention (angle measured
        from the +Z world axis), so heading_err = car.heading - tangent_heading
        is directly comparable.
        """
        idx, _ = self.nearest(pos)
        t = self.tangent_xz(idx)
        return float(np.arctan2(t[0], t[1]))

    def signed_lateral_error(self, pos: np.ndarray) -> float:
        idx, _ = self.nearest(pos)
        t = self.tangent_xz(idx)
        # Left-hand normal in the (x, z) plane.
        normal = np.array([-t[1], t[0]])
        delta = pos[_HORIZ] - self._xz[idx]
        return float(np.dot(delta, normal))

    def progress_m(self, pos: np.ndarray) -> float:
        idx, _ = self.nearest(pos)
        return float(self.cum_length[idx])

    def lookahead_points(self, pos: np.ndarray, distances_m) -> np.ndarray:
        """Return (len(distances_m), 3) XYZ points at the given arc distances ahead."""
        base = self.progress_m(pos)
        out = np.zeros((len(distances_m), 3))
        for i, d in enumerate(distances_m):
            target = (base + d) % self.total_length
            j = int(np.searchsorted(self.cum_length, target, side="right") - 1)
            j = max(0, min(j, self.n - 1))
            out[i] = self.points[j]
        return out
