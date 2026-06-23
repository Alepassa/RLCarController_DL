"""Test the fast_lane.ai parser on synthetic data."""
import struct
import tempfile
import pathlib

import numpy as np
import pytest

from ac_rl.ai_line_loader import load_ai_line


def _make_fast_lane_bytes(points):
    # Layout: version=7, sample_count, two unused i32s, then 20-byte spline records.
    # Record fields: x f32, y f32, z f32, length f32, id i32.
    header = struct.pack("<iiii", 7, len(points), 0, 0)
    body = b"".join(
        struct.pack("<ffffi", x, y, z, 0.0, i) for i, (x, y, z) in enumerate(points)
    )
    return header + body


def test_load_ai_line_parses_xyz():
    pts_in = [(0.0, 0.0, 0.0), (1.5, 2.5, 3.5), (-10.0, 0.0, 100.0)]
    blob = _make_fast_lane_bytes(pts_in)
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "fast_lane.ai"
        p.write_bytes(blob)
        arr = load_ai_line(p)
    assert arr.shape == (3, 3)
    np.testing.assert_allclose(arr, np.array(pts_in), atol=1e-5)


def test_load_ai_line_rejects_truncated():
    blob = struct.pack("<iiii", 7, 100, 0, 0)  # header says 100 points but no body
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "fast_lane.ai"
        p.write_bytes(blob)
        with pytest.raises(ValueError):
            load_ai_line(p)


def test_load_ai_line_rejects_implausible_count():
    blob = struct.pack("<iiii", 7, -1, 0, 0)
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "fast_lane.ai"
        p.write_bytes(blob)
        with pytest.raises(ValueError):
            load_ai_line(p)
