import struct
import os

def load_ai_line(track_path):
    """Return list of [x, y, z] for each point in the track's fast_lane.ai."""
    candidates = [
        os.path.join(track_path, "ai", "fast_lane.ai"),
        os.path.join(track_path, "ai", "fast_lane.ai_lane"),
    ]
    path = None
    for c in candidates:
        if os.path.exists(c):
            path = c
            break
    if path is None:
        raise IOError("no fast_lane.ai found under " + track_path)

    with open(path, "rb") as f:
        data = f.read()

    # Header: 4 int32s (version, detail, lapTime, sampleCount)
    _version, _detail, _lapTime, count = struct.unpack_from("<iiii", data, 0)
    offset = 16
    SPLINE_REC = 4 + 4 + 4 + 4 + 4   # x y z length id
    points = []
    for i in range(count):
        x, y, z = struct.unpack_from("<fff", data, offset)
        points.append([x, y, z])
        offset += SPLINE_REC
    return points
