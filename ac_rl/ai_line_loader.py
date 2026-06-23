"""Direct loader for AC's fast_lane.ai files (trainer side).

We read the file from disk instead of going through an in-game plugin —
AC's bundled Python 3.3 has too many DLL-loading issues to rely on it for
non-trivial IPC.

Verified file layout (Monza fast_lane.ai, header version 7):

    offset  size  field
       0     4    version (int32, == 7 on tested files)
       4     4    sample_count (int32)
       8     4    unused (zero)
      12     4    unused (zero)
      16+   20 * sample_count   spline records: (x f32, y f32, z f32,
                                length_from_start f32, id int32)
      ...           additional per-sample AI data (speed/gas/brake/...)
                    that we do not need; ignored.
"""
import os
import struct
from pathlib import Path
from typing import Iterable

import numpy as np


# Common Steam install locations; override via `ac_install` argument if needed.
DEFAULT_AC_BASES = [
    Path(os.path.expandvars(r"%PROGRAMFILES(X86)%\Steam\steamapps\common\assettocorsa")),
    Path(os.path.expandvars(r"%PROGRAMFILES%\Steam\steamapps\common\assettocorsa")),
    Path("C:/Program Files (x86)/Steam/steamapps/common/assettocorsa"),
    Path("D:/SteamLibrary/steamapps/common/assettocorsa"),
    Path("E:/SteamLibrary/steamapps/common/assettocorsa"),
]


def _candidate_bases(extra: Iterable[Path] = ()) -> list[Path]:
    return [p for p in (list(extra) + DEFAULT_AC_BASES) if p and p.exists()]


def find_ai_line_file(
    track_name: str,
    track_config: str = "",
    ac_install: Path | None = None,
) -> Path:
    """Locate fast_lane.ai for a given track + optional layout."""
    bases = _candidate_bases([ac_install] if ac_install else [])
    if not bases:
        raise FileNotFoundError(
            "no AC install found; pass ac_install=Path('...') or set PROGRAMFILES(X86)"
        )

    tried = []
    for base in bases:
        track_dir = base / "content" / "tracks" / track_name
        candidates = [track_dir / track_config / "ai", track_dir / "ai"] if track_config else [track_dir / "ai"]
        for ai_dir in candidates:
            for name in ("fast_lane.ai", "fast_lane.ai_lane"):
                p = ai_dir / name
                tried.append(p)
                if p.exists():
                    return p
    raise FileNotFoundError(
        "fast_lane.ai not found for track={!r} config={!r}; tried:\n  {}".format(
            track_name, track_config, "\n  ".join(str(t) for t in tried)
        )
    )


def load_ai_line(path: Path) -> np.ndarray:
    """Parse fast_lane.ai and return an (N, 3) float64 array of XYZ points."""
    data = Path(path).read_bytes()
    if len(data) < 16:
        raise ValueError("fast_lane.ai too small: {} bytes".format(len(data)))
    _version, count, _u1, _u2 = struct.unpack_from("<iiii", data, 0)
    if count <= 0 or count > 1_000_000:
        raise ValueError("fast_lane.ai: implausible sample count {}".format(count))

    offset = 16
    SPLINE_REC = 4 + 4 + 4 + 4 + 4  # x, y, z, length_from_start, id
    expected = offset + SPLINE_REC * count
    if len(data) < expected:
        raise ValueError(
            "fast_lane.ai truncated: need {} bytes for {} points, got {}".format(
                expected, count, len(data)
            )
        )

    pts = np.zeros((count, 3), dtype=np.float64)
    for i in range(count):
        x, y, z = struct.unpack_from("<fff", data, offset)
        pts[i] = (x, y, z)
        offset += SPLINE_REC
    return pts
