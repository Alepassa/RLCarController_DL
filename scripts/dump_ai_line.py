"""Diagnostic: discover the current track via shared memory and load its ai_line.

Run with AC in Practice mode on the track you want to inspect:
    .venv\\Scripts\\python.exe scripts\\dump_ai_line.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ac_rl.shared_memory import SharedMemoryReader
from ac_rl.ai_line_loader import find_ai_line_file, load_ai_line


def main():
    sm = SharedMemoryReader()
    s = sm.read_static()
    track = str(s.trackName).strip("\x00").strip()
    config = str(s.trackConfig).strip("\x00").strip()
    print(f"track={track!r}  config={config!r}")
    if not track:
        print("ERROR: track name empty — is AC running and loaded into a session?")
        return
    path = find_ai_line_file(track, config)
    print(f"ai_line file: {path}")
    pts = load_ai_line(path)
    print(f"loaded {len(pts)} points; bbox: x={pts[:,0].min():.1f}..{pts[:,0].max():.1f} "
          f"z={pts[:,2].min():.1f}..{pts[:,2].max():.1f}")
    print(f"first 3: {pts[:3].tolist()}")


if __name__ == "__main__":
    main()
