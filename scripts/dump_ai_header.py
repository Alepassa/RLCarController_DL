"""Diagnostic: dump the raw header bytes of the current track's fast_lane.ai.

Prints hex bytes and several plausible interpretations so we can confirm the
exact field layout. Run with AC in Practice mode (any track):

    .venv\\Scripts\\python.exe scripts\\dump_ai_header.py
"""
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ac_rl.shared_memory import SharedMemoryReader
from ac_rl.ai_line_loader import find_ai_line_file


def main():
    sm = SharedMemoryReader()
    s = sm.read_static()
    track = str(s.trackName).strip("\x00").strip()
    config = str(s.trackConfig).strip("\x00").strip()
    print(f"track={track!r}  config={config!r}")
    path = find_ai_line_file(track, config)
    print(f"file: {path}")
    print(f"size: {path.stat().st_size} bytes")

    data = path.read_bytes()[:64]

    print("\nfirst 64 bytes (hex):")
    for i in range(0, len(data), 16):
        chunk = data[i:i + 16]
        hex_part = " ".join(f"{b:02x}" for b in chunk)
        print(f"  {i:04x}  {hex_part}")

    print("\nas 16 little-endian int32:")
    ints = struct.unpack_from("<16i", data, 0)
    for i, v in enumerate(ints):
        print(f"  off=0x{i*4:02x} ({i*4:2d}): {v}")

    print("\nas 16 little-endian float32 (same bytes):")
    floats = struct.unpack_from("<16f", data, 0)
    for i, v in enumerate(floats):
        print(f"  off=0x{i*4:02x} ({i*4:2d}): {v}")


if __name__ == "__main__":
    main()
