"""Manual smoke test for AC shared memory. Run with AC in Practice mode."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ac_rl.shared_memory import SharedMemoryReader

def main():
    r = SharedMemoryReader()
    s = r.read_static()
    print(f"track={str(s.trackName).strip(chr(0))!r}  config={str(s.trackConfig).strip(chr(0))!r}")
    for _ in range(20):
        p = r.read_physics()
        g = r.read_graphics()
        print(f"packet={p.packetId} speed={p.speedKmh:.1f} laps={g.completedLaps} pos={g.normalizedCarPosition:.3f}")
        time.sleep(0.1)
    r.close()


if __name__ == "__main__":
    main()
