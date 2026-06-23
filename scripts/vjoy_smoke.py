"""Manual smoke test for vJoy. Open joy.cpl while running to verify axes."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ac_rl.vjoy_controller import VJoyController

c = VJoyController()
print("Sweeping steer left/right; check Windows Game Controller properties (joy.cpl).")
for v in [-1.0, -0.5, 0.0, 0.5, 1.0]:
    c.set(v, 0.0)
    print(f"steer={v}")
    time.sleep(1.0)
c.set(0.0, 1.0); time.sleep(1); print("full gas")
c.set(0.0, -1.0); time.sleep(1); print("full brake")
c.neutral()
