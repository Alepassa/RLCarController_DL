"""End-to-end smoke for the gym env.

Three phases so the teleport is easy to see in-game:

  1. After env.__init__ + initial reset, drive forward 100 steps with gentle gas.
  2. Trigger another env.reset() — the car should snap back to the previous
     checkpoint (ac.ext_takeAStepBack). You should see this in AC.
  3. Drive forward again for 100 more steps.

Prerequisites (Practice mode):
    - vJoy device #1 bound as the AC controller: X=steer, Y=gas, Z=brake.
    - AC_RL_Bridge plugin enabled (window visible in-game).
    - A car loaded on a track that has fast_lane.ai.

Run from project root:
    .venv\\Scripts\\python.exe scripts\\env_smoke.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from ac_rl.env import AssettoCorsaEnv


def _drive(env, action, steps, label):
    print(f"--- {label}: {steps} steps with action steer={action[0]:+.2f} throttle={action[1]:+.2f}")
    total_r = 0.0
    obs = None
    for i in range(steps):
        obs, r, term, trunc, info = env.step(action)
        total_r += r
        if i % 20 == 0:
            print(
                f"  step={i:3d} r={r:+7.3f} lat={info['lateral_error_m']:+6.2f}m "
                f"v={info['speed_kmh']:6.1f}km/h spline={info['spline_pos']:.3f} "
                f"reason={info['termination_reason']!r}"
            )
        if term:
            print(f"  terminated at step {i}: {info['termination_reason']!r}")
            return total_r, True
    return total_r, False


def main():
    env = AssettoCorsaEnv()
    print(f"ai_line: {len(env.line.points)} points, total length {env.line.total_length:.1f} m")

    obs, _ = env.reset()
    print(f"initial obs (first 6 dims): {obs[:6]}")

    action = np.array([0.0, 0.3], dtype=np.float32)  # gentle gas, no steer

    total = 0.0
    r, terminated = _drive(env, action, 100, "PHASE 1: drive forward")
    total += r
    if terminated:
        env.close()
        print(f"total reward: {total:.2f}")
        return

    print("\n>>> RESETTING — watch the car snap back to the previous checkpoint <<<\n")
    time.sleep(1.0)
    env.reset()
    time.sleep(1.0)

    r, _ = _drive(env, action, 100, "PHASE 2: drive forward after reset")
    total += r

    env.close()
    print(f"\ntotal reward: {total:.2f}")


if __name__ == "__main__":
    main()
