import argparse
import pathlib
import numpy as np
from stable_baselines3 import SAC
from ac_rl.env import AssettoCorsaEnv
from ac_rl.config import Config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=pathlib.Path)
    p.add_argument("--laps", type=int, default=3)
    args = p.parse_args()

    env = AssettoCorsaEnv(Config())
    try:
        model = SAC.load(str(args.checkpoint), env=env)
    except (ValueError, KeyError):
        # Fixed-entropy checkpoint (no ent_coef_optimizer): retry with custom_objects
        model = SAC.load(str(args.checkpoint), env=env, custom_objects={"ent_coef": 0.1})

    laps_done = 0
    offtracks = 0
    total_reward = 0.0
    lat_errs = []
    obs, _ = env.reset()
    while laps_done < args.laps:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        total_reward += r
        lat_errs.append(abs(info["lateral_error_m"]))
        if info.get("lap_completed_now"):
            laps_done += 1
            print(f"lap {laps_done} completed  (speed={info['speed_kmh']:.0f} km/h  lat={info['lateral_error_m']:.2f} m)")
        if term:
            reason = info["termination_reason"]
            if reason == "timeout":
                print("timeout — reset and continue")
            else:
                offtracks += 1
                print(f"failed: {reason}")
                if offtracks > 5:
                    print("too many failures; abort")
                    break
            obs, _ = env.reset()
    print(f"reward total={total_reward:.1f}  mean|lat|={np.mean(lat_errs):.2f} m  laps={laps_done}/{args.laps}")
    env.close()


if __name__ == "__main__":
    main()
