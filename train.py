import argparse
import time
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from ac_rl.env import AssettoCorsaEnv
from ac_rl.config import Config
from ac_rl.logging import EpisodeLogger, StepSampler, TrainingCallback


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--run-id", type=str, default=time.strftime("run_%Y%m%d_%H%M%S"))
    p.add_argument("--checkpoint-every", type=int, default=20_000)
    p.add_argument("--warm-start", type=str, default=None,
                   help="Path to a SAC .zip to initialise weights from (fine-tuning). The "
                        "reward/env can differ from the source run; obs layout must match.")
    p.add_argument("--learning-starts", type=int, default=10_000,
                   help="Random-experience steps before gradient updates. Use a small value "
                        "(e.g. 2000) when warm-starting so the loaded policy is exploited fast.")
    p.add_argument("--learning-rate", type=float, default=None,
                   help="Override the optimizer learning rate (e.g. 1e-4 to keep a warm-started "
                        "policy close to the loaded weights).")
    p.add_argument("--ent-coef", type=str, default=None,
                   help="Override SAC entropy coefficient. Pass a fixed float (e.g. '0.05') to "
                        "disable auto-entropy (which re-inflates exploration on warm-start).")
    args = p.parse_args()

    cfg = Config()
    run_dir = cfg.runs_dir / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = run_dir / "tb"
    ck_dir = run_dir / "checkpoints"

    env = AssettoCorsaEnv(cfg)
    if args.warm_start:
        print(f"[warm-start] loading weights from {args.warm_start}")
        co = {"learning_starts": args.learning_starts, "tensorboard_log": str(tb_dir)}
        try:
            model = SAC.load(args.warm_start, env=env, device="cuda",
                             tensorboard_log=str(tb_dir), custom_objects=co)
        except (ValueError, KeyError, AttributeError):
            # The checkpoint was saved with a FIXED ent_coef (no ent_coef_optimizer) -- e.g. one
            # of our own --ent-coef fine-tunes. Reconstruct with a fixed ent_coef so the saved and
            # expected parameter sets match (auto-entropy baselines load fine on the first try).
            ec = float(args.ent_coef) if (args.ent_coef and not args.ent_coef.startswith("auto")) else 0.05
            co["ent_coef"] = ec
            model = SAC.load(args.warm_start, env=env, device="cuda",
                             tensorboard_log=str(tb_dir), custom_objects=co)
            print(f"[warm-start] fixed-entropy checkpoint; reloaded with ent_coef={ec}")
    else:
        model = SAC(
            "MlpPolicy", env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            batch_size=256,
            learning_starts=args.learning_starts,
            gamma=0.99,
            tau=0.005,
            train_freq=1,
            gradient_steps=1,
            tensorboard_log=str(tb_dir),
            verbose=1,
            device="cuda"
        )

    # Stability overrides (esp. for warm-start): lower LR keeps updates near the loaded policy;
    # a fixed small ent_coef disables SAC auto-entropy (which re-inflates exploration on warm-start).
    if args.learning_rate is not None:
        model.learning_rate = args.learning_rate
        model.lr_schedule = lambda _: args.learning_rate
        print(f"[override] learning_rate -> {args.learning_rate}")
    if args.ent_coef is not None:
        if args.ent_coef.startswith("auto"):
            raise SystemExit("--ent-coef expects a fixed float (e.g. 0.05) to disable auto-entropy")
        model.ent_coef_optimizer = None
        model.ent_coef_tensor = th.tensor(float(args.ent_coef), device=model.device)
        print(f"[override] ent_coef -> fixed {args.ent_coef} (auto-entropy disabled)")

    ep_log = EpisodeLogger(run_dir / "episodes.parquet")
    step_samp = StepSampler(run_dir / "steps_sampled.parquet", every=cfg.step_sample_every)
    cb = CallbackList([
        TrainingCallback(ep_log, step_samp),
        CheckpointCallback(save_freq=args.checkpoint_every, save_path=str(ck_dir), name_prefix="sac"),
    ])

    try:
        model.learn(total_timesteps=args.steps, callback=cb, tb_log_name="sac")
    finally:
        model.save(run_dir / "final.zip")
        ep_log.flush()
        step_samp.flush()
        env.close()


if __name__ == "__main__":
    main()
