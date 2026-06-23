import argparse
import pathlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save(fig, path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_run(run_dir: pathlib.Path):
    ep = pd.read_parquet(run_dir / "episodes.parquet")
    out = run_dir / "plots"

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ep["episode_id"], ep["total_reward"], alpha=0.3, label="raw")
    ax.plot(ep["episode_id"], ep["total_reward"].rolling(100, min_periods=1).mean(), label="100-ep mean")
    ax.set_xlabel("episode"); ax.set_ylabel("total reward"); ax.legend(); ax.set_title("Reward")
    _save(fig, out / "reward.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ep["episode_id"], ep["mean_lateral_error"].rolling(50, min_periods=1).mean())
    ax.set_xlabel("episode"); ax.set_ylabel("mean |lateral error| (m)")
    ax.set_title("Lateral error")
    _save(fig, out / "lateral.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    lap_frac = ep["lap_completed"].rolling(100, min_periods=1).mean()
    ax.plot(ep["episode_id"], lap_frac)
    ax.set_xlabel("episode"); ax.set_ylabel("frac episodes with lap completed (100-ep)")
    ax.set_ylim(0, 1)
    _save(fig, out / "laps.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ep["episode_id"], ep["top_speed"].rolling(50, min_periods=1).mean())
    ax.set_xlabel("episode"); ax.set_ylabel("top speed (km/h)")
    _save(fig, out / "top_speed.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    reasons = ep["termination_reason"].value_counts()
    reasons.plot(kind="bar", ax=ax)
    ax.set_title("Termination reasons (overall)")
    _save(fig, out / "termination_reasons.png")

    ss_path = run_dir / "steps_sampled.parquet"
    if ss_path.exists():
        ss = pd.read_parquet(ss_path)
        idx = np.linspace(0, len(ss) - 1, num=min(5_000, len(ss))).astype(int)
        ss = ss.iloc[idx]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(ss["spline_pos"], ss["lateral_error_m"], s=2, alpha=0.3)
        ax.set_xlabel("spline pos"); ax.set_ylabel("lateral error (m)")
        ax.set_title("Lateral error vs track position (sampled steps)")
        _save(fig, out / "lateral_vs_spline.png")

    print("plots written to", out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=pathlib.Path)
    args = p.parse_args()
    plot_run(args.run_dir)


if __name__ == "__main__":
    main()
