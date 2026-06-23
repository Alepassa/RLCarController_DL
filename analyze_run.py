"""One-shot analysis of a training run for the clamp-on-baseline experiment.

Usage:  python analyze_run.py [run_dir]   (default: runs/train_monza_clamp_v1)

Prints everything we watch: lap completion, terminations (esp. stopped vs offtrack),
the post-clamp steering saw (applied_steer sign-flips), where it fails, real forward
progress, and the trend over the run -- all compared to the v3_baseline reference.
Saves <run>/analysis.png. Read-only; robust to partial/in-progress runs.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

run = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/train_monza_clamp_v1")
BASELINE = Path("runs/train_monza_v3_baseline")
HZ = 20.0

ep_path = run / "episodes.parquet"
if not ep_path.exists():
    print(f"[!] {ep_path} not found — has the run produced episodes yet?")
    sys.exit(0)
ep = pd.read_parquet(ep_path).sort_values("episode_id").reset_index(drop=True)
ss = pd.read_parquet(run / "steps_sampled.parquet").sort_values("step").reset_index(drop=True) \
    if (run / "steps_sampled.parquet").exists() else pd.DataFrame()

ep["rps"] = ep["total_reward"] / ep["length"].clip(lower=1)
ep["dur_s"] = ep["length"] / HZ
n = len(ep)


def steer_series(df):
    """Post-clamp applied steering if logged, else the raw policy command action[0]."""
    if "applied_steer" in df.columns and df["applied_steer"].abs().sum() > 0:
        return df["applied_steer"], "applied_steer (post-clamp)"
    a = np.stack(df["action"].to_numpy())[:, 0]
    return pd.Series(a, index=df.index), "action[0] (raw, no applied_steer logged)"


def saw_metrics(s):
    sign = np.sign(s.fillna(0))
    flips = (sign.diff().abs() > 0).sum()
    return 100 * flips / max(len(s), 1), float(s.abs().mean())


# ---------- textual summary ----------
print(f"\n=== {run.name} ===")
print(f"episodi: {n} | step: {int(ep['step_global'].iloc[-1])} | durata media: {ep['dur_s'].mean():.1f}s")
print(f"reward/step media: {ep['rps'].mean():+.2f}")
print(f"GIRI: totali={int(ep['laps_completed'].sum())} | multi-giro(>1)={int((ep['laps_completed']>1).sum())}"
      f" | max/episodio={int(ep['laps_completed'].max())}   (baseline: 466 / 197 / 3)")
print("terminazioni:")
print(ep["termination_reason"].value_counts().to_string())
stopped_frac = (ep["termination_reason"] == "stopped").mean() * 100
print(f"  -> stopped = {stopped_frac:.0f}% degli episodi   (il rischio da sorvegliare)")

if len(ss):
    s, lbl = steer_series(ss)
    ss = ss.assign(_steer=s.values)
    mature = ss.iloc[len(ss) // 2:]
    flips, absmean = saw_metrics(mature["_steer"])
    print(f"\nSEGA (terzo maturo, {lbl}):")
    print(f"  sign-flips = {flips:.0f}%   |steer| medio = {absmean:.3f}   (baseline: 50% / 0.59)")

# ---------- forward-progress check (lesson learned: spline_reached can be backwards) ----------
print("\nPROGRESSO:")
print(f"  spline_reached: media={ep['spline_reached'].mean():.2f} max={ep['spline_reached'].max():.2f}"
      if "spline_reached" in ep.columns else "  (spline_reached non loggato in questo run)")
if len(ss):
    mid = ((ss["spline_pos"] > 0.3) & (ss["spline_pos"] < 0.7)).mean() * 100
    print(f"  frazione campioni nel CENTRO giro (s0.3-0.7): {mid:.0f}%  (se ~0 -> non guida mai il giro)")
    off_ep = ep[ep["termination_reason"].isin(["offtrack", "stopped", "lateral_excess", "no_progress"])]
    if "spline_final" in ep.columns and len(off_ep):
        h = pd.cut(off_ep["spline_final"], bins=[0, .1, .2, .3, .36, .45, .6, .75, .9, 1.0]).value_counts().sort_index()
        print("  dove finiscono gli episodi falliti (spline_final):")
        print("   ", {str(k): int(v) for k, v in h.items() if v})

# ---------- trend ----------
print("\nTREND (blocchi da ~1/8 del run):")
k = max(10, n // 8)
for i in range(0, n, k):
    b = ep.iloc[i:i + k]
    print(f"  ep {b['episode_id'].min():>3}-{b['episode_id'].max():>3}: "
          f"rps={b['rps'].mean():+6.1f} laps={int(b['laps_completed'].sum()):3} "
          f"dur={b['dur_s'].mean():4.1f}s lat={b['mean_lateral_error'].mean():.2f} "
          f"v={b['top_speed'].mean():3.0f} stopped%={(b['termination_reason']=='stopped').mean()*100:3.0f}")

# ---------- plots ----------
fig, ax = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle(f"{run.name} — {n} ep, {int(ep['step_global'].iloc[-1])} step", fontsize=13)

a = ax[0, 0]
a.plot(ep["episode_id"], ep["laps_completed"], ".", alpha=0.4)
a.plot(ep["episode_id"], ep["laps_completed"].rolling(15, min_periods=1).mean(), lw=2)
a.axhline(1, ls="--", color="gray"); a.set_title("Giri completati / episodio (target: >=1, multi-giro)")
a.set_xlabel("episodio"); a.grid(alpha=0.3)

a = ax[0, 1]
vc = ep["termination_reason"].value_counts()
a.barh(vc.index.astype(str), vc.values, color="C3")
a.set_title("Terminazioni (stopped = rischio)"); a.grid(alpha=0.3, axis="x")

a = ax[1, 0]
if len(ss):
    a.plot(ss["spline_pos"], ss["_steer"], ".", ms=3, alpha=0.3)
    a.set_title("Sterzo applicato vs posizione giro (sega = banda spessa)")
    a.set_xlabel("spline pos"); a.set_ylabel("applied steer"); a.grid(alpha=0.3)

a = ax[1, 1]
a.plot(ep["episode_id"], ep["dur_s"], ".", alpha=0.3)
a.plot(ep["episode_id"], ep["dur_s"].rolling(15, min_periods=1).mean(), lw=2, color="C1")
a.set_title("Durata episodio (s) — sopravvivenza"); a.set_xlabel("episodio"); a.grid(alpha=0.3)

plt.tight_layout()
out = run / "analysis.png"
plt.savefig(out, dpi=110)
print(f"\n[salvato] {out}")
