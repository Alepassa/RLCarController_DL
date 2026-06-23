import time
import pathlib
import pandas as pd
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeLogger:
    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._buffer = []
        self._rs, self._lats, self._spds = [], [], []
        self._ep_id = 0
        self._t0 = time.time()

    def record_step(self, reward: float, lateral: float, speed: float):
        self._rs.append(reward)
        self._lats.append(abs(lateral))
        self._spds.append(speed)

    def end_episode(self, step_global: int, reason: str, lap_completed: bool, lap_time,
                    laps_completed: int = 0):
        if not self._rs:
            return
        rs = np.array(self._rs)
        lats = np.array(self._lats)
        spds = np.array(self._spds)
        self._buffer.append({
            "episode_id": self._ep_id,
            "step_global": step_global,
            "wall_time_s": time.time() - self._t0,
            "total_reward": float(rs.sum()),
            "length": int(len(rs)),
            "mean_lateral_error": float(lats.mean()),
            "max_lateral_error": float(lats.max()),
            "mean_speed": float(spds.mean()),
            "top_speed": float(spds.max()),
            "lap_completed": bool(lap_completed),
            "laps_completed": int(laps_completed),
            "lap_time": float(lap_time) if lap_time is not None else float("nan"),
            "termination_reason": reason,
        })
        self._ep_id += 1
        self._rs, self._lats, self._spds = [], [], []
        if len(self._buffer) >= 10:
            self.flush()

    def flush(self):
        if not self._buffer:
            return
        df_new = pd.DataFrame(self._buffer)
        if self.path.exists():
            df_old = pd.read_parquet(self.path)
            df = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df = df_new
        df.to_parquet(self.path, index=False)
        self._buffer.clear()


class StepSampler:
    def __init__(self, path, every: int):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.every = every
        self._buf = []
        self._i = 0

    def record(self, obs, action, reward, info):
        if self._i % self.every == 0:
            self._buf.append({
                "step": self._i,
                "obs": obs.tolist() if hasattr(obs, "tolist") else list(obs),
                "action": list(map(float, action)),
                "reward": float(reward),
                "lateral_error_m": float(info.get("lateral_error_m", 0.0)),
                "speed_kmh": float(info.get("speed_kmh", 0.0)),
                "spline_pos": float(info.get("spline_pos", 0.0)),
                "applied_steer": float(info.get("applied_steer", 0.0)),
            })
        self._i += 1
        if len(self._buf) >= 200:
            self.flush()

    def flush(self):
        if not self._buf:
            return
        df_new = pd.DataFrame(self._buf)
        if self.path.exists():
            df_old = pd.read_parquet(self.path)
            df = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df = df_new
        df.to_parquet(self.path, index=False)
        self._buf.clear()


class TrainingCallback(BaseCallback):
    def __init__(self, ep_logger: EpisodeLogger, step_sampler: StepSampler, verbose=0):
        super().__init__(verbose)
        self.ep = ep_logger
        self.ss = step_sampler

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        rewards = self.locals.get("rewards")
        actions = self.locals.get("actions")
        obs = self.locals.get("new_obs")
        dones = self.locals.get("dones")
        for i, info in enumerate(infos):
            self.ep.record_step(
                reward=float(rewards[i]) if rewards is not None else 0.0,
                lateral=float(info.get("lateral_error_m", 0.0)),
                speed=float(info.get("speed_kmh", 0.0)),
            )
            self.ss.record(
                obs=np.asarray(obs[i]) if obs is not None else np.zeros(1),
                action=actions[i] if actions is not None else [0.0, 0.0],
                reward=float(rewards[i]) if rewards is not None else 0.0,
                info=info,
            )
            if dones is not None and dones[i]:
                laps = int(info.get("laps_completed", 0))
                self.ep.end_episode(
                    step_global=self.num_timesteps,
                    reason=info.get("termination_reason", ""),
                    lap_completed=laps > 0,
                    lap_time=None,
                    laps_completed=laps,
                )
        return True

    def _on_training_end(self) -> None:
        self.ep.flush()
        self.ss.flush()
