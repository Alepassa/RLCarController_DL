import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ac_rl.config import Config
from ac_rl.ai_line import AILine
from ac_rl.observation import CarState, build_observation
from ac_rl.reward import RewardInputs, compute_reward
from ac_rl.termination import TerminationTracker
from ac_rl.shared_memory import SharedMemoryReader
from ac_rl.vjoy_controller import VJoyController
from ac_rl.reset_trigger import ResetTrigger
from ac_rl.ai_line_loader import find_ai_line_file, load_ai_line
from ac_rl.steering import apply_steer_rate_limit


class AssettoCorsaEnv(gym.Env):
    """Gymnasium env for Assetto Corsa via shared memory + vJoy.

    The ai_line is loaded directly from disk (the trainer reads
    fast_lane.ai). Reset is triggered via file IPC with the AC_RL_Bridge
    Python plugin, which calls ac.ext_takeAStepBack() (CSP extension).
    Reset and step are paced at cfg.control_hz.
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: Config | None = None):
        super().__init__()
        self.cfg = cfg or Config()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.cfg.obs_dim,), dtype=np.float32
        )

        self.sm = SharedMemoryReader()
        self.joy = VJoyController()
        self.reset_trigger = ResetTrigger()

        # Discover the active track from shared memory and load its ai_line.
        s = self.sm.read_static()
        track = str(s.trackName).strip("\x00").strip()
        config = str(s.trackConfig).strip("\x00").strip()
        if not track:
            raise RuntimeError("track name not available in acpmf_static; is AC running?")
        ai_path = find_ai_line_file(track, config)
        pts = load_ai_line(ai_path)
        self.line = AILine(pts)

        self.term = TerminationTracker(self.cfg)
        self._prev_steer = 0.0
        self._prev_throttle = 0.0
        self._last_packet = 0
        self._last_progress_m = 0.0
        self._last_pos = None  # previous-step position, for the corner-aware clamp curvature lookup
        self._dt = 1.0 / self.cfg.control_hz

    def _read_state(self) -> tuple[CarState, float, float]:
        p = self.sm.wait_new_physics(self._last_packet, timeout_s=0.5)
        g = self.sm.read_graphics()
        self._last_packet = p.packetId
        pos = np.array([g.carCoordinates[0], g.carCoordinates[1], g.carCoordinates[2]])
        vel = np.array([p.velocity[0], p.velocity[1], p.velocity[2]])
        state = CarState(
            position=pos,
            velocity=vel,
            heading_rad=float(p.heading),
            yaw_rate=0.0,  # not in struct directly; left at 0 for now
            num_tyres_out=int(p.numberOfTyresOut),
        )
        return state, float(p.speedKmh), float(g.normalizedCarPosition)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.joy.neutral()
        self.reset_trigger.trigger()
        if not self.reset_trigger.wait_ack(timeout_s=2.0):
            # Plugin didn't ack — likely not running. Continue anyway; AC may
            # have processed the reset just slowly, or the user needs to debug.
            pass
        time.sleep(0.5)  # let physics settle
        self.term.reset()
        self._prev_steer = 0.0
        self._prev_throttle = 0.0
        state, _, _ = self._read_state()
        self._last_progress_m = self.line.progress_m(state.position)
        self._last_pos = state.position
        obs = build_observation(state, self.line, 0.0, 0.0, self.cfg)
        return obs, {}

    def step(self, action):
        steer_raw = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip(action[1], -1.0, 1.0))

        # Corner-aware steering rate-limit: tight on straights (kills the ondeggio), loose in
        # corners (fast turn-in for the chicane). Curvature looked up at the previous position
        # (close enough; the window itself looks clamp_ahead_m ahead). The applied command feeds
        # vJoy, the observation and the reward's delta_steer.
        curv = self.line.curvature_ahead(self._last_pos, ahead_m=self.cfg.clamp_ahead_m,
                                          behind_m=self.cfg.clamp_behind_m) \
            if self._last_pos is not None else 0.0
        frac = min(1.0, curv / self.cfg.clamp_curv_full) if self.cfg.clamp_curv_full > 0 else 1.0
        rate = self.cfg.clamp_straight + (self.cfg.clamp_corner - self.cfg.clamp_straight) * frac
        steer = apply_steer_rate_limit(self._prev_steer, steer_raw, rate * self._dt)
        self.joy.set(steer, throttle)

        t_start = time.perf_counter()
        state, speed_kmh, spline_pos = self._read_state()

        lateral = self.line.signed_lateral_error(state.position)
        lateral_norm = lateral / self.cfg.track_half_width_m
        tangent_h = self.line.tangent_heading(state.position)
        heading_err = float(np.arctan2(
            np.sin(state.heading_rad - tangent_h),
            np.cos(state.heading_rad - tangent_h),
        ))

        prog_m = self.line.progress_m(state.position)
        d_prog = prog_m - self._last_progress_m
        if d_prog < -self.line.total_length / 2:
            d_prog += self.line.total_length  # wrap
        self._last_progress_m = prog_m
        self._last_pos = state.position

        reason = self.term.update(
            offtrack=state.num_tyres_out >= 3,
            lateral_norm=lateral_norm,
            speed_kmh=speed_kmh,
            spline_pos=spline_pos,
        )

        offtrack_terminal = reason in ("offtrack", "lateral_excess")
        r = compute_reward(
            RewardInputs(
                progress_delta_m=d_prog,
                lateral_error_m=lateral,
                heading_error_rad=heading_err,
                delta_steer=steer - self._prev_steer,
                delta_throttle=throttle - self._prev_throttle,
                offtrack_terminal=offtrack_terminal,
                speed_kmh=speed_kmh,
                curvature=curv,  # same 30 m-ahead curvature used by the clamp; gates the straight penalty
                lap_completed_now=self.term.lap_completed_now,
            ),
            self.cfg,
        )

        terminated = reason is not None
        truncated = False
        obs = build_observation(state, self.line, steer, throttle, self.cfg)
        self._prev_steer, self._prev_throttle = steer, throttle

        elapsed = time.perf_counter() - t_start
        if elapsed < self._dt:
            time.sleep(self._dt - elapsed)

        info = {
            "termination_reason": reason or "",
            "lateral_error_m": lateral,
            "speed_kmh": speed_kmh,
            "progress_delta_m": d_prog,
            "spline_pos": spline_pos,
            "applied_steer": steer,
            "lap_completed_now": self.term.lap_completed_now,
            "laps_completed": self.term.laps_completed,
        }
        return obs, float(r), terminated, truncated, info

    def close(self):
        try: self.joy.neutral()
        except Exception: pass
        try: self.sm.close()
        except Exception: pass
