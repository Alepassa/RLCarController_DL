from ac_rl.config import Config


class TerminationTracker:
    """Decides whether an episode should end and tracks per-episode counters.

    Lap completion is NOT a termination — it is exposed via `lap_completed_now`
    (True only for the step where the S/F line is crossed) and `laps_completed`
    (cumulative count this episode). The episode ends only on offtrack /
    lateral_excess / stopped / no_progress / timeout. This lets a well-behaved
    agent run multiple laps in a single episode instead of getting reset
    after every lap.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dt = 1.0 / cfg.control_hz
        self.offtrack_t = 0.0
        self.stopped_t = 0.0
        self.no_prog_t = 0.0
        self.total_t = 0.0
        self.last_spline = None
        self.last_spline_for_progress = None
        self.laps_completed = 0
        self.lap_completed_now = False

    def reset(self):
        self.offtrack_t = self.stopped_t = self.no_prog_t = self.total_t = 0.0
        self.last_spline = None
        self.last_spline_for_progress = None
        self.laps_completed = 0
        self.lap_completed_now = False

    def update(self, offtrack: bool, lateral_norm: float, speed_kmh: float, spline_pos: float):
        cfg = self.cfg
        self.total_t += self.dt

        self.offtrack_t = self.offtrack_t + self.dt if offtrack else 0.0
        self.stopped_t = self.stopped_t + self.dt if speed_kmh < cfg.stopped_speed_kmh else 0.0

        if self.last_spline_for_progress is None:
            self.last_spline_for_progress = spline_pos
            self.no_prog_t = 0.0
        else:
            delta = (spline_pos - self.last_spline_for_progress) % 1.0
            if delta < 0.001:
                self.no_prog_t += self.dt
            else:
                self.no_prog_t = 0.0
                self.last_spline_for_progress = spline_pos

        self.lap_completed_now = (
            self.last_spline is not None and self.last_spline > 0.9 and spline_pos < 0.1
        )
        if self.lap_completed_now:
            self.laps_completed += 1
        self.last_spline = spline_pos

        if abs(lateral_norm) > cfg.lateral_error_terminate_norm:
            return "lateral_excess"
        if self.offtrack_t >= cfg.offtrack_duration_s:
            return "offtrack"
        if self.stopped_t >= cfg.stopped_duration_s:
            return "stopped"
        if self.no_prog_t >= cfg.no_progress_duration_s:
            return "no_progress"
        if self.total_t >= cfg.max_episode_seconds:
            return "timeout"
        return None
