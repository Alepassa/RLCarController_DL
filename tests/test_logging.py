import tempfile, pathlib
import pandas as pd
from ac_rl.logging import EpisodeLogger

def test_episode_logger_writes_parquet():
    with tempfile.TemporaryDirectory() as d:
        log = EpisodeLogger(pathlib.Path(d) / "episodes.parquet")
        log.record_step(reward=1.0, lateral=0.5, speed=100.0)
        log.record_step(reward=0.5, lateral=0.7, speed=110.0)
        log.end_episode(step_global=2, reason="offtrack", lap_completed=False, lap_time=None)
        log.flush()
        df = pd.read_parquet(pathlib.Path(d) / "episodes.parquet")
        assert len(df) == 1
        assert df.iloc[0]["total_reward"] == 1.5
        assert df.iloc[0]["termination_reason"] == "offtrack"
        assert abs(df.iloc[0]["mean_speed"] - 105.0) < 1e-6
