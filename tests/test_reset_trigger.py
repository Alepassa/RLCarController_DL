"""Round-trip test for the file-based ResetTrigger.

We don't have the AC plugin running here, so we simulate its half of the
protocol with a tiny stub that reads cmd.bin and writes ack.bin.
"""
import os
import struct
import threading
import time
from pathlib import Path

from ac_rl.reset_trigger import ResetTrigger


def _stub_plugin(stop_event: threading.Event, cmd_path: Path, ack_path: Path):
    last = 0
    while not stop_event.is_set():
        try:
            with open(cmd_path, "rb") as f:
                data = f.read(4)
            seq = struct.unpack("<I", data)[0] if len(data) == 4 else 0
        except (FileNotFoundError, PermissionError, OSError):
            # On Windows, os.replace can transiently block opens — just retry.
            seq = 0
        if seq != 0 and seq != last:
            last = seq
            tmp = str(ack_path) + ".tmp"
            with open(tmp, "wb") as f:
                f.write(struct.pack("<I", seq))
            os.replace(tmp, ack_path)
        time.sleep(0.005)


def test_trigger_and_wait_ack(tmp_path):
    cmd = tmp_path / "cmd.bin"
    ack = tmp_path / "ack.bin"
    rt = ResetTrigger(cmd_path=cmd, ack_path=ack)
    stop = threading.Event()
    t = threading.Thread(target=_stub_plugin, args=(stop, cmd, ack), daemon=True)
    t.start()
    try:
        rt.trigger()
        assert rt.wait_ack(timeout_s=2.0)
        rt.trigger()
        assert rt.wait_ack(timeout_s=2.0)
    finally:
        stop.set()
        t.join(timeout=1.0)


def test_wait_ack_times_out_when_no_plugin(tmp_path):
    rt = ResetTrigger(cmd_path=tmp_path / "cmd.bin", ack_path=tmp_path / "ack.bin")
    rt.trigger()
    assert rt.wait_ack(timeout_s=0.2) is False
