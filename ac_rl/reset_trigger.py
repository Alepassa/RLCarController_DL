"""File-based IPC with the AC_RL_Bridge plugin.

Two binary files in ~/Documents/Assetto Corsa/ac_rl_ipc/:
- cmd.bin  (4 bytes LE uint32): trainer-written cmd seq
- ack.bin  (4 bytes LE uint32): plugin-written ack seq

Atomic writes via os.replace (NTFS-atomic for same-volume rename).
"""
import os
import struct
import time
from pathlib import Path


IPC_DIR = Path.home() / "Documents" / "Assetto Corsa" / "ac_rl_ipc"
CMD_PATH = IPC_DIR / "cmd.bin"
ACK_PATH = IPC_DIR / "ack.bin"


class ResetTrigger:
    def __init__(self, cmd_path: Path = CMD_PATH, ack_path: Path = ACK_PATH):
        self.cmd_path = Path(cmd_path)
        self.ack_path = Path(ack_path)
        self.cmd_path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = 0
        # Wipe any stale ack so wait_ack doesn't return immediately on first call
        try:
            self.ack_path.unlink()
        except FileNotFoundError:
            pass

    @staticmethod
    def _atomic_write_seq(path: Path, seq: int) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            f.write(struct.pack("<I", seq))
        # Retry os.replace: on Windows, if the AC plugin has cmd.bin open for
        # reading at the same instant, replace fails with PermissionError.
        # The plugin reads in microseconds, so 1-2 retries usually suffice.
        for _ in range(20):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                time.sleep(0.01)
        os.replace(tmp, path)  # last attempt; let it raise if still stuck

    @staticmethod
    def _read_seq(path: Path) -> int:
        try:
            with open(path, "rb") as f:
                data = f.read(4)
            if len(data) != 4:
                return 0
            return struct.unpack("<I", data)[0]
        except (FileNotFoundError, PermissionError, OSError):
            # Windows: os.replace by the other side can transiently block opens.
            return 0

    def trigger(self) -> None:
        self._seq += 1
        self._atomic_write_seq(self.cmd_path, self._seq)

    def last_ack(self) -> int:
        return self._read_seq(self.ack_path)

    def wait_ack(self, timeout_s: float = 2.0) -> bool:
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            if self.last_ack() == self._seq:
                return True
            time.sleep(0.01)
        return False
