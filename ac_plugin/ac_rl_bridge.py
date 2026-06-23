# ac_plugin/ac_rl_bridge.py
# Runs inside Assetto Corsa. AC's bundled Python is 3.3 and its C-extension
# stdlib modules (socket, mmap, ...) often fail to load due to missing/old
# DLLs. To stay robust we only use built-ins + stdlib that are pure Python
# (or compiled into the interpreter): os, struct, open(), ac.
#
# IPC protocol with the external trainer:
#   ~/Documents/Assetto Corsa/ac_rl_ipc/cmd.bin   (4 bytes LE uint32, written by trainer)
#   ~/Documents/Assetto Corsa/ac_rl_ipc/ack.bin   (4 bytes LE uint32, written by this plugin)
#
# When cmd_seq changes vs the last value we saw, call ac.ext_takeAStepBack()
# (CSP extension) and write the same seq into ack.bin so the trainer knows
# the reset has been applied.

import os
import struct
import ac

IPC_DIR = os.path.join(os.path.expanduser("~"), "Documents", "Assetto Corsa", "ac_rl_ipc")
CMD_PATH = os.path.join(IPC_DIR, "cmd.bin")
ACK_PATH = os.path.join(IPC_DIR, "ack.bin")

_label = None
_last_seq = 0
_count = 0
_status = "init"


def _safe_log(msg):
    try:
        ac.log("[ac_rl_bridge] " + str(msg))
    except Exception:
        pass


def _ensure_dir():
    try:
        if not os.path.isdir(IPC_DIR):
            os.makedirs(IPC_DIR)
    except Exception as e:
        _safe_log("mkdir failed: " + repr(e))


def _read_seq(path):
    try:
        with open(path, "rb") as f:
            data = f.read(4)
        if len(data) != 4:
            return 0
        return struct.unpack("<I", data)[0]
    except Exception:
        return 0


def _atomic_write_seq(path, seq):
    tmp = path + ".tmp"
    try:
        with open(tmp, "wb") as f:
            f.write(struct.pack("<I", seq))
        os.replace(tmp, path)
    except Exception as e:
        _safe_log("write ack failed: " + repr(e))


def acMain(ac_version):
    global _label, _status
    _ensure_dir()
    app = ac.newApp("AC_RL_Bridge")
    ac.setSize(app, 240, 80)
    _label = ac.addLabel(app, "starting...")
    _status = "ipc dir: " + IPC_DIR
    _safe_log("started; " + _status)
    return "AC_RL_Bridge"


def acUpdate(deltaT):
    global _last_seq, _count, _status

    seq = _read_seq(CMD_PATH)
    if seq != 0 and seq != _last_seq:
        _last_seq = seq
        try:
            ac.ext_takeAStepBack()
            _count += 1
            _atomic_write_seq(ACK_PATH, seq)
            _status = "ack #" + str(_count) + " seq=" + str(seq)
            _safe_log(_status)
        except AttributeError:
            _status = "ext_takeAStepBack not available (CSP missing?)"
            _safe_log(_status)
        except Exception as e:
            _status = "reset err: " + repr(e)
            _safe_log(_status)

    if _label is not None:
        try:
            ac.setText(_label, _status)
        except Exception:
            pass
