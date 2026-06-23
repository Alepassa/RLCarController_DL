"""Standalone smoke test for the AC_RL_Bridge file-based reset.

Prerequisites:
    1. AC_RL_Bridge plugin installed and enabled in AC (see ac_plugin/README.md).
    2. CSP installed (provides ac.ext_takeAStepBack).
    3. AC running in Practice mode on any track.

Run from the project root:
    .venv\\Scripts\\python.exe scripts\\reset_smoke.py
"""
import sys
import time
from pathlib import Path

# Make the project root importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ac_rl.reset_trigger import ResetTrigger, CMD_PATH, ACK_PATH


def main():
    print(f"cmd file: {CMD_PATH}")
    print(f"ack file: {ACK_PATH}")
    rt = ResetTrigger()
    for i in range(3):
        print(f"-> reset #{i + 1}")
        rt.trigger()
        if rt.wait_ack(timeout_s=2.0):
            print(f"   ack received (ack_seq={rt.last_ack()})")
        else:
            print(f"   NO ack within 2s — last ack={rt.last_ack()}, expected={rt._seq}")
            print("   troubleshooting: check the AC_RL_Bridge app window in AC,")
            print("                    and Documents/Assetto Corsa/logs/log.txt for [ac_rl_bridge].")
        time.sleep(2.0)


if __name__ == "__main__":
    main()
