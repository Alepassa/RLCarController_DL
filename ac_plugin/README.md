# AC_RL_Bridge plugin

A minimal AC Python app that calls **`ac.ext_takeAStepBack()`** (CSP extension) on demand from the trainer.

**Why so minimal?** AC ships with a Python 3.3 build whose C-extension stdlib modules (`socket`, `mmap`, ...) often fail to load due to missing DLLs. We bypass the problem entirely by talking to the trainer through two tiny binary files.

## How it works

Two files under `~/Documents/Assetto Corsa/ac_rl_ipc/`:

| File | Direction | Format |
|------|-----------|--------|
| `cmd.bin` | trainer → plugin | 4 bytes LE uint32 (command seq) |
| `ack.bin` | plugin → trainer | 4 bytes LE uint32 (ack seq) |

Each frame the plugin reads `cmd.bin`. When `cmd_seq` changes vs the previous value, it calls `ac.ext_takeAStepBack()` and writes the new seq into `ack.bin`. Writes use `os.replace` so the trainer never sees a half-written file.

The plugin no longer exports the ai_line — the trainer reads `fast_lane.ai` directly from the track folder.

## Install

Copy **`ac_rl_bridge.py`** (and `ai_line_loader.py` if you want it for reference; not required at runtime) into:

```
Documents\Assetto Corsa\apps\python\AC_RL_Bridge\
```
or
```
<AC install>\apps\python\AC_RL_Bridge\
```

The folder name must be `AC_RL_Bridge` (matches `ac.newApp` call).

In AC: Options → General → enable "AC_RL_Bridge". Load a Practice session. Hover the right edge of the screen and click the AC_RL_Bridge sidebar entry to show the window — it should display:

```
ipc dir: C:\Users\<you>\Documents\Assetto Corsa\ac_rl_ipc
```

(After the first reset, it switches to `ack #N seq=M`.)

## Smoke test

With AC in Practice and the app enabled, from the project root:

```powershell
.venv\Scripts\python.exe scripts\reset_smoke.py
```

The car should be reset three times in sequence. If `NO ack within 2s` is printed, check `Documents\Assetto Corsa\logs\log.txt` for `[ac_rl_bridge]` entries.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `NO ack within 2s`, plugin window absent in-game | App not enabled in Options → General |
| Plugin window shows `ext_takeAStepBack not available` | CSP missing or outdated |
| Log shows `mkdir failed` or `write ack failed` | Permissions on `Documents\Assetto Corsa\ac_rl_ipc\`; clear the folder and retry |
| Plugin doesn't appear at all, log shows `ImportError` | Python 3.3 DLL issue — but this rewrite uses only built-ins, so it shouldn't happen |
