# Learn to Drive a Car with Reinforcement Learning

## 👥 Group and Project Information

- **Group ID:** FiCo
- **Project ID:** 25

## 📝 Project Description

This project trains an autonomous racing agent directly inside **Assetto Corsa**, a physics-accurate racing simulator, using **Soft Actor-Critic (SAC)** — an off-policy, maximum-entropy deep reinforcement learning algorithm designed for continuous action spaces. The agent controls steering and throttle/brake of a Formula 1 car around the Monza circuit in real time, receiving observations from the simulator's shared memory at 20 Hz and outputting commands through a virtual joystick driver (vJoy).

Unlike supervised or imitation-learning approaches, the agent receives no pre-recorded driving data: it learns entirely from scratch through interaction with the simulator, guided by a shaped reward function that rewards progress along the AI racing line, penalises track-boundary violations, grass contact and corner overspeed, and suppresses steering oscillations via a curvature-gated term active only on straights. The SAC policy is represented by two MLP networks (actor and twin critics, 2×256 neurons each) and trains off-policy from a 10⁶-transition replay buffer. After approximately 300,000 environment steps (~4 hours on a mid-range GPU), the agent consistently completes full laps of Monza at competitive speeds.

## 📖 Official Report

For all theoretical details, performance analysis, the architecture used, and group contributions, please refer to our formal paper: [REPORT.md](docs/REPORT.md)

## 📁 Project Structure

```
RLCarController/
├── ac_rl/                      # Core RL package
│   ├── env.py                  # Gymnasium environment wrapping Assetto Corsa
│   ├── reward.py               # Reward function (progress, penalties, oscillation gate)
│   ├── observation.py          # 20-D observation builder
│   ├── config.py               # All hyperparameters (RewardWeights, Config)
│   ├── ai_line.py              # Racing line geometry and lateral error computation
│   ├── shared_memory.py        # AC physics data via Windows shared memory
│   ├── vjoy_controller.py      # vJoy interface for steering/throttle output
│   └── ...
├── ac_plugin/                  # Assetto Corsa Python plugin (bridge)
│   └── ac_rl_bridge.py         # Exposes IPC socket to the RL environment
├── scripts/                    # Utility scripts (smoke tests, AI line dump)
├── tests/                      # Unit test suite (pytest)
├── references/
│   └── monza_Yux100_airef.npz  # Monza AI racing line reference data
├── runs/
│   └── train_monza_smooth/     # Final trained model
│       ├── final.zip           # Ready-to-use SAC checkpoint
│       └── checkpoints/        # Intermediate checkpoints
├── train.py                    # Training entry point
├── eval.py                     # Evaluation entry point
├── plot_run.py                 # Plot training curves
└── requirements.txt
```

## 🛠 Technical Reproducibility

### 1. Environment Setup

**Prerequisites:**
- Assetto Corsa (Windows) with the Ferrari F1 car and Monza track installed
- [vJoy](http://vjoystick.sourceforge.net) virtual joystick driver
- Python 3.11, CUDA-capable GPU recommended for training

```bash
git clone https://github.com/Alepassa/RLCarController.git
cd RLCarController
pip install -r requirements.txt
```

Install the AC plugin (must be done once):
```bash
cp ac_plugin/ac_rl_bridge.py "<AssettoCorsaRoot>/apps/python/ac_rl_bridge/ac_rl_bridge.py"
```
Then enable the app in-game from Assetto Corsa's settings.

> Assetto Corsa must be running with an active session before launching training or evaluation.

### 2. Training

```bash
python train.py --steps 300000 --run-id my_run
```

Resume from a checkpoint:
```bash
python train.py --load-checkpoint runs/train_monza_smooth/final.zip --steps 100000 --run-id my_run_resumed
```

Training logs and checkpoints are saved to `runs/<run-id>/`. Monitor with TensorBoard:
```bash
tensorboard --logdir runs/my_run/tb
```

### 3. Evaluation

A fully trained model is included in the repository at `runs/train_monza_smooth/final.zip` — no training required to test it. Run it in deterministic mode (no exploration noise) for 3 laps:

```bash
python eval.py runs/train_monza_smooth/final.zip --laps 3
```

The script prints per-lap speed, mean lateral error, and total reward at the end.
