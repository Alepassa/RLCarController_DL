# Pure Reinforcement Learning for AI-Line Following in Assetto Corsa
- **Group ID**: FiCo
- **Project ID**: 25

---

## 1. Introduction and Objective

The goal of this project is to train a reinforcement-learning agent (no imitation
learning, no behavior cloning, no human warm-start) to drive a Formula car — the Ferrari R25
(F2004) — around real Assetto Corsa tracks, following the track's `ai_line` at free speed. The
agent learns from scratch to control **steering** and **throttle/brake** through a virtual
joystick (vJoy), receiving the game state through shared memory.

The problem is relevant because it tackles continuous control in a *real, non-instrumented
simulator* rather than a custom physics toy: the environment is the actual game running in real
time, with all the noise, latency and fragility that entails. The driving philosophy is to impose
no prior on how to drive — the policy must discover the trajectory, the speed profile and the
steering management on its own, guided only by the reward.

The initial hypothesis is that a *simple* reward built around a strong "glue-to-line" term
(quadratic lateral error) plus a progress incentive is enough to learn lap completion. The success
criterion, fixed up front, is threefold: (1) on the training track, complete **3 consecutive laps**
without going off-track, with mean `lateral_error` **< 1 m**; (2) on an unseen evaluation track,
complete **at least one lap** off-track-free; (3) qualitatively **smooth** behavior — no hysterical
steering oscillation. Criterion (3) turned out to drive most of the experimental work: **the baseline
reaches the laps but with a "sawtooth" wheel (left-right oscillation on straights), and nearly all
later experiments are attempts to kill that oscillation without destroying lap-completion ability.**

## 2. Contribution and Added Value

We built, from scratch, a full RL training stack that drives **the real Assetto Corsa game** — not
a simulator we control. Three processes cooperate on a single Windows PC: the game itself, a small
in-game plugin (`AC_RL_Bridge`) whose only job is to teleport/reset the car on command, and a Python
trainer implementing a `gymnasium.Env` that reads physics from shared memory, writes actions through
vJoy, loads the `ai_line` from disk, computes reward/termination, and runs **SAC** (Soft
Actor-Critic, Stable-Baselines3).

Our added value over "running existing code" is substantial and concentrated in three areas:

- **A custom, track-agnostic environment.** A 20-dimensional observation built entirely from
  relative, normalized features (signed lateral error, heading error, a six-point curvature-aware
  look-ahead of the racing line rotated into the car's local frame, previous actions) so that the
  policy generalizes across tracks with no absolute coordinates. A geometry layer parses AC's binary
  `fast_lane.ai` file and exposes nearest-point, signed lateral error, tangent heading, arc-length
  progress, look-ahead and **curvature** queries, all projected onto the X-Z plane in AC's heading
  convention.
- **A novel reward + actuator co-design to remove steering oscillation.** Our key contribution is a
  **corner-aware steering rate-limit** coupled with a **curvature-gated steering penalty**. Both are
  driven by the same look-ahead curvature of the racing line, so the actuator and the objective stay
  consistent: on straights both suppress steering (killing the saw at its source), in corners both
  release it (so the chicane stays drivable). This cut the steering-activity metric `|steer|` from
  **0.59 to 0.15** while still completing laps every time in deterministic evaluation.
- **A robust, debuggable infrastructure.** A file-based 4-byte IPC for resets (after UDP and
  shared-memory protocols proved too fragile against AC's internal Python build), per-run Parquet
  logging of episodes and sampled steps, deterministic evaluation, and a "pure adapter" code design
  that makes almost the entire system unit-testable **offline without the game running** (10 test
  suites).

Across the project we ran ~54 training experiments organized into six families of hypotheses, each
with its own design spec, plan and branch.

## 3. Data Used

This is an online RL project, so the "data" is not a fixed dataset but the **interaction stream** the
agent generates with the live game, plus one static geometric asset.

- **Static asset — the racing line.** The only external data is AC's `fast_lane.ai` binary for the
  track, parsed directly from the Steam track folder (header version 7: an int32 version, an int32
  sample count, then 20-byte spline records of `x, y, z, length_from_start, id`). The track name is
  read from shared memory, so the environment **auto-discovers** the active track and loads its line.
  We trained on **Monza**.
- **Experience data — generated online.** The agent interacts at a **20 Hz** control loop (50 ms per
  step). State comes from three memory-mapped pages exposed by AC: physics (~333 Hz: speed, velocity,
  heading, tyres-out), graphics (~60 Hz: normalized spline position, world coordinates, completed
  laps) and static (track/car identity). Each step is synchronized to the real physics tick via a
  short busy-wait on the physics `packetId`, so observation and action are aligned to the simulator
  rather than to wall-clock time. There is no train/test split in the supervised sense; the protocol
  is **train stochastically, evaluate deterministically** on the same and on unseen tracks.
- **Preprocessing / feature engineering.** Every observation feature is normalized to roughly
  `[-1, 1]` (lateral error / 6 m half-width, heading error / π, speed / 300 km/h, look-ahead points
  / 160 m, etc.). The six look-ahead points (at 5/10/20/40/80/160 m of arc length) are **rotated into
  the car's local forward/left frame** using the AC heading convention; this rotation is what makes
  the representation independent of global orientation and track.
- **Scale.** ~54 runs were logged; the reference baseline alone accumulated ~1.5 M environment steps
  and 466 completed laps. Per-run artifacts are Parquet files (one row per episode; one sampled step
  every 100) plus SAC checkpoints every 20 k steps.

## 4. Methodology and Architecture

### 4.1 Soft Actor-Critic

The reinforcement learning algorithm at the core of this project is **Soft Actor-Critic (SAC)** [Haarnoja et al., 2018], an off-policy, maximum-entropy deep RL algorithm designed for continuous control. The choice is motivated by three structural requirements of the problem.

#### Continuous Action Space

Classical RL algorithms such as DQN operate over discrete action sets and are therefore unsuitable for driving control, where steering and throttle are physically continuous quantities. SAC addresses this by parameterizing the policy as a Gaussian distribution: for each action dimension, the actor network outputs a mean $\mu$ and a standard deviation $\sigma$. At each step, an action is sampled via the reparameterization trick:

$$a = \tanh(\mu + \sigma \cdot \varepsilon), \qquad \varepsilon \sim \mathcal{N}(0,1)$$

where $\varepsilon$ is drawn externally. This keeps the computation graph fully differentiable and allows gradient-based policy updates. The $\tanh$ squashing function constrains the output to $(-1,1)$, matching the physical range of the actuators.

#### Sample Efficiency

Assetto Corsa cannot be parallelized: the game exposes a single physics instance through shared memory, so the agent interacts with exactly one environment at a time. This makes sample efficiency non-negotiable. On-policy methods such as PPO discard every collected transition after each gradient update, requiring continuous interaction. SAC is off-policy: every transition $(s,a,r,s')$ is stored in a replay buffer of capacity $10^6$ and reused across multiple gradient steps, substantially reducing the number of real simulator interactions needed to reach a competent policy.

#### Stability and Exploration

Earlier off-policy algorithms such as DDPG are notoriously brittle: they frequently collapse into degenerate local optima (e.g., a near-stationary policy) and are highly sensitive to hyperparameter choices [Haarnoja et al., 2018]. SAC avoids this through the maximum-entropy objective:

$$J(\pi) = \sum_t \mathbb{E}_{(s_t,a_t)\sim\rho_\pi}\!\left[r(s_t,a_t) + \alpha\,\mathcal{H}\!\left(\pi(\cdot|s_t)\right)\right]$$

By adding the entropy term $\alpha \cdot \mathcal{H}(\pi)$ to the reward, the agent is incentivized to remain stochastic unless a decisive action is clearly superior. This prevents premature convergence and, as noted in the original paper, makes the policy robust to model errors and disturbances—a relevant property in a physics simulator with variable grip and suspension dynamics.

### 4.2 Network Architecture

SAC is instantiated with an `MlpPolicy` on CUDA, comprising three fully-connected networks each with two hidden layers of 256 units and ReLU activations—the same architecture reported in the original paper.

#### Actor Network $\pi_\phi(a|s)$

Maps the 20-dimensional observation to $(\mu,\sigma)$ for each of the two action dimensions (steer, throttle/brake). Outputs are squashed via $\tanh$.

#### Twin Critic Networks $Q_{\theta_1}, Q_{\theta_2}$

Independently estimate the soft Q-value $Q(s,a)$ from the concatenation of state and action.
At every gradient step, the minimum of the two estimates is used for both the policy gradient and the Bellman target—the **clipped double-Q trick** [Fujimoto et al., 2018]
that prevents the overestimation bias known to destabilize value-based methods in continuous spaces.

#### Soft Bellman Target

The critic update minimizes the residual with respect to:

$$\hat{Q} = r + \gamma\left[\min(Q_{\theta_1},Q_{\theta_2}) - \alpha\log\pi\right]$$

where the entropy term $-\alpha \log \pi$ teaches the critics to assign higher value to states that preserve future freedom of action.

#### Hyperparameters

```text
learning_rate   = 3e-4
buffer_size     = 1_000_000
batch_size      = 256
learning_starts = 10_000
gamma           = 0.999
tau             = 0.005
train_freq      = 2
gradient_steps  = 2
```

### 4.3 Action Space

The action space consists of two continuous signals sent to the simulator via vJoy:

- **steer** $\in [-1,1]$: mapped to vJoy's X axis. Full left at $-1$, full right at $+1$.
- **throttle** $\in [-1,1]$: sign-exclusive. Positive values command gas, negative values command brake; the two are never applied simultaneously.

This collapses the physically separate throttle and brake pedals into a single bipolar signal, reducing the action dimensionality and eliminating the unrealistic scenario of simultaneous braking and acceleration.

### 4.4 Observation Space

The agent receives a 20-dimensional observation vector at each step:

- current speed (km/h)
- lateral displacement from the racing line (m)
- heading error with respect to the track direction (rad)
- 6 look-ahead curvature samples along the racing line
- previous steering and throttle commands

The look-ahead curvature samples are the same signal shared by the de-oscillation mechanism described in §4.6, enabling the agent to anticipate corners before they require a steering correction.

### 4.5 Reward Function

The reward at each step is a weighted sum of six terms:

$$r = 1.0\,\Delta s - 2.0\,\text{lateral}^2 - 0.5\,\text{heading}^2 + 0.05\,v - 0.05\left(|\Delta\text{steer}| + |\Delta\text{throttle}|\right) + 50$$

plus the one-shot off-track penalty $-5.0$ on termination.

#### Progress

$$+1.0 \cdot \Delta s$$

The primary learning signal. It rewards advancement along the spline parameterization of the racing line, giving the agent a continuous gradient toward lap completion regardless of speed or trajectory quality.

#### Quadratic Lateral Penalty

$$-2.0 \cdot \text{lateral}^2$$

The single most influential term. It penalizes the squared distance from the racing line, acting as an invisible "glue" that binds the car to the optimal trajectory. The quadratic form is deliberate: deviations grow super-linearly, so a 2 m error is penalized four times more than a 1 m error, creating a strong restoring force near the line while tolerating minor fluctuations. Every ablation that removed or weakened this term produced agents that drove fast but drifted 2–3 m off the racing line.

#### Heading Penalty

$$-0.5 \cdot \text{heading}^2$$

Penalizes the angular misalignment between the car's longitudinal axis and the local track direction. This is distinct from the lateral penalty: an agent can be momentarily close to the racing line but pointing toward the wall (e.g., in corner entry with excessive understeer). The heading term forces the agent to track the continuously rotating track direction throughout the corner, which in practice means initiating the steering input *before* lateral error accumulates.

#### Speed Bonus

$$+0.05 \cdot v$$

A small continuous incentive to maintain speed. Without it, the agent's dominant strategy for minimizing lateral and heading error is to drive slowly, which is technically correct but not the intended behavior.

#### Jerk Penalty

$$-0.05 \cdot \left(|\Delta\text{steer}| + |\Delta\text{throttle}|\right)$$

Penalizes the $L_1$ norm of control variation between consecutive steps. This discourages the left-right steering oscillation that SAC tends to produce early in training when entropy is high, and promotes smooth, physically plausible command sequences.

#### Off-Track Penalty

$$-5.0$$

Applied once when an episode-ending off-track event is detected. The weight was reduced from an initial value of $-10.0$ after observing that a large terminal penalty dominated the critic loss and pushed the policy into a degenerate "stand still = never fail" attractor, where the agent learned to avoid all movement to minimize the probability of termination.

#### Lap Bonus

$$+50$$

A large, non-terminal reward triggered when the spline coordinate crosses from $>0.9$ to $<0.1$, indicating lap completion. Crucially, this does **not** terminate the episode: a competent agent continues driving and can chain multiple laps within a single 300 s episode, receiving the bonus at each crossing.

**Our modifications (the de-oscillation mechanism).** The final configuration (branch
`baseline-deoscillate`) adds two coupled components that share one signal — the look-ahead curvature
of the racing line:

1. **Corner-aware steering rate-limit (actuator side).** A uniform rate-limit cannot win on Monza:
   the rate that kills the straight-line saw (~2.5 /s) is too slow for the 90° first chicane, and the
   rate that takes the chicane (~3.5+/s) lets the oscillation back. So the limit is *scaled by
   curvature*:
   `rate = clamp_straight + (clamp_corner − clamp_straight)·min(1, curv/clamp_curv_full)`, with
   `clamp_straight=2.5`, `clamp_corner=4.0`, `clamp_curv_full=0.015 /m`, looking **30 m ahead** (so
   the wheel is ready before turn-in) and held loose **15 m past corner exit** for exit reactivity.
   The applied (clamped) steer — not the raw policy output — is what feeds vJoy, the observation and
   the reward's `Δsteer`. The rate-limiter itself is a pure function modelling a real actuator's
   finite slew rate, which low-pass-filters the left-right saw into its smooth average **without
   touching the objective**.
2. **Curvature-gated steering penalty (reward side).** A new term `−0.5·|Δsteer|·gate`, with
   `gate = max(0, 1 − curv/clamp_curv_full)`, is active at full strength on straights and fades to
   zero in corners. This is exactly the *per-sector gating* that the ungated L1 steering penalty (the
   `steersmooth` family) lacked — and it is what stops the penalty from fighting the chicane.

We also added an **anti-stop** term (`−w_low·max(0, threshold − speed)`, `w_low=0.2`, threshold
20 km/h) that only grows *below* a low speed, so legitimate slow corners (~40–50 km/h) are untouched
and only genuine stalling is penalized.

**Observation, termination and the step loop.** The 20-D observation is described in §3. An episode
terminates on the first of: lateral norm > 2 (instant violent off-track), tyres-out ≥ 3 for > 1 s,
speed < 1 km/h for > 3 s (stopped), no spline progress for 10 s, or a 300 s safety cap. Lap
completion is **not** a termination — it is detected by the spline crossing from > 0.9 to < 0.1.
Each `step()` clips the policy action, applies the corner-aware clamp, writes to vJoy, busy-waits for
the next physics tick, computes lateral/heading/Δprogress and reward, builds the next observation,
and paces the loop to 20 Hz.

**Training tooling.** `train.py` supports `--warm-start` (initialize weights from an existing SAC
checkpoint for fine-tuning) and a **fixed `--ent-coef`** that disables SAC's auto-entropy — necessary
because on warm-start the auto-entropy re-inflates exploration and destroys the loaded policy (the
baseline converges to `ent_coef ≈ 0.277`, which the fine-tunes then fix). Almost the entire stack is
unit-tested offline (geometry, observation, reward gating, termination, rate-limiter, shared-memory
structs, file-IPC, logging) because every non-`env` module is a pure, side-effect-free adapter.

## 5. Results and Discussion

The reference is the **v3 baseline** (~1.5 M steps, `ent_coef` auto → 0.277). **It is still unbeaten on
raw lap count, but oscillates badly.** The de-oscillation work (family 6, `train_monza_smooth`)
fine-tunes from it and trades raw laps for **smoothness and reliability**.

**Table 1**: Quantitative results (training metrics unless noted)

| Model | Total laps | Steering activity `\|steer\|` | Mean lateral error | Stopped % (train) |
| :--- | :---: | :---: | :---: | :---: |
| v3 Baseline | **466** | 0.59 | < 1 m (on the line) | 26% |
| Our Final Model (`smooth`) | 22 | **0.15** | < 1 m | 78% |

**Why the baseline wins on laps but loses on smoothness.** The 466-vs-22 lap gap is almost entirely
**experience**: the baseline trained for ~1.5 M steps, the fine-tune for ~144 k. The baseline's
reward is simple and effective, but its wheel sign-flips ~50% of the time. Our final model deliberately 
sacrifices raw lap throughput to **eliminate the oscillation**: `|steer|` drops from 0.59 to 0.15, and 
in **deterministic evaluation it completes laps every single time, with no off-tracks and no stops**, 
even though its *stochastic-training* stopped% reads 78%. 
This train/eval distinction is essential: training is exploratory and noisy, evaluation is
the policy actually exploited, and our claims about reliability are measured in eval.

**Where models are weak.** The chronic failure mode of every fine-tune is the **"stopped attractor"**:
any reward change to the baseline landscape shifts the local optimum toward standing still, because a
stationary car cannot go off-track. The anti-stop term mitigates this but is intensity-sensitive —
`w_low=0.2` is insufficient on long runs (the follow-up `train_monza_long` peaked at 19 laps in 72
episodes, then relapsed to ~99% stopped), while combined values above ~0.35 destabilize training.
The hardest *track sector* is consistently the first chicane: it is exactly where a uniform clamp or
an ungated steering penalty breaks the car, which is what motivated the curvature-gated design.

**What the experiment families taught us (qualitative).** Across ~54 runs in six families:
removing the quadratic lateral term (boundary, kinetic, "ab" families) always gave fast cars at
2–3 m off-line; ungated smoothness penalties (L1 `steersmooth`, CAPS action-smoothness, slip-angle
penalties) always fought the corners and could not complete the chicane; warm-starting from the
baseline consistently beat training from scratch (which converged slowly to a worse plateau); and
changing several parameters at once was reliably fatal (the `long` crash changed three knobs
together, whereas `smooth` changed one component at a time and succeeded). The single mechanism that
worked was separating the two problems the uniform clamp conflated — and gating both the actuator and
the reward on the same curvature signal.

## 6. Conclusion and Limitations

We trained a Reinforcement Learning agent to follow the racing line in Assetto Corsa at Monza and 
addressed the qualitative smoothness criterion. A corner-aware steering-rate limit, combined with
a curvature-gated steering penalty, reduced steering oscillations by approximately 75% (|steer| from
0.59 to 0.15), while preserving reliable lap completion during deterministic evaluation. 
More broadly, our results suggest a design rule: smoothness regularization should be curvature-aware,
and both actuator limits and reward shaping should be governed by the same contextual signal.

**Current limitations.**
- **Sample efficiency / lap throughput.** The smooth model trains for far fewer steps than the
  baseline and does not yet match its raw lap count; the de-oscillation reward also reshapes the
  landscape toward the stopped attractor, which we contain but do not eliminate.
- **Anti-stop fragility on long runs.** `w_low=0.2` is enough for short runs but relapses on long
  training; the correct intensity is run-dependent and not yet adaptive.
- **Environment fragility.** Because the agent drives the real game, the stack is sensitive to AC's
  internal Python build (which forced the move to a 4-byte file-IPC), requires a one-time manual setup
  (vJoy assignment, plugin install), and cannot be parallelized — only one environment exists.

**Future experiments.** If we had more time, we could: 
1. run a long training from the `smooth` configuration (checkpoint `sac_140000`) without changing anything,
to consolidate and grow the 22 laps; 
2. make the anti-stop weight schedule itself rather than be a fixed constant, to defeat
the stopped attractor on long runs; 
3. test cross-track generalization more systematically on an unseen evaluation track.

## 7. Additional Information

### 7.1 Contribution Breakdown
- **Ignazio Coco**: Environment & game interface — shared-memory structs, vJoy controller,
  file-IPC reset bridge (`ac_plugin/`), `ai_line` loader and X-Z geometry.
- **Alessandro Passanisi**: Learning & reward — SAC training pipeline (`train.py`), reward design and
  the curvature-gated steering penalty, observation engineering, termination logic.
- **Yuri Filistad**: Experiments & analysis — the ~54-run experimental campaign across the six
  families, logging/plotting tooling, deterministic evaluation, and the corner-aware clamp tuning.

### 7.2 Use of Artificial Intelligence
AI coding assistants were used for **boilerplate and supporting code** (Parquet logging, plotting
scripts, ctypes struct definitions, unit-test scaffolding), for **debugging** the AC IPC fragility
(the UDP → shared-memory → file-IPC migration), and for **documentation** drafting. The reward
design, the corner-aware de-oscillation mechanism, the experimental methodology and the
responsibility for all results are ours.
