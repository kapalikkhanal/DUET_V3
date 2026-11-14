# Egg-tray transport: an arm-only RL stabiliser on frozen DUET

The DUET v2 locomotion policy `poultry_v3_squat_91500` walks the G1 on poultry litter. This adds a second, separately trained policy. It moves only the ten arm joints, making small corrections around a carrying pose so a tray of loose eggs stays steady while DUET walks.

DUET is not retrained, fine-tuned or modified. Its checkpoint, ONNX export, deploy contract, deploy configs and C++ controller are untouched.

```
arm_target = carry_reference + RL_residual     (10 arm joints, bounded)
leg/waist targets = DUET 91500 (unchanged)      (13 joints, frozen)
```

## Contents

- [Setup](#setup)
- [Commands](#commands)
- [What was built](#what-was-built)
- [Verification](#verification)
- [Results](#results)
- [Limitations](#limitations)
- [Files](#files)

## Setup

**Python.** Use an interpreter with mjlab 1.2.0, mujoco-warp 3.5.0, MuJoCo 3.5.0 and rsl-rl-lib 5.0.1.

- MuJoCo 3.6 with mujoco-warp 3.5 crashes in `put_data` once the model exceeds 60 DOF. The egg-tray scene has 71.
- On the development machine, a venv layered on the `unitree_sim_env` conda env was used:

```bash
~/miniconda3/envs/unitree_sim_env/bin/python -m venv --system-site-packages ~/venvs/duet_eggtray
~/venvs/duet_eggtray/bin/pip install --no-deps rsl-rl-lib==5.0.1 mujoco==3.5.0
```

All commands below run from the repository root with `PYTHONPATH=.`, using that interpreter as `python`.

**Poultry-floor assets.** These are the existing builder's output, with the same seed and preset:

```bash
python scripts/poultry_floor/make_poultry_floor.py --out logs/eggtray/poultry_floor --no-verify
```

The tasks look for floor assets in this order:

1. `$DUET_POULTRY_FLOOR_ASSETS`
2. `logs/eggtray/poultry_floor/meshes/poultry_floor`
3. the documented default `~/Mujoco_kapalikkhanal/...`

## Commands

| Step | Command |
|---|---|
| Phase A: verify the env with fixed arms | `python scripts/eggtray/phase_a_verify.py --terrain flat --n 32` (or `--terrain poultry`) |
| Re-measure the carry feed-forward | `python scripts/eggtray/phase_a_verify.py --calibrate --n 32` |
| Phase B: small run, flat | `bash scripts/eggtray/train.sh Unitree-G1-23Dof-EggTray-Flat phaseB_flat_256 256 150 1` |
| Phase C: main run, litter | `bash scripts/eggtray/train.sh Unitree-G1-23Dof-EggTray-Poultry phaseC_poultry_1024 1024 500 1` |
| Evaluate three modes | `python scripts/eggtray/evaluate.py --terrain poultry --contact built_up --modes fixed feedback rl --checkpoint <model.pt> --seeds 0 1 2 --n 64 --out logs/eggtray/eval/built_up` |
| Out-of-distribution floor | the same with `--contact fresh_deep` |
| Twelve eggs | the same with `--eggs 12` |
| Side-by-side video | `python scripts/eggtray/play.py --terrain poultry --compare fixed rl --checkpoint <model.pt> --out logs/eggtray/videos/compare.mp4` (add `--closeup` for the tray view) |

Notes on the commands:

- `train.sh` takes `TASK RUN_NAME NUM_ENVS ITERATIONS SEED`.
- It wraps `scripts/train.py`, after checking the DUET checkpoint's SHA-256 and the floor assets.
- Output goes through `scripts/tools/compact_log.py`.
- Runs go to `logs/rsl_rl/EggTray_G1_23dof/<date>_<run>/`. Each checkpoint has an `arm_policy.onnx` beside it, carrying the observation layout, joint order, bounds, carry reference and DUET hash as metadata.

Registered tasks:

| Task | Use |
|---|---|
| `Unitree-G1-23Dof-EggTray-Flat` | Debugging |
| `Unitree-G1-23Dof-EggTray-Poultry` | Generated broiler-house floor, `built_up` contact, with the 0-6 cm loose-litter drag layer |
| `Unitree-G1-23Dof-EggTray-Poultry-12Egg` | Evaluation with 12 eggs |

## What was built

### Frozen DUET (`src/tasks/eggtray/frozen_duet.py`)

DUET runs inside the environment as an action term with **zero** agent action dimensions.

**Inputs.**

- Each control step it reads the `duet` observation group, a copy of the PoultrySurface actor group: the deployed 71-D layout in its original order.
- Its last-action input points at DUET's own previous output rather than the agent's actions.

**Network and action processing.**

- It runs the checkpoint's normaliser and MLP.
- Raw actions are clipped at ±30, as in PoultrySurface training.
- Targets are applied with DUET's per-joint scale, default-pose offset and encoder-bias correction.

**Checks at build time.**

- The checkpoint's SHA-256 must be `0e1e2cbf…`.
- Joint order, action scale and offsets must match `exported/poultry_v3_squat_91500/deploy_contract.json`.
- Sizes must be 71 → 13.

The network has no trainable parameters and is not registered with the optimizer. The runner checks this at start-up and refuses to train otherwise.

### Tray and eggs (`tray_model.py`, `carry.py`)

The tray and eggs are generated procedurally in MuJoCo, using convex collision geometry only.

| | Tray | Egg |
|---|---|---|
| Shape | Base plate; one tapered cup per egg (four plates inclined 60°, 30 mm deep, 55 mm pitch); rim; two side handles | Capsule for collision (43 mm wide, 58 mm long), ellipsoid for display |
| Mass | 0.22 kg | 60 g, with ellipsoid inertia |
| Joints | Free body, held by the hands | Free body, one per cup |
| Sensors | Accelerometer and gyro (tray IMU) | — |

**How eggs sit.** Each egg rests on the four cup walls. Nothing attaches it to the tray, and it can rattle, ride up a wall or leave its cup.

**Contacts.** Tray and eggs use their own contact bit, so eggs touch the tray and each other and nothing else.

**Plate thickness.** Collision plates are 8 mm thick, built outward from the cup surface so the cup shape is unchanged. With 3 mm plates, 20-80 N impacts pushed eggs through.

**Grasp.** The tray is held between the palms by a **soft weld at each hand**: timeconst 0.01 s, the stiffest recommended at a 5 ms step.

- Two welds on one body form a closed chain. MuJoCo's soft, regularised solver resolves it.
- The arm PD loops are compliant, so inconsistent arm targets give a bounded squeeze rather than a solver fight.
- The weld stretch is reported and penalised. It stays below 0.2 mm.

**Why two hands.** Right-hand-only carrying was tested and rejected. The 23-DOF G1 has no wrist pitch or yaw, so one 5-DOF arm cannot set the tray's six DOF, and the offset load puts about 0.8 N m on the grasp. In the test the tray rotated in the grasp and the eggs spilled.

**Carry pose.** The pose is a joint dictionary (`DEFAULT_CARRY_POSE`) and can be replaced by a GR00T pose. Everything else is derived from it by CPU forward kinematics: tray placement, weld frames, and a least-squares map from tray twist to arm joints.

- On the G1, elbow = 0 is the L shape. The default pose (shoulder pitch −0.10, elbow 0.10) therefore has the upper arm hanging, the elbow at about 90° and the palms at about elbow height (≈0.89 m).
- Arm gains are soft (14.25 N m/rad), so the loaded arms sag about 0.14 rad and tip the tray 19° nose-down.
- `CARRY_FEEDFORWARD` corrects this. It is measured in two stages: gravity sag, then levelling the tray in the world while DUET stands, which absorbs DUET's forward lean under the load. Standing tilt ends at 1.0°.

### Arm residual action (`mdp/actions.py`)

The arm policy has 10 outputs. Its target is

```
target = clamp(reference + clamp(0.05·a, ±0.15 rad), soft joint limits)
```

with the change in target rate-limited to 2 rad/s. Torque is limited by the actuator (25 N m).

- The constructor rejects any joint that is not one of the ten arm joints.
- The reference can be randomised per episode by a small, tray-consistent offset, so the policy learns around a family of carry poses.

### Observations

| Group | Terms | Size |
|---|---|---|
| Actor (deployable) | Arm encoder angles relative to the reference, arm joint velocities, robot IMU (angular velocity, gravity direction), walking and height commands, DUET's gait clock, tray IMU (attitude, gyro, accelerometer), previous arm action, carry-reference offset | 61 |
| Critic (privileged) | Actor terms, plus robot and tray linear velocity, egg offsets and velocities in their cups, weld stretch, foot contacts, DUET's action, litter depth | 121 |

The actor needs a tray-mounted IMU on hardware.

### Rewards (`mdp/rewards.py`)

The manager scales rewards by dt; the per-second values below are effective.

| Term | Weight | Purpose |
|---|---|---|
| `tray_acc` | +2.0 · exp(−\|a\|²/3²) | Primary: low gravity-compensated tray acceleration (50 Hz finite difference) |
| `tray_tilt` | −20 · (tilt − 3°)₊² | Tilt beyond a deadband only; a small deliberate tilt is free |
| `tray_ang_vel` | −0.2 | Roll and pitch rate |
| `egg_lost` | −500 (−10 per egg) | An egg left its cup; the episode also ends in training |
| `egg_motion` | −2 | Egg speed relative to its cup (rattle) |
| `egg_impact` | −0.5 · (F − 1.5 N)₊ | Net egg contact force above 2.5× its weight |
| `action_rate`, `arm_joint_acc`, `residual`, `grasp_strain` | small | Smoothness, staying near the pose, no squeeze |
| `is_terminated` | −200 | Fall, lost egg, or tray past 40° |

**Egg contact force** is computed every physics substep from the egg's momentum change, `F = m(dv/dt − g)`. MuJoCo Warp's contact sensor was not used: on resting eggs its per-substep net force read a 4 N median against the true ≈0.6 N (checked on the CPU with `mj_contactForce`), whatever the contact stiffness or solver budget.

### Training randomisation

| Item | Range |
|---|---|
| Tray mass | 0.15-0.30 kg |
| Egg mass | 50-70 g |
| Tray centre-of-mass offset | ±1 cm |
| Cup friction | 0.35-0.9 |
| Loose-litter depth | 0-6 cm |
| Walking commands | x −0.4..0.6 m/s, y ±0.3 m/s, yaw ±0.6 rad/s; 20% standing |
| Pelvis height command | 0.70-0.78 m |
| Pushes | Mild, every 6-10 s |

DUET's own observation noise and its robot randomisation are also kept: foot friction, encoder bias, torso and hand mass.

### Evaluation modes (`scripts/eggtray/evaluate.py`)

| Mode | Arms |
|---|---|
| Baseline A, `fixed` | DUET 91500 with the calibrated carry pose held |
| Baseline B, `feedback` | Conventional tray levelling from the tray IMU: leaky integral on tilt plus rate damping, and a "sky-hook" vertical loop on the accelerometer, mapped to joints through the same tray-twist map and bounds |
| Proposed, `rl` | DUET 91500 plus the trained residual policy |

**Identical conditions.**

- Every mode uses the same scripted 31 s routine: stand, forward 0.4, stop, backward 0.3, turn 0.5, sideways 0.25, forward 0.5, stop.
- For each seed, one environment is built and every mode is run after re-seeding, so spawn, litter, payload and commands are identical.
- Pushes and observation noise are off.
- Results are reported as the mean ± standard deviation across seeds, with pooled percentiles in the JSON.

## Verification

Run with `scripts/eggtray/phase_a_verify.py`.

| Check | Result |
|---|---|
| DUET checkpoint SHA-256 before and after every run | Unchanged |
| In-env DUET vs exported ONNX on live observations | Max \|Δ\| 3e-6 |
| Leg targets vs DUET output, including under ±10 random arm actions | Exact (0.0) |
| Arm targets within reference ± 0.15 rad, soft limits and rate limit | Always |
| Agent action space | `arm_residual` (10), DUET (0), litter (0), state (0) |
| Parameters | 202,765 DUET parameters frozen; 245,909 arm-policy parameters trained |
| NaNs | None |
| Contacts, constraint rows | Below allocation (≤57 contacts and ≤312 rows per world on litter) |

## Results

**Phase C, the main run** (poultry floor, 1024 envs, 500 iterations, about 40 min). Reward plateaued from about iteration 250.

**Checkpoint selection.** Checkpoints 300, 400 and 499 were evaluated over 3 seeds × 64 episodes on `built_up`:

| Checkpoint | Tray acc. RMS | Tilt RMS | DUET lin. / yaw tracking error |
|---|---|---|---|
| 300 | 1.79 m/s² | 2.58° | — |
| 400 (chosen) | 1.50 m/s² | 2.82° | 0.129 m/s / 0.108 rad/s |
| 499 | 1.38 m/s² | 3.05° | 0.133 m/s / 0.112 rad/s |

Checkpoint 400 is used below because locomotion accuracy may not degrade. Evaluation settings for every table: mean ± std across 3 seeds × 64 episodes, the 31 s routine, random 0-6 cm loose litter.

### Without pushes

Change is relative to fixed arms.

**`built_up` litter, 6 eggs**

| Metric | Fixed | Feedback | Arm RL | RL vs fixed |
|---|---|---|---|---|
| Episodes with all eggs kept | 0.93 ± 0.05 | 0.90 ± 0.02 | **0.99 ± 0.02** | — |
| Robot falls | 0.016 | 0.005 | 0.010 | — |
| Tray acc. RMS (m/s²) | 2.85 | 2.89 | **1.50** | −47% |
| Tray acc. RMS, walking (m/s²) | 3.12 | 3.14 | **1.62** | −48% |
| Tray tilt RMS (°) | 3.42 | **1.99** | 2.83 | −17% |
| Tray tilt max (°) | 9.4 | **6.2** | 9.5 | — |
| Tray jerk RMS (m/s³) | 56.8 | 65.8 | **34.0** | −40% |
| Peak egg contact force (N) | 3.52 | 4.16 | **2.47** | −30% |
| DUET lin. / yaw tracking error | 0.122 / 0.097 | 0.122 / 0.097 | 0.130 / 0.108 | +6% / +12% |
| Transport distance (m) | 6.99 | 6.93 | 6.96 | — |

**`fresh_deep` litter, 6 eggs** (out of distribution)

| Metric | Fixed | Feedback | Arm RL | RL vs fixed |
|---|---|---|---|---|
| Episodes with all eggs kept | 0.89 | 0.86 | **0.93** | — |
| Robot falls | 0.021 ± 0.009 | 0.047 ± 0.031 | 0.052 ± 0.036 | see note |
| Tray acc. RMS (m/s²) | 2.96 | 3.05 | **1.72** | −42% |
| Tray tilt RMS (°) | 3.52 | **2.28** | 3.19 | −9% |
| Tray jerk RMS (m/s³) | 57.9 | 69.0 | **37.7** | −35% |
| Peak egg contact force (N) | 4.06 | 4.67 | **2.92** | −28% |
| DUET lin. / yaw tracking error | 0.142 / 0.114 | 0.141 / 0.115 | 0.149 / 0.131 | +5% / +15% |

**`built_up` litter, 12 eggs** (never trained; 6-egg feed-forward)

| Metric | Fixed | Feedback | Arm RL | RL vs fixed |
|---|---|---|---|---|
| Episodes with all eggs kept | 0.85 | 0.87 | **0.97** | — |
| Robot falls | 0.026 | 0.016 | **0.000** | — |
| Tray acc. RMS (m/s²) | 2.93 | 2.95 | **1.59** | −46% |
| Tray tilt RMS (°) | 3.11 | **2.11** | 5.47 | +76% |
| Tray jerk RMS (m/s³) | 55.7 | 63.1 | **34.1** | −39% |
| Peak egg contact force (N) | 4.32 | 5.02 | **3.27** | −24% |

### With moderate pushes

Eight shoves of 0.3-0.5 m/s while standing, walking, turning and stepping sideways, on `built_up` litter with 6 eggs. The old DUET 50k (squat-reach) legs are included for reference; they have the same deploy contract.

| Metric | DUET 50k + fixed | DUET 91500 + fixed | DUET 91500 + arm RL |
|---|---|---|---|
| Episodes with all eggs kept | 0.68 ± 0.04 | 0.91 ± 0.05 | **0.95 ± 0.05** |
| Robot falls | 0.094 | 0.021 | 0.026 |
| Tray acc. RMS (m/s²) | 3.16 | 3.24 | **2.08** (−34% vs 91500 + fixed) |
| Tray tilt RMS / max (°) | 3.8 / 15.9 | 3.6 / 11.1 | **3.0 / 10.9** |
| Eggs pushed above 5 N per episode | 1.26 | 1.15 | **0.31** |
| DUET lin. / yaw tracking error | 0.228 / 0.152 | 0.168 / 0.111 | 0.174 / 0.122 |

Videos are in `logs/eggtray/videos/`, both side by side with 50k + fixed on the left and 91500 + arm RL on the right:

- `old50k_fixed_vs_new_rl_pushes.mp4`
- `old50k_fixed_vs_new_rl_pushes_closeup.mp4`

### Reading the results

- **Main target.** Across all three push-free conditions the arm policy cuts tray vibration by 42-48%, and by 34% under pushes. It does so without fewer runs keeping all their eggs, and with lower peak egg forces.
- **The feedback leveller** reduces tilt more than RL does, but adds vibration, jerk and egg force.
- **Locomotion.**
  - Moving the arms costs DUET a little turning accuracy: yaw-rate error is 12-15% higher, linear error about 6% higher.
  - On `fresh_deep`, falls were 5.2% with arm RL against 2.1% with fixed arms. The seed-to-seed spread is ±3.6%, so this is not resolved and needs more seeds.
- **Twelve eggs.** The policy keeps eggs and damps vibration, but holds the tray about 5° off level. The 4-row tray is heavier and offset, and the feed-forward was calibrated for 6 eggs. Re-calibrating for 12 eggs, or training with them, is the fix.
- **Two rare events, both only with arm RL under pushes:**
  - DUET output above the deploy guard's ±15 limit: 0.07 steps per episode on average.
  - One seed with weld stretch of several millimetres.

  Both need checking before any hardware use.

**Phase B** (flat, 256 envs, 150 iterations, about 5 min). One seed, 32 episodes, same routine:

| Metric | Fixed arms | Arm RL | Change |
|---|---|---|---|
| Tray acceleration RMS | 2.66 m/s² | 1.96 m/s² | −26% |
| Tray acceleration RMS, walking | 2.95 m/s² | 2.17 m/s² | −27% |
| Tray acceleration peak | 7.6 m/s² | 6.2 m/s² | −19% |
| Tray tilt RMS / max | 3.3° / 5.9° | 2.4° / 4.3° | −28% / −27% |
| Tray jerk RMS | 55.5 m/s³ | 44.3 m/s³ | −20% |
| Egg impulse above 2× weight | 0.13 N s | 0.08 N s | −38% |
| Eggs kept / falls | 100% / 0% | 100% / 0% | — |
| DUET linear-velocity tracking error | 0.111 m/s | 0.114 m/s | — |

The corrections are small: about 0.02 rad per joint on average. They are hard to see by eye in the videos even though the tray numbers change.

## Limitations

- **Egg damage is not modelled.**
  - Reported egg loads are net contact forces from the egg's momentum change over 5 ms physics steps, and impulses above twice the egg's weight.
  - Shorter, sharper impact peaks are not resolved.
  - No load-to-crack relationship has been calibrated, so these are damage-risk proxies, not breakage predictions.
  - For scale, the simulated peaks (2-4 N) are an order of magnitude below the ~30-40 N quasi-static strength usually quoted for chicken eggshells.
- **Contact model.** Egg-tray contact stiffness, friction and the thick collision plates are modelling choices, not measurements of a real tray.
- **Grasp.** Each hand is a soft weld at the palm, so the hands cannot slip or drop the tray. Real Dex3 grasp forces and slip are not modelled.
- **Payload.** Tested at 0.58 kg (6 eggs, plus tray) and 0.94 kg (12 eggs), held by both hands. The arm gains are soft, so a heavier load needs a new feed-forward calibration.
- **Hardware.** The policy needs a tray IMU that the robot does not yet have. The arm targets would go to `rt/arm_targets` through the existing interface, but nothing has been deployed or tested on the robot.
- **Locomotion.** DUET's own falls on litter are unchanged by the arm policy. See the results for the measured tracking errors and fall rates.

## Files

| Path | Contents |
|---|---|
| `src/tasks/eggtray/` | Task: `frozen_duet.py`, `tray_model.py`, `carry.py`, `mdp/`, `config/g1_23dof/` (env, PPO, registration), `rl/runner.py` |
| `src/tasks/common/poultry_farm_terrain.py` | The poultry floor terrain, moved from `eval_poultry_floor.py` (which now imports it; behaviour unchanged) |
| `scripts/eggtray/` | `train.py`, `train.sh`, `phase_a_verify.py`, `evaluate.py`, `play.py`, `common.py` |
| `logs/eggtray/` (git-ignored) | Floor assets, Phase A reports, training logs, evaluations, videos |
