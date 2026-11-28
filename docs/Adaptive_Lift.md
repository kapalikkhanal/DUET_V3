# Adaptive_Lift

Train this task with the Python environment used for the DUET_V3 47k run. Run this command from the repository root.

```sh
bash scripts/train_adaptive_lift.sh train
```

The launcher runs unit tests, two short GPU terrain trials and a two-update scratch PPO trial before starting a fresh 50,001-update run with 4,096 environments on one GPU. Use `check` to stop after preflight. Set `PYTHON` to the training interpreter or `NUM_ENVS` to the workstation batch size if needed. Both trainer invocations use fresh weights, observation statistics, optimizer and curriculum counters. The smoke checkpoint is kept in its own run directory. The launcher prints the compact console log path. TensorBoard logs and checkpoints go under `logs/rsl_rl/Adaptive_Lift/`.

## Command and terrain

The two clearance channels are the requested swing apex of the lowest sole point above its local load-bearing floor, in left/right order. A 7 cm command on 3 cm of litter asks for about 4 cm of free space above the litter. Each physical foot has seven collision capsules. The reward takes the minimum of their fourteen downward support points after referencing each point to its local bed. At a discontinuous tile seam this is a sampled approximation. The peak is recorded over the airborne portion of a completed moving swing with at least 60 ms of flight. The target is latched at takeoff so a mid-swing command change applies to the next swing.

Commands are uniform from 4 to 11 cm on every substrate. They resample every 2–5 s while standing or walking, with a 3.5 cm/s limit between draws. Three quarters of draws request the same height for both feet. The other quarter samples each side independently. The actor observes just the command pair scaled to [−1, 1] in each of six history frames. It retains the original lean-task channels, including arm position and velocity. The critic has 286 inputs. The 13 outputs control twelve leg joints and waist yaw. The existing external arm action runs its own randomized trajectories.

| Spawn terrain | First PPO update | Bed and layer |
| --- | ---: | --- |
| Rigid | 0 | Bare floor |
| Compacted | 1,000 | Bed raised 1.2 cm with 0.48–0.72 cm loose cover |
| Loose | 2,500 | Level bed with 2.4–3.6 cm loose cover |
| Mixed | 4,000 | Adjacent patches of all three materials |

Each reset samples uniformly from the open columns. The substrate follows the actual foot position when the robot crosses a boundary. These are analytical approximations for training. Layer depths keep the full command range above the loose surface. Pointwise lateral resistance acts only inside litter, with a 45 N per-foot resultant cap. The positive terrain contact time constants are 20–30 ms with unit damping. Foot friction varies from 0.50 to 1.05. The inherited randomization includes motor delay and gain, encoder bias, pushes, hand and torso payloads and center-of-mass shifts.

## Reward and curriculum choices

| Objective | Implementation |
| --- | --- |
| Walking, turning and stopping | Keep linear and yaw tracking, backward speed, alternating gait, lateral foot spacing and the settle window. Align the air-time target with the 0.33 s planned swing. Keep the fall, hard self-collision and joint-limit costs. |
| Lift and plowing | Track a single support-relative apex on completed swings. Penalize early horizontal travel below the latched target and buried travel through litter, including before liftoff. A standing robot receives no swing credit. |
| Landing and slip | Penalize excess sole tilt at verified touchdown and vertical load above 320 N during the first 40 ms of loading after flight. Increase contact slip weight from −0.25 to −0.35. |
| Motion and power | Keep first- and second-difference action costs, raw action bounds and joint velocity damping. Restrict the joint acceleration cost to policy-controlled joints. Absolute mechanical power sums the twelve leg joints, at −0.08 per 100 W with a 400 W cap. |
| Squat and lean | Keep pelvis-height tracking, squat-aware posture, support and standing balance. The squat floor reaches 0.22 m. Lean opens at update 6,000 and reaches 1.25 rad at 14,000, with a final stoop band of 0.32–0.62 m. The lean-aware anchor and fall logic allow the pelvis to move back during a stoop. |

Velocity, arm, squat, payload and push settings use the DUET poultry scratch configuration. Lift, plow, landing and power weights are zero through update 800 and ramp to full value at 3,000 to give balance and velocity tracking a head start. The runner retains scratch PPO exploration, live observation normalization and the DUET mirror loss, including left/right swaps for every clearance pair.

## Reading the run

Preflight exercises the real simulator, normal trainer, command transitions and all four substrates. The randomized-action trials check mechanics and finite values. Stability of the learned gait requires trained checkpoint evaluation. Compare falls, velocity/yaw tracking, and slips with the fixed 47k reference under matched motions, seeds, terrain and arm/payload conditions. Check squat depth and lean separately. Sweep 4, 6, 7.5, 9 and 11 cm and test transitions while walking. Score each complete swing against its latched command, and compare leg power at matched achieved speed.

TensorBoard includes existing fall, velocity, arm, lean and contact-slide metrics along with leg power in watts. The ratio of `swing_error_sum_m` to `completed_swings` over the same logging window gives an episode-normalized estimate of absolute apex error. Both metrics are per-step episode averages. Use per-swing measurements for comparisons between checkpoints.
