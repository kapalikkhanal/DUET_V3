# Adaptive_Lift

Train this task with the Python environment used for the DUET_V3 47k run. Run this command from the repository root.

```sh
bash scripts/train_adaptive_lift.sh train
```

The launcher runs unit tests, a GPU preflight and a two-update scratch PPO trial before starting a fresh 50,001-update run with 4,096 environments on one GPU. The GPU preflight ray-casts every litter cell of the compiled floor against the analytic map, then runs two short randomized-action trials. Use `check` to stop after preflight. Set `PYTHON` to the training interpreter or `NUM_ENVS` to the workstation batch size if needed. Both trainer invocations use fresh weights, observation statistics, optimizer and curriculum counters. The smoke checkpoint is kept in its own run directory. The launcher prints the compact console log path. TensorBoard logs and checkpoints go under `logs/rsl_rl/Adaptive_Lift/`.

## Command

The two clearance channels are the requested swing apex of the lowest sole point above its local load-bearing floor, in left/right order. A 7 cm command on 3 cm of litter asks for about 4 cm of free space above the litter. Each physical foot has seven collision capsules. The reward takes the minimum of their fourteen downward support points after referencing each point to its local bed. At a discontinuous tile seam this is a sampled approximation. The peak is recorded over the airborne portion of a completed moving swing with at least 60 ms of flight. The target is latched at takeoff so a mid-swing command change applies to the next swing.

Commands are uniform from 4 to 11 cm on every substrate. They resample every 2–5 s while standing or walking, with a 3.5 cm/s limit between draws. Three quarters of draws request the same height for both feet. The other quarter samples each side independently. The actor observes the command pair scaled to [−1, 1] in each of six history frames, after the original lean-task channels (444 inputs). The critic also observes the litter state and the applied pull (297 inputs, see below). The 13 outputs control twelve leg joints and waist yaw. The existing external arm action runs its own randomized trajectories.

For deployment, `clearance_commands` in deploy.yaml needs `minimum: 0.04, maximum: 0.11`. The controller's command filter should not move faster than the 3.5 cm/s training rate.

## Terrain

Terrain, spawn, friction, pushes, payloads and the terrain-level curriculum are the 47k lean task's: 20 columns × 10 rows of 8 m tiles. The 47k flat (15%) and soft_floor (60%) columns become litter and split-level tiles. The heightfield columns (pyramid slope, inverted slope, random rough, waves: 25%) are unchanged, and the lift, plow and landing terms rest on them as in the 47k task.

| Columns | Tile | Bed and layer |
| --- | --- | --- |
| 3 | Rigid | Bare hard floor; randomized foot friction 0.3–1.6 applies |
| 4 | Compacted | Bed raised 1.2 cm, 0.48–0.72 cm loose cover |
| 4 | Loose | Level bed, loose layer drawn per episode from 2 cm to the curriculum's deepest |
| 2 | Mixed | 2 m × 4 m patches of all three materials |
| 2 | Split-level | Rigid 0.62 m strips along x, alternately low and raised; step height by terrain row |

The split-level step follows the terrain-level curriculum: 2.0 cm at row 0, rising quadratically to 7.0 cm at row 9 (rows 0–5, where robots start, are 2.0–3.5 cm). Seams run along x, so walking along one puts one foot up, and walking or sidestepping across steps up and down every 0.62 m. Seven of the tile's ten spawn points are on seams, and a robot spawned there is turned to face along the seam (±0.4 rad, either way), so about three quarters of the robots on these tiles start with one foot up, left or right; standing robots must then balance there, since the position anchor holds them in place. Random headings while walking put each foot uphill about half the time. A swing that changes support level (a strip edge, or a compacted bed edge) still counts for touchdown, heel-first and slip, but is not scored for apex height, because its peak above the local bed includes the step.

Litter cells use the 47k soft_floor contact: solref and solimp scale with the row difficulty from hard to soft (about 1 cm average sinkage while standing on the softest rows), with a per-tile friction of 0.3–1.0. The deepest loose layer is 3 cm through update 1,500 and ramps to 8 cm at 4,000. Litter drag is the 47k model: 500–1500 N s/m², with a 150 N per-foot cap, acting only inside the layer and without vertical support. The substrate follows the actual foot position across tile boundaries.

The critic observes `litter_truth`: the per-foot mean litter depth and support elevation under the sole, whether each foot is on a litter tile, and the episode's drag strength. The actor stays blind to the terrain. The observation is read-only: observations run after resets and a sensor refresh, and a version that called the swing state machine there consumed touchdowns and takeoffs before the rewards saw them (208 lost against 395 seen in a reset-heavy test). The preflight fails if any observation calls it.

## Pushes and pulls

The 47k velocity pushes remain (a 0.7 m/s kick every 3–5 s). With `PULL_COMPLIANCE = True` (top of the config) there are also sustained, hand-like pulls: horizontal, 0–100 N uniform, held 1–3 s with 0.25 s ramps, on the torso or the pelvis, every 4–10 s, never started while a lean or a squat below 0.55 m is commanded. Their scale is zero until update 2,000 and reaches full at 6,000. While a pull is above 40 N and for 1 s after it, the stand-still terms (position anchor, feet still, base motion, feet square) are off, the standing posture uses the walking tolerances (a step needs hip roll), zero-velocity tracking stops paying for standing still, and the anchor re-latches where the robot ends up. A strong pull is answered by stepping with it and then standing square, as the stock firmware does; pulls under 40 N, far above the reaction of moving arms and payloads, are still resisted, so station keeping while the arms work is unchanged. The upright and centre-of-mass support terms stay on, so stepping is cheaper than leaning into the pull. The critic observes the pull in the pelvis frame and the release flag. Set `PULL_COMPLIANCE = False` to keep the 47k standing behaviour.

For reference, the inherited randomization: foot friction 0.3–1.6 (litter tiles 0.3–1.0), torso COM ±10/6/5 cm, hand payload 0.25–1.75 kg and hand COM ±5 cm, torso payload −0.5 to +2.0 kg, encoder bias ±0.015 rad, motor delay 0–20 ms per episode, leg kp ±10% and kd ±20%, arm gains ±50%, randomized arm trajectories, random stops and stopped backward walkers.

## Reward and curriculum choices

| Objective | Implementation |
| --- | --- |
| Walking, turning and stopping | Keep linear tracking, backward speed, alternating gait, lateral foot spacing and the settle window. Following HOMIE's G1 terms, yaw tracking is exp(−4 err²) at weight 2.0 (was exp(−2 err²) at 1.75, which pays almost fully for small yaw errors; the roll/pitch-rate factor inside the term is unchanged), and a separate lateral term exp(−4 (v_y − v_y,cmd)²) at 1.0 is added. A fifth of moving velocity draws each become a pure in-place turn or a pure sidestep at 35–100% of the current range. Air-time target matches the 0.33 s planned swing. |
| Lift and plowing | Track a single support-relative apex on completed swings. Penalize early horizontal travel below the latched target. Penalize horizontal travel buried in litter, including before liftoff, but only below the commanded clearance: walking at a 4 cm command through 8 cm of litter is obeying the command. A standing robot receives no swing credit. |
| Landing and slip | The 47k→50k landing fine-tune's form and weights, which cut heel-first steps from about 100% to under 10%. `touchdown_tilt` (−2.0) charges sole tilt beyond 1.5° over the last 1.5 cm of descent (above the local bed) and at the landing step. There is no standing or loading-phase tilt term, because a loaded heel sinks on soft litter. `touchdown_load` (−1.2) charges vertical load above the robot's own weight (315 N bare, more with payload; the fine-tune's fixed 320 N charged every payload landing) during the first 40 ms after flight. Contact slip weight −0.35. |
| Motion and power | Keep first- and second-difference action costs, raw action bounds and joint velocity damping. Restrict the joint acceleration cost to policy-controlled joints. Absolute mechanical power sums the twelve leg joints, at −0.08 per 100 W with a 400 W cap. |
| Upright | Unchanged. While walking the orientation cost is quadratic (5° costs about 0.02/s); while standing it is linear beyond 0.5° (5° costs about 0.39/s), against −200 for a fall, so a recovery lean on a step or after a pull is cheap while a lasting tilt is not. |
| Squat and lean | Keep pelvis-height tracking, squat-aware posture, support and standing balance. The squat floor reaches 0.22 m. Lean opens at update 6,000 and reaches 1.25 rad at 14,000, with a final stoop band of 0.32–0.62 m. |

Velocity, arm, squat, payload and push settings use the DUET poultry scratch configuration. Lift, plow, landing and power weights are zero through update 800 and ramp to full value at 3,000 to give balance and velocity tracking a head start. The runner retains scratch PPO exploration, live observation normalization and the DUET mirror loss, including left/right swaps for every clearance and litter pair.

## Reading the run

TensorBoard includes the existing fall, velocity, yaw, arm, lean and contact-slide metrics along with leg power in watts. Per-swing ratios over the same logging window:

- `swing_error_sum_m / level_swings`: mean absolute apex error over swings that stay on one support level.
- `heel_first_landings / completed_swings`: fraction of landings more than 3° toe-up just before contact. The 47k policy is near 100% on flat ground; the 50k landing fine-tune was under 10%.
- `landing_pitch_sum_deg / completed_swings`: mean sole pitch just before contact, negative for heel first.

- `pull_force_n`, `pull_released`: mean applied pull and the fraction of time the stand-still terms are released.

Split-level test (also for the 47k reference; `--task Adaptive_Lift` for this task's checkpoints):

```sh
PYTHONPATH=. python scripts/eval_split_level.py CKPT --task Adaptive_Lift
```

It reports falls, speed tracking, torso tilt, pelvis roll toward the lower foot, uphill-minus-downhill knee bend, stance-foot slip and whether the feet are still on different levels, for 0, 2, 3.5, 5 and 7 cm steps with the left or the right foot up: standing across a seam, walking along it, stopping, turning on it, and walking and sidestepping across the strips. The 47k reference walks along a 7 cm seam at 78% of the commanded speed with level pelvis, but crosses strips at 64% (5 cm) and 46% (7 cm, one fall in 16), sidesteps across them at 24% and 9%, and after stopping it usually steps so both feet share one level.

Stability of the learned gait requires trained checkpoint evaluation. Compare falls, velocity/yaw/lateral tracking and slips with the 47k reference under matched motions, seeds, terrain and arm/payload conditions, including pure turns and sidesteps. Check squat depth and lean separately. Sweep 4, 6, 7.5, 9 and 11 cm on each litter tile and test transitions while walking. Score each complete swing against its latched command, and compare leg power at matched achieved speed.
