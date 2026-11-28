# Adaptive_Lift

Train this task with the Python environment used for the DUET_V3 47k run. Run this command from the repository root.

```sh
bash scripts/train_adaptive_lift.sh train
```

The launcher runs unit tests, a GPU preflight and a two-update scratch PPO trial before starting a fresh 50,001-update run with 4,096 environments on one GPU. The GPU preflight ray-casts every litter cell of the compiled floor against the analytic map, then runs two short randomized-action trials. Use `check` to stop after preflight. Set `PYTHON` to the training interpreter or `NUM_ENVS` to the workstation batch size if needed. Both trainer invocations use fresh weights, observation statistics, optimizer and curriculum counters. The smoke checkpoint is kept in its own run directory. The launcher prints the compact console log path. TensorBoard logs and checkpoints go under `logs/rsl_rl/Adaptive_Lift/`.

## Command

The two clearance channels are the requested swing apex of the lowest sole point above its local load-bearing floor, in left/right order. A 7 cm command on 3 cm of litter asks for about 4 cm of free space above the litter. Each physical foot has seven collision capsules. The reward takes the minimum of their fourteen downward support points after referencing each point to its local bed. At a discontinuous tile seam this is a sampled approximation. The peak is recorded over the airborne portion of a completed moving swing with at least 60 ms of flight. The target is latched at takeoff so a mid-swing command change applies to the next swing.

Commands are uniform from 4 to 11 cm on every substrate. They resample every 2–5 s while standing or walking, with a 3.5 cm/s limit between draws. Three quarters of draws request the same height for both feet. The other quarter samples each side independently. The actor observes the command pair scaled to [−1, 1] in each of six history frames, after the original lean-task channels (444 inputs). The critic also observes the litter state (293 inputs, see below). The 13 outputs control twelve leg joints and waist yaw. The existing external arm action runs its own randomized trajectories.

For deployment, `clearance_commands` in deploy.yaml needs `minimum: 0.04, maximum: 0.11`. The controller's command filter should not move faster than the 3.5 cm/s training rate.

## Terrain

Terrain, spawn, friction, pushes, payloads and the terrain-level curriculum are the 47k lean task's: 20 columns × 10 rows of 8 m tiles. The 47k flat (15%) and soft_floor (60%) columns become litter tiles. The heightfield columns (pyramid slope, inverted slope, random rough, waves: 25%) are unchanged, and the lift, plow and landing terms rest on them as in the 47k task.

| Columns | Tile | Bed and layer |
| --- | --- | --- |
| 3 | Rigid | Bare hard floor; randomized foot friction 0.3–1.6 applies |
| 4 | Compacted | Bed raised 1.2 cm, 0.48–0.72 cm loose cover |
| 5 | Loose | Level bed, loose layer drawn per episode from 2 cm to the curriculum's deepest |
| 3 | Mixed | 2 m × 4 m patches of all three materials |

Litter cells use the 47k soft_floor contact: solref and solimp scale with the row difficulty from hard to soft (about 1 cm average sinkage while standing on the softest rows), with a per-tile friction of 0.3–1.0. The deepest loose layer is 3 cm through update 1,500 and ramps to 8 cm at 4,000. Litter drag is the 47k model: 500–1500 N s/m², with a 150 N per-foot cap, acting only inside the layer and without vertical support. The substrate follows the actual foot position across tile boundaries.

The critic observes `litter_truth`: the per-foot mean litter depth and bed elevation under the sole, whether each foot is on a litter tile, and the episode's drag strength. The actor stays blind to the terrain.

## Reward and curriculum choices

| Objective | Implementation |
| --- | --- |
| Walking, turning and stopping | Keep linear tracking, backward speed, alternating gait, lateral foot spacing and the settle window. Following HOMIE's G1 terms, yaw tracking is exp(−4 err²) at weight 2.0 (was exp(−2 err²) at 1.75, which pays almost fully for small yaw errors), and a separate lateral term exp(−4 (v_y − v_y,cmd)²) at 1.0 is added. A fifth of moving velocity draws each become a pure in-place turn or a pure sidestep at 35–100% of the current range. Air-time target matches the 0.33 s planned swing. |
| Lift and plowing | Track a single support-relative apex on completed swings. Penalize early horizontal travel below the latched target. Penalize horizontal travel buried in litter, including before liftoff, but only below the commanded clearance: walking at a 4 cm command through 8 cm of litter is obeying the command. A standing robot receives no swing credit. |
| Landing and slip | The 47k→50k landing fine-tune's form and weights, which cut heel-first steps from about 100% to under 10%. `touchdown_tilt` (−2.0) charges sole tilt beyond 1.5° over the last 1.5 cm of descent (above the local bed) and at the landing step. There is no standing or loading-phase tilt term, because a loaded heel sinks on soft litter. `touchdown_load` (−1.2) charges vertical load above 320 N during the first 40 ms after flight. Contact slip weight −0.35. |
| Motion and power | Keep first- and second-difference action costs, raw action bounds and joint velocity damping. Restrict the joint acceleration cost to policy-controlled joints. Absolute mechanical power sums the twelve leg joints, at −0.08 per 100 W with a 400 W cap. |
| Squat and lean | Keep pelvis-height tracking, squat-aware posture, support and standing balance. The squat floor reaches 0.22 m. Lean opens at update 6,000 and reaches 1.25 rad at 14,000, with a final stoop band of 0.32–0.62 m. |

Velocity, arm, squat, payload and push settings use the DUET poultry scratch configuration. Lift, plow, landing and power weights are zero through update 800 and ramp to full value at 3,000 to give balance and velocity tracking a head start. The runner retains scratch PPO exploration, live observation normalization and the DUET mirror loss, including left/right swaps for every clearance and litter pair.

## Reading the run

TensorBoard includes the existing fall, velocity, yaw, arm, lean and contact-slide metrics along with leg power in watts. Per-swing ratios over the same logging window:

- `swing_error_sum_m / completed_swings`: mean absolute apex error.
- `heel_first_landings / completed_swings`: fraction of landings more than 3° toe-up just before contact. The 47k policy is near 100% on flat ground; the 50k landing fine-tune was under 10%.
- `landing_pitch_sum_deg / completed_swings`: mean sole pitch just before contact, negative for heel first.

Stability of the learned gait requires trained checkpoint evaluation. Compare falls, velocity/yaw/lateral tracking and slips with the 47k reference under matched motions, seeds, terrain and arm/payload conditions, including pure turns and sidesteps. Check squat depth and lean separately. Sweep 4, 6, 7.5, 9 and 11 cm on each litter tile and test transitions while walking. Score each complete swing against its latched command, and compare leg power at matched achieved speed.
