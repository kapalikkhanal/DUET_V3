"""Compare fixed-arm DUET, a conventional tray leveller, and DUET + arm RL.

All modes run the frozen DUET 91500 checkpoint for the legs and the same
scripted routine (common.ROUTINE: stand, forward 0.4, stop, backward 0.3,
turn 0.5, sideways 0.25, forward 0.5, stop; 31 s). For each seed one
environment is built (startup randomisation fixed by the seed) and every mode
is run after re-seeding, so spawn poses, litter depth, payload and commands are
identical across modes. Observation noise is off and pushes only with --pushes; eggs and tray
tipping do not end an episode (falls are recorded; a fallen robot is excluded
from the averages from that moment).

  PYTHONPATH=. python scripts/eggtray/evaluate.py --terrain poultry --contact built_up \
      --modes fixed feedback rl --checkpoint logs/rsl_rl/EggTray_G1_23dof/<run>/model_N.pt \
      --seeds 0 1 2 --n 64 --out logs/eggtray/eval/built_up
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

KEY_METRICS = [
  ("all_eggs_kept", "episodes with all eggs kept", 1.0),
  ("fell", "robot fall rate", 1.0),
  ("acc_rms", "tray acc RMS (m/s^2)", 1.0),
  ("acc_rms_walking", "tray acc RMS walking (m/s^2)", 1.0),
  ("acc_peak", "tray acc peak (m/s^2)", 1.0),
  ("acc_imu_rms", "tray accelerometer RMS (m/s^2)", 1.0),
  ("tilt_rms_deg", "tray tilt RMS (deg)", 1.0),
  ("tilt_max_deg", "tray tilt max (deg)", 1.0),
  ("ang_vel_rms", "tray roll/pitch rate RMS (rad/s)", 1.0),
  ("jerk_rms", "tray jerk RMS (m/s^3)", 1.0),
  ("egg_force_peak_n", "egg contact force peak (N)", 1.0),
  ("egg_excess_impulse_ns", "egg impulse above 2x weight (N s)", 1.0),
  ("eggs_over_2n", "eggs ever above 2 N", 1.0),
  ("eggs_over_5n", "eggs ever above 5 N", 1.0),
  ("egg_speed_mean", "egg speed in cup (m/s)", 1.0),
  ("lin_vel_err", "DUET lin. vel. tracking error (m/s)", 1.0),
  ("yaw_rate_err", "DUET yaw-rate tracking error (rad/s)", 1.0),
  ("transport_m", "successful transport distance (m)", 1.0),
  ("residual_mean_rad", "mean |arm residual| (rad)", 1.0),
  ("grasp_strain_max_mm", "max weld stretch (mm)", 1.0),
  ("duet_extreme_steps", "DUET |a|>15 steps", 1.0),
]


def trace_cb(store, env_id=0):
  def cb(k, t, name, u):
    s = C.state(u).sample()
    store["t"].append(t)
    store["acc"].append(float(s["acc"][env_id].norm()))
    store["tilt"].append(float(torch.rad2deg(s["tilt"][env_id])))
    store["kept"].append(float((~s["lost"][env_id]).float().sum()))
    store["residual"].append(u.action_manager.get_term("arm_residual").residual[env_id].cpu().tolist())
  return cb


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--terrain", default="poultry", choices=("flat", "poultry"))
  ap.add_argument("--contact", default="built_up", choices=("built_up", "fresh_deep", "training", "rigid", "floor"))
  ap.add_argument("--eggs", type=int, default=6, choices=(6, 12))
  ap.add_argument("--litter", type=float, default=None, help="fixed loose-litter depth (m); default random 0-6 cm")
  ap.add_argument("--modes", nargs="+", default=["fixed", "feedback", "rl"],
                  help="arm mode, optionally @DUET legs (91500 default), e.g. fixed@squatreach50k")
  ap.add_argument("--checkpoint", default=None)
  ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
  ap.add_argument("--n", type=int, default=64)
  ap.add_argument("--pushes", default="none", choices=sorted(C.PUSH_SETS))
  ap.add_argument("--out", default="logs/eggtray/eval/run")
  a = ap.parse_args()
  os.makedirs(a.out, exist_ok=True)

  per_mode: dict[str, list] = {m: [] for m in a.modes}
  traces: dict[str, dict] = {}
  for seed in a.seeds:
    cfg = C.make_cfg(a.terrain, a.eggs, a.contact, num_envs=a.n, litter=a.litter)
    wrapped = C.make_env(cfg, seed=seed)
    u = wrapped.unwrapped
    for mode in a.modes:
      u.seed(seed)
      ctrl = C.make_controller(mode, u, a.checkpoint)
      acc = C.Accum(u.num_envs, u.device)
      store = {"t": [], "acc": [], "tilt": [], "kept": [], "residual": []}
      C.run_routine(wrapped, ctrl, acc, frame_cb=trace_cb(store) if seed == a.seeds[0] else None,
                    pushes=C.PUSH_SETS[a.pushes])
      per_mode[mode].append(acc.per_env())
      if seed == a.seeds[0]:
        traces[mode] = store
      r = acc.per_env()
      print(f"seed {seed} {mode:>8}: all eggs kept {r['all_eggs_kept'].mean():.3f}  falls {r['fell'].mean():.3f}  "
            f"acc RMS {r['acc_rms'].mean():.3f}  tilt RMS {r['tilt_rms_deg'].mean():.2f}  "
            f"peak egg force {r['egg_force_peak_n'].mean():.2f}", flush=True)
    wrapped.close()
    del wrapped, u
    torch.cuda.empty_cache()

  summary = {m: C.summarize(per_mode[m]) for m in a.modes}
  base = summary[a.modes[0]]  # relative changes are against the first mode
  print(f"\n{a.terrain}/{a.contact}, {a.eggs} eggs, litter {'random 0-6 cm' if a.litter is None else a.litter}, "
        f"{len(a.seeds)} seeds x {a.n} episodes, routine {C.ROUTINE_S:.0f} s; mean +- std across seeds")
  header = f"{'metric':>40} " + " ".join(f"{m:>22}" for m in a.modes)
  print(header)
  for key, label, _ in KEY_METRICS:
    row = f"{label:>40} "
    for m in a.modes:
      v = summary[m][key]
      cell = f"{v['mean']:.3f}+-{v['std_across_seeds']:.3f}"
      if base is not None and m != a.modes[0] and abs(base[key]["mean"]) > 1e-9:
        cell += f" ({100 * (v['mean'] / base[key]['mean'] - 1):+.0f}%)"
      row += f"{cell:>22} "
    print(row)

  report = {"args": vars(a), "routine": C.ROUTINE, "summary": summary, "traces": traces}
  with open(os.path.join(a.out, "summary.json"), "w") as f:
    json.dump(report, f, indent=1)
  per_ep = {m: {k: np.concatenate([r[k] for r in per_mode[m]]).tolist() for k in per_mode[m][0]} for m in a.modes}
  with open(os.path.join(a.out, "per_episode.json"), "w") as f:
    json.dump(per_ep, f)
  try:
    plot(report, a.out)
  except Exception as e:  # plotting is optional
    print(f"[WARN] plotting failed: {e}")
  print(f"\nwrote {a.out}/summary.json, per_episode.json and plots")


def plot(report, out):
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  summary, traces = report["summary"], report["traces"]
  modes = list(summary)
  colors = {"fixed": "#7a7a7a", "feedback": "#c98a2b", "rl": "#2b6cb0", "fixed@squatreach50k": "#b04a4a"}
  keys = [("acc_rms", "tray acc RMS (m/s²)"), ("tilt_rms_deg", "tray tilt RMS (°)"),
          ("ang_vel_rms", "roll/pitch rate RMS (rad/s)"), ("egg_force_peak_n", "peak egg force (N)"),
          ("all_eggs_kept", "episodes, all eggs kept"), ("fell", "fall rate")]
  fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
  for ax, (k, label) in zip(axes.flat, keys):
    means = [summary[m][k]["mean"] for m in modes]
    stds = [summary[m][k]["std_across_seeds"] for m in modes]
    ax.bar(modes, means, yerr=stds, capsize=4, color=[colors.get(m, "#555") for m in modes])
    ax.set_title(label, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
  fig.suptitle(f"{report['args']['terrain']} / {report['args']['contact']}, {report['args']['eggs']} eggs, "
               f"pushes {report['args'].get('pushes', 'none')} "
               f"(mean ± std across {len(report['args']['seeds'])} seeds)")
  fig.tight_layout()
  fig.savefig(os.path.join(out, "metrics.png"), dpi=130)
  plt.close(fig)

  fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
  for m in modes:
    tr = traces.get(m)
    if not tr:
      continue
    axes[0].plot(tr["t"], tr["acc"], lw=0.8, color=colors.get(m), label=m)
    axes[1].plot(tr["t"], tr["tilt"], lw=0.8, color=colors.get(m), label=m)
    axes[2].plot(tr["t"], tr["kept"], lw=1.2, color=colors.get(m), label=m)
  for seg in report["routine"]:
    for ax in axes:
      ax.axvline(seg[1], color="#ddd", lw=0.6, zorder=0)
    axes[0].text(seg[1] + 0.1, axes[0].get_ylim()[1] * 0.92, seg[0], fontsize=7, color="#666")
  axes[0].set_ylabel("|tray acc| (m/s²)")
  axes[1].set_ylabel("tray tilt (°)")
  axes[2].set_ylabel("eggs kept")
  axes[2].set_xlabel("time (s), environment 0")
  axes[0].legend(loc="upper right", fontsize=8)
  for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
  fig.tight_layout()
  fig.savefig(os.path.join(out, "traces.png"), dpi=130)
  plt.close(fig)


if __name__ == "__main__":
  main()
