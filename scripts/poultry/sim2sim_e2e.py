"""End-to-end test of the deployed controller: g1_ctrl + unitree_mujoco, no viewer.

Runs the unitree_mujoco Python simulator headless (same model, same SDK bridge,
same 5 ms step as simulate_python/unitree_mujoco.py), plays a fixed sequence of
remote presses on rt/wirelesscontroller (what keyboard.py sends), and prints
what the robot did. Start g1_ctrl yourself in another terminal, or pass
--ctrl to have this script start and stop it.

  python scripts/poultry/sim2sim_e2e.py --ctrl deploy/robots/g1_23dof/build/g1_ctrl

Sequence (seconds after start):
   3  F1 -> FixStand (legs ease to the default pose over 3 s)
   8  F2 -> Velocity (policy in control), no command
  10  elastic band released (what pressing 9 in the viewer does); --release-at
      changes this. Released BEFORE Velocity, the robot topples forward in
      FixStand (a fixed pose cannot balance) and the policy takes over a
      falling robot.
  18  forward 0.4 m/s for 6 s, then stop
  30  sideways 0.3 m/s for 6 s, then stop
  42  turn 0.6 rad/s for 6 s, then stop; ends at 52

--keep-band never releases the band (the robot hangs from it).
Uses DDS domain 0 on lo: do not run while another simulator or controller is up.
"""

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

import mujoco
import numpy as np

SIM_DIR = Path.home() / "unitree_mujoco" / "simulate_python"
sys.path.insert(0, str(SIM_DIR))
import config  # noqa: E402  (unitree_mujoco's own config.py)
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher  # noqa: E402
from unitree_sdk2py.idl.default import unitree_go_msg_dds__WirelessController_  # noqa: E402
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_  # noqa: E402
from unitree_sdk2py_bridge import ElasticBand, UnitreeSdk2Bridge  # noqa: E402

KEY_F1, KEY_F2 = 1 << 6, 1 << 7

ap = argparse.ArgumentParser()
ap.add_argument("--ctrl", default=None, help="path to g1_ctrl; started with --network lo")
ap.add_argument("--scene", default=str(SIM_DIR / config.ROBOT_SCENE))
ap.add_argument("--keep-band", action="store_true")
ap.add_argument("--keep-table", action="store_true",
                help="leave scene_23dof.xml's table and box 0.6 m in front of the robot "
                     "(by default they are moved out of the way: walking forward runs into them)")
ap.add_argument("--release-at", type=float, default=10.0,
                help="seconds after start when the band is released (Velocity starts at 8)")
ap.add_argument("--policy-dir", default=None, help="passed to g1_ctrl (default: config.yaml policy_dir)")
ap.add_argument("--band-height", type=float, default=0.70,
                help="lower the band until the pelvis hangs at this height, feet just touching "
                     "(what pressing 8 in the viewer does); 0 keeps the stock band, which lifts "
                     "the robot about 0.5 m off the floor")
a = ap.parse_args()

m = mujoco.MjModel.from_xml_path(a.scene)
d = mujoco.MjData(m)
m.opt.timestep = config.SIMULATE_DT
if not a.keep_table:
  for name in ("table", "pick_box"):
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
      continue
    if m.body_jntnum[bid]:
      d.qpos[m.jnt_qposadr[m.body_jntadr[bid]]] += 30.0
    else:
      m.body_pos[bid, 0] += 30.0
  mujoco.mj_forward(m, d)
ChannelFactoryInitialize(0, "lo")
bridge = UnitreeSdk2Bridge(m, d)
bridge.SetupDDSWireless()
band = ElasticBand()
if a.band_height > 0:
  weight = m.body_subtreemass[m.body("pelvis").id] * 9.81
  band.length = band.point[2] - a.band_height - weight / band.stiffness
torso = m.body("torso_link").id
pub = ChannelPublisher("rt/wirelesscontroller", WirelessController_)
pub.Init()

PHASES = [  # (start s, label, keys, lx, ly, rx)
  (0, "passive", 0, 0, 0, 0),
  (3, "F1 fixstand", KEY_F1, 0, 0, 0), (3.2, "fixstand", 0, 0, 0, 0),
  (8, "F2 velocity", KEY_F2, 0, 0, 0), (8.2, "stand", 0, 0, 0, 0),
  (18, "forward 0.4", 0, 0, 0.4, 0), (24, "stop", 0, 0, 0, 0),
  (30, "sideways 0.3", 0, 0.3, 0, 0), (36, "stop", 0, 0, 0, 0),
  (42, "turn 0.6", 0, 0, 0, 0.6), (48, "stop", 0, 0, 0, 0),
]
END = 52.0
t0 = time.monotonic()


def phase(t):
  return [p for p in PHASES if p[0] <= t][-1]


def publish():
  msg = unitree_go_msg_dds__WirelessController_()
  while (t := time.monotonic() - t0) < END:
    _, _, keys, lx, ly, rx = phase(t)
    msg.keys, msg.lx, msg.ly, msg.rx, msg.ry = keys, lx, ly, rx, 0.0
    pub.Write(msg)
    time.sleep(0.01)


ctrl = None
if a.ctrl:
  ctrl_log = open("/tmp/g1_ctrl_e2e.log", "w")
  cmd = [str(Path(a.ctrl).resolve()), "--network", "lo"]
  if a.policy_dir:
    cmd += ["--policy-dir", str(Path(a.policy_dir).resolve())]
  ctrl = subprocess.Popen(cmd,
                          cwd=str(Path(a.ctrl).resolve().parent), stdout=ctrl_log, stderr=subprocess.STDOUT)
threading.Thread(target=publish, daemon=True).start()


def pitch_roll():
  w, x, y, z = d.qpos[3:7]
  R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
  g = R.T @ np.array([0.0, 0.0, -1.0])
  return np.degrees(np.arctan2(g[0], -g[2])), np.degrees(np.arctan2(-g[1], -g[2]))


print(f"scene {a.scene}\nstep {m.opt.timestep} s, band {'kept' if a.keep_band else f'released at {a.release_at:g} s'}, "
      f"band length {band.length:.2f} m, policy {a.policy_dir or 'from config.yaml'}")
print(f"{'t':>5} {'phase':>13} {'band':>4} {'pelvis z':>8} {'x cm':>7} {'y cm':>7} {'yaw deg':>7} {'pitch':>6} {'roll':>6} {'feet on floor':>13}")
next_log, sim_t, fell = 0.0, 0.0, False
foot_ids = [m.body(n).id for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
while (t := time.monotonic() - t0) < END:
  band.enable = a.keep_band or t < a.release_at
  d.xfrc_applied[torso, :3] = band.Advance(d.qpos[:3], d.qvel[:3]) if band.enable else 0.0
  mujoco.mj_step(m, d)
  sim_t += m.opt.timestep
  if t >= next_log:
    p, r = pitch_roll()
    yaw = np.degrees(2 * np.arctan2(d.qpos[6], d.qpos[3]))
    floor = {m.geom_bodyid[c.geom1] for c in d.contact[:d.ncon]} | {m.geom_bodyid[c.geom2] for c in d.contact[:d.ncon]}
    feet = sum(f in floor for f in foot_ids)
    print(f"{t:5.1f} {phase(t)[1]:>13} {'on' if band.enable else 'off':>4} {d.qpos[2]:8.3f} {100 * d.qpos[0]:+7.1f} {100 * d.qpos[1]:+7.1f} {yaw:+7.1f} {p:+6.1f} {r:+6.1f} {feet:>13}")
    sys.stdout.flush()
    next_log += 1.0
    if d.qpos[2] < 0.35 and not fell:
      fell = True
      print("      ^ pelvis below 0.35 m: FELL")
  lag = sim_t - (time.monotonic() - t0)
  if lag > 0:
    time.sleep(lag)
print(f"real-time factor {sim_t / (time.monotonic() - t0):.2f}")
if ctrl:
  ctrl.terminate()
  ctrl.wait(timeout=5)
