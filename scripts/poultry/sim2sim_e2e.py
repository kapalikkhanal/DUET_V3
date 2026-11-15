"""Simulator for testing the deployed controller (g1_ctrl) on the unitree_mujoco model.

Same robot model and SDK bridge as ~/unitree_mujoco/simulate_python, with one
difference that matters: the motor PD torque is computed from the current joint
state at every physics step, and the physics runs at 2.5 ms. The stock Python
simulator computes it in another thread from sensor values that are one 5 ms
step old (MuJoCo fills the sensors before it integrates a step); with stiff,
lightly damped gains (ankle 40/2, waist 250/5) that makes the ankles and waist
ring at ~30 Hz, and a policy that is fine looks as if it cannot stand. A real
motor runs its PD loop far faster than either. --stock-bridge --substeps 1
reproduces the stock behaviour.

Automatic test, no window (plays the sequence below and prints the motion):

  python scripts/poultry/sim2sim_e2e.py --ctrl deploy/robots/g1_23dof/build/g1_ctrl

Watch and drive it yourself (window; run keyboard.py in another terminal):

  python scripts/poultry/sim2sim_e2e.py --viewer --manual --ctrl deploy/robots/g1_23dof/build/g1_ctrl
  keyboard.py: 1 = FixStand, wait 3 s, 2 = policy; then click the window and
  press 9 to release the band (8 / 7 lower / raise it first if needed).

--policy-dir picks the policy folder; without it g1_ctrl uses config.yaml.

Sequence (seconds after start):
   3  F1 -> FixStand (legs ease to the default pose over 3 s)
   8  F2 -> Velocity (policy in control), no command
  10  elastic band released (what pressing 9 in the viewer does); --release-at
      changes this. Released BEFORE Velocity, the robot topples forward in
      FixStand (a fixed pose cannot balance) and the policy takes over a
      falling robot.
  18  forward 0.4 m/s for 6 s, then stop
  30  sideways 0.3 m/s for 6 s, then stop
  42  turn 0.6 rad/s for 6 s, then stop
  54  backward 0.3 m/s for 6 s, then stop; ends at 64

After the run a gait table is printed for each walking phase: speed, step rate,
how regular the steps are, left/right swing time, side-to-side sway, roll and
heading wobble.

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
ap.add_argument("--viewer", action="store_true", help="open the MuJoCo window")
ap.add_argument("--long-walk", type=float, nargs=3, metavar=("VX", "VY", "YAW"), default=None,
                help="instead of the standard sequence, hold this one command for 30 s after standing "
                     "(about 80 steps, enough for a trustworthy step-regularity number)")
ap.add_argument("--manual", action="store_true",
                help="no scripted remote and no time limit: drive it with keyboard.py; the band is "
                     "released with key 9 in the window. Needs --viewer.")
ap.add_argument("--stock-bridge", action="store_true",
                help="let the stock Python bridge compute the motor torque, as unitree_mujoco.py does: "
                     "from another thread, using joint states that are up to one physics step old. "
                     "With stiff, lightly damped gains (ankle 40/2, waist 250/5) that makes the ankles "
                     "and waist ring at ~30 Hz. By default this script computes the PD torque itself, "
                     "from the latest command and the current state, right before every physics step.")
ap.add_argument("--sim-dt", type=float, default=config.SIMULATE_DT, help="simulator loop period, s")
ap.add_argument("--substeps", type=int, default=2,
                help="physics steps (each with a fresh PD torque) per loop period; 2 gives a 2.5 ms "
                     "motor loop while states are still published every 5 ms")
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
m.opt.timestep = a.sim_dt
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


class _NoCtrl:
  """The simulation data as the bridge sees it, except that its torque writes go nowhere."""

  def __init__(self, data):
    object.__setattr__(self, "_data", data)
    object.__setattr__(self, "ctrl", np.zeros(m.nu))

  def __getattr__(self, name):
    return getattr(self._data, name)


bridge = UnitreeSdk2Bridge(m, d)  # reads the publish period from the model: still sim_dt here
if not a.stock_bridge:
  bridge.mj_data = _NoCtrl(d)
m.opt.timestep = a.sim_dt / a.substeps
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
  (54, "backward 0.3", 0, 0, -0.3, 0), (60, "stop", 0, 0, 0, 0),
]
END = 64.0
if a.long_walk:
  vx, vy, wz = a.long_walk
  PHASES = PHASES[:5] + [(14, f"walk {vx:g} {vy:g} {wz:g}", 0, -vy, vx, -wz), (44, "stop", 0, 0, 0, 0)]
  END = 47.0
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
if a.manual and not a.viewer:
  sys.exit("--manual needs --viewer")
if not a.manual:
  threading.Thread(target=publish, daemon=True).start()
viewer = None
if a.viewer:
  import mujoco.viewer
  viewer = mujoco.viewer.launch_passive(m, d, key_callback=band.MujuocoKeyCallback)
next_sync = 0.0

from unitree_sdk2py.core.channel import ChannelSubscriber  # noqa: E402
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_  # noqa: E402

latest = {"msg": None, "count": 0, "stamps": []}


def _on_cmd(msg):
  latest["msg"] = msg
  latest["count"] += 1
  latest["stamps"].append(time.monotonic())


_cmd_sub = ChannelSubscriber("rt/lowcmd", LowCmd_)
_cmd_sub.Init(_on_cmd, 10)
NU = m.nu
SPD, XY, GAIT = [], [], []
QADR = np.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(NU)])
VADR = np.array([m.jnt_dofadr[m.actuator_trnid[i, 0]] for i in range(NU)])


def pitch_roll():
  w, x, y, z = d.qpos[3:7]
  R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
  g = R.T @ np.array([0.0, 0.0, -1.0])
  return np.degrees(np.arctan2(g[0], -g[2])), np.degrees(np.arctan2(-g[1], -g[2]))


print(f"scene {a.scene}\nmotor torque: {'stock bridge' if a.stock_bridge else 'computed every physics step'}; step {m.opt.timestep} s, band {'kept' if a.keep_band else f'released at {a.release_at:g} s'}, "
      f"band length {band.length:.2f} m, policy {a.policy_dir or 'from config.yaml'}")
print(f"{'t':>5} {'phase':>13} {'band':>4} {'pelvis z':>8} {'x cm':>7} {'y cm':>7} {'yaw deg':>7} {'pitch':>6} {'roll':>6} {'feet on floor':>13}")
next_log, sim_t, fell = 0.0, 0.0, False
foot_ids = [m.body(n).id for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
while ((t := time.monotonic() - t0) < END or a.manual) and (viewer is None or viewer.is_running()):
  if not a.manual:
    band.enable = a.keep_band or t < a.release_at
  d.xfrc_applied[torso, :3] = band.Advance(d.qpos[:3], d.qvel[:3]) if band.enable else 0.0
  cmd = None
  if not a.stock_bridge and latest["msg"] is not None:
    mc = latest["msg"].motor_cmd
    cmd = np.array([(mc[i].tau, mc[i].kp, mc[i].q, mc[i].kd, mc[i].dq) for i in range(NU)]).T
  for _ in range(a.substeps):
    if cmd is not None:
      # From the state itself, not from sensordata: after mj_step the sensors
      # still hold the values from BEFORE that step, i.e. one step of delay.
      d.ctrl[:NU] = cmd[0] + cmd[1] * (cmd[2] - d.qpos[QADR]) + cmd[3] * (cmd[4] - d.qvel[VADR])
    mujoco.mj_step(m, d)
    sim_t += m.opt.timestep
  if not a.manual:  # kept cheap: anything slow here makes the simulator fall behind real time
    GAIT.append((t, d.qpos[0], d.qpos[1], d.qpos[3], d.qpos[4], d.qpos[5], d.qpos[6],
                 d.xpos[foot_ids[0], 2], d.xpos[foot_ids[1], 2], sim_t))
  if 11.0 <= t < 17.0 and not a.manual:
    SPD.append((d.sensordata[NU + 5], d.sensordata[NU + 12]))
    XY.append((d.qpos[0], d.qpos[1]))
  if t >= next_log:
    p, r = pitch_roll()
    yaw = np.degrees(2 * np.arctan2(d.qpos[6], d.qpos[3]))
    floor = {m.geom_bodyid[c.geom1] for c in d.contact[:d.ncon]} | {m.geom_bodyid[c.geom2] for c in d.contact[:d.ncon]}
    feet = sum(f in floor for f in foot_ids)
    print(f"{t:5.1f} {'manual' if a.manual else phase(t)[1]:>13} {'on' if band.enable else 'off':>4} {d.qpos[2]:8.3f} {100 * d.qpos[0]:+7.1f} {100 * d.qpos[1]:+7.1f} {yaw:+7.1f} {p:+6.1f} {r:+6.1f} {feet:>13}")
    sys.stdout.flush()
    next_log += 1.0
    if d.qpos[2] < 0.35 and not fell:
      fell = True
      print("      ^ pelvis below 0.35 m: FELL")
  if viewer is not None and t >= next_sync:
    viewer.sync()
    next_sync = t + 0.03
  lag = sim_t - (time.monotonic() - t0)
  if lag > 0:
    time.sleep(lag)
if len(latest["stamps"]) > 10:
  gaps = np.diff(np.array(latest["stamps"])) * 1000
  print(f"command messages: {len(gaps) / (latest['stamps'][-1] - latest['stamps'][0]):.0f} per second; "
        f"gap between messages median {np.median(gaps):.1f} ms, 99% {np.percentile(gaps, 99):.1f} ms, max {gaps.max():.0f} ms "
        f"(motor torque from the {'stock bridge' if a.stock_bridge else 'current state at every physics step'})")
if SPD:
  v = np.array(SPD)
  print(f"standing 11-17 s: ankle-roll speed {np.sqrt((v[:, 0] ** 2).mean()):.2f} rad/s rms, waist {np.sqrt((v[:, 1] ** 2).mean()):.2f} rad/s rms "
        f"(a quiet stance is below 0.1); pelvis travelled {100 * np.linalg.norm(np.array(XY[-1]) - np.array(XY[0])):.0f} cm")
if GAIT:
  G0 = np.array(GAIT, dtype=float)
  qw, qx, qy, qz = G0[:, 3], G0[:, 4], G0[:, 5], G0[:, 6]
  gx, gy, gz = -2 * (qx * qz - qw * qy), -2 * (qy * qz + qw * qx), -(1 - 2 * (qx * qx + qy * qy))
  floor_z = np.percentile(G0[:, 7:9], 5, axis=0)  # ankle height with the foot flat on the floor
  G = np.column_stack([G0[:, 0], G0[:, 1], G0[:, 2], 2 * np.arctan2(qz, qw), np.degrees(np.arctan2(-gy, -gz)),
                       np.degrees(np.arctan2(gx, -gz)), G0[:, 7] < floor_z[0] + 0.012, G0[:, 8] < floor_z[1] + 0.012, G0[:, 9]])
  print("\ngait, measured from 1.5 s after each walking command until it ends:")
  print(f"{'phase':>13} | speed m/s | steps/s | step-time jitter | swing L / R s | sway cm | roll deg | pitch deg | heading wobble deg | heading drift deg/s")
  for (ts, label, _, lx, ly, rx), nxt in zip(PHASES, PHASES[1:] + [(END,)]):
    if not (lx or ly or rx):
      continue
    g = G[(G[:, 0] >= ts + 1.5) & (G[:, 0] < nxt[0])]
    if len(g) < 100:
      continue
    tt = g[:, 8] - g[0, 8]  # simulated time
    det = lambda x: x - np.polyval(np.polyfit(tt, x, 1), tt)  # noqa: E731
    yaw = np.unwrap(g[:, 3])
    head = yaw.mean()
    lat = -(g[:, 1] - g[0, 1]) * np.sin(head) + (g[:, 2] - g[0, 2]) * np.cos(head)
    dist = np.hypot(g[-1, 1] - g[0, 1], g[-1, 2] - g[0, 2])
    swing, tds = [[], []], []
    for f in (0, 1):
      on = g[:, 6 + f] > 0.5
      lo, td = np.where(~on[1:] & on[:-1])[0], np.where(on[1:] & ~on[:-1])[0]
      tds += list(tt[td])
      for i0 in lo:
        nx = td[td > i0]
        if len(nx):
          swing[f].append(tt[nx[0]] - tt[i0])
    per = np.diff(np.sort(tds))
    per = per[per > 0.08]  # ignore contact chatter at touchdown
    amp = lambda x: 2 * np.sqrt(2) * x.std()  # noqa: E731  (peak to peak of a sine with this spread)
    print(f"{label:>13} | {dist / tt[-1]:9.2f} | {1 / per.mean():7.2f} | {100 * per.std() / per.mean():15.0f}% | "
          f"{np.mean(swing[0]):.2f} / {np.mean(swing[1]):.2f}   | {100 * amp(det(lat)):7.1f} | {amp(g[:, 4]):8.1f} | {amp(g[:, 5]):9.1f} | "
          f"{amp(det(np.degrees(yaw))):18.1f} | {np.degrees(yaw[-1] - yaw[0]) / tt[-1]:+.1f}")
print(f"real-time factor {sim_t / (time.monotonic() - t0):.2f}")
if ctrl:
  ctrl.terminate()
  ctrl.wait(timeout=5)
