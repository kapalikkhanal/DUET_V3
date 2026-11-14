"""Record what the robot did AND what it was told to do (run next to g1_ctrl).

Successor to DUET/Paper/hardware_kit/record_lowstate.py. That one stored only
rt/lowstate, which is enough to describe a gait but not to fit the motors: for
that the commanded joint target of every sample is needed as well. This one
stores both streams, each with its own timestamps:

  rt/lowstate  t, tick, q, dq, tau, quat, gyro, acc, remote
  rt/lowcmd    t_cmd, q_des, dq_des, kp, kd, tau_ff

Joint arrays are in policy order (deploy.yaml joint_ids_map), 23 columns.
``remote`` is the raw 40-byte wireless-remote block, from which the stick
commands are recovered later. ``tick`` is the robot's own millisecond counter.

  python scripts/hardware/record_robot.py --out trials/lab_stand_01 \
      --controller poultry_91500 --surface "lab floor" \
      --note "stand 60 s, no commands" --iface enp132s0

Stop with Ctrl-C. If no rt/lowcmd arrives (for example with the default
firmware, whose commands may not be visible on the network) the state stream is
still saved and the script says so.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

# Policy joint order -> Unitree SDK motor index (deploy.yaml joint_ids_map).
JOINT_IDS_MAP = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26]


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--out", required=True, help="output path without extension")
  ap.add_argument("--iface", default="lo", help="network interface for DDS (robot: its 192.168.123.x NIC)")
  ap.add_argument("--controller", required=True,
                  help="what is driving the robot, e.g. poultry_91500 | default_firmware | poultry_trial2k")
  ap.add_argument("--surface", required=True, help='e.g. "lab floor", "farm litter 5 cm"')
  ap.add_argument("--note", default="", help="what was commanded, anything unusual")
  a = ap.parse_args()

  from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
  from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

  state = {k: [] for k in ("t", "tick", "q", "dq", "tau", "quat", "gyro", "acc", "remote")}
  cmd = {k: [] for k in ("t_cmd", "q_des", "dq_des", "kp", "kd", "tau_ff")}
  rec = {"on": True}
  f32 = np.float32

  def on_state(msg) -> None:
    if not rec["on"]:
      return
    ms = msg.motor_state
    state["t"].append(time.time())
    state["tick"].append(msg.tick)
    state["q"].append(np.array([ms[i].q for i in JOINT_IDS_MAP], f32))
    state["dq"].append(np.array([ms[i].dq for i in JOINT_IDS_MAP], f32))
    state["tau"].append(np.array([ms[i].tau_est for i in JOINT_IDS_MAP], f32))
    state["quat"].append(np.array(msg.imu_state.quaternion, f32))
    state["gyro"].append(np.array(msg.imu_state.gyroscope, f32))
    state["acc"].append(np.array(msg.imu_state.accelerometer, f32))
    state["remote"].append(np.array(msg.wireless_remote, np.uint8))

  def on_cmd(msg) -> None:
    if not rec["on"]:
      return
    mc = msg.motor_cmd
    cmd["t_cmd"].append(time.time())
    cmd["q_des"].append(np.array([mc[i].q for i in JOINT_IDS_MAP], f32))
    cmd["dq_des"].append(np.array([mc[i].dq for i in JOINT_IDS_MAP], f32))
    cmd["kp"].append(np.array([mc[i].kp for i in JOINT_IDS_MAP], f32))
    cmd["kd"].append(np.array([mc[i].kd for i in JOINT_IDS_MAP], f32))
    cmd["tau_ff"].append(np.array([mc[i].tau for i in JOINT_IDS_MAP], f32))

  # The buffers hold arrays with no reference cycles. With the collector on, its
  # full passes blocked the DDS callback for seconds in a 17 min recording.
  gc.disable()
  ChannelFactoryInitialize(0, a.iface)
  sub_state = ChannelSubscriber("rt/lowstate", LowState_)
  sub_state.Init(on_state, 10)
  sub_cmd = ChannelSubscriber("rt/lowcmd", LowCmd_)
  sub_cmd.Init(on_cmd, 10)
  print(f"recording rt/lowstate + rt/lowcmd on {a.iface} -> {a.out}.npz  (Ctrl-C to stop)")
  t0 = time.time()
  try:
    while True:
      time.sleep(1.0)
      el = max(time.time() - t0, 1e-6)
      print(f"  {el:5.0f} s   state {len(state['t'])} ({len(state['t']) / el:.0f} Hz)"
            f"   cmd {len(cmd['t_cmd'])} ({len(cmd['t_cmd']) / el:.0f} Hz)   ", end="\r")
  except KeyboardInterrupt:
    pass
  # The DDS threads keep delivering while we save: stop them, then trim each
  # stream to a common length.
  rec["on"] = False
  time.sleep(0.05)
  n = min(len(v) for v in state.values())
  m = min(len(v) for v in cmd.values())
  if n == 0:
    print("\nno rt/lowstate received: check --iface and that the robot is on")
    return 1
  if m == 0:
    print("\nNOTE: no rt/lowcmd received. Saving the state stream only; the motor fit needs the commands.")
  out = Path(a.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  meta = {
    "controller": a.controller, "surface": a.surface, "note": a.note, "iface": a.iface,
    "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(state["t"][0])),
    "joint_ids_map": JOINT_IDS_MAP, "format": "record_robot v1",
  }
  arrays = {k: np.asarray(v[:n]) for k, v in state.items()}
  arrays.update({k: np.asarray(v[:m]) for k, v in cmd.items()})
  np.savez_compressed(out.with_suffix(".npz"), **arrays, meta=json.dumps(meta))
  dur = state["t"][n - 1] - state["t"][0]
  print(f"\nsaved {n} state samples and {m} command samples, {dur:.1f} s -> {out.with_suffix('.npz')}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
