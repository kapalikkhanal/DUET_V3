import math
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import mujoco
import numpy as np
import torch
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from src.tasks.common.mdp.symmetry import _TERM_RULES
from src.tasks.duet.config.g1_23dof.adaptive_lift import LOOSE_DEPTH_STAGES, TASK, TILE_PROPORTIONS
from src.tasks.duet.duet_env_cfg import ITER
from src.tasks.duet.mdp import adaptive_lift as lift
from src.tasks.duet.mdp.lift_terrain import (LAYOUT, MATERIALS, LiftTile, SplitTile, column_kinds, material_at,
                                             split_heights, split_offset)
from src.tasks.duet.mdp.poultry_surface import litter_wrench, point_kinematics

BASE = "Unitree-G1-23Dof-Duet-PoultrySurface-Lean"


def pitch(theta):
  """Body rotation about y; positive theta is toe down."""
  c, s = math.cos(theta), math.sin(theta)
  return torch.tensor([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


class AdaptiveLiftTests(unittest.TestCase):
  def test_continuous_command_transitions_and_partial_reset(self):
    torch.manual_seed(7)
    command = lift.LiftCommandCfg(resampling_time_range=(2., 5.)).build(NS(num_envs=256, device="cpu", step_dt=.02))
    ids = torch.arange(256)
    command.reset(ids)
    self.assertGreater(command.command[:, 0].unique().numel(), 240)
    self.assertTrue(command.command.ge(.04).all() and command.command.le(.11).all())
    paired = command.command[:, 0] == command.command[:, 1]
    self.assertTrue(.6 < paired.float().mean() < .9)
    command.value[:] = torch.tensor([.04, .11])
    command.goal[:] = torch.tensor([.11, .04])
    command.time_left.fill_(5.)
    previous = command.value.clone()
    command.compute(.02)
    torch.testing.assert_close(command.value - previous,
                               torch.tensor([.0007, -.0007]).expand_as(command.value))
    previous = command.value.clone()
    command.reset(ids[:8])
    torch.testing.assert_close(command.value[8:], previous[8:])
    command.value[:] = torch.tensor([.04, .11])
    encoded = lift.clearance_commands(NS(command_manager=NS(get_term=lambda _: command)))
    torch.testing.assert_close(encoded[0], torch.tensor([-1., 1.]))
    self.assertEqual(_TERM_RULES["clearance_commands"][0], [1, 0])
    self.assertEqual(_TERM_RULES["litter_truth"][0], [1, 0, 3, 2, 5, 4, 6])

  def test_column_layout_and_spatial_materials(self):
    generator = load_env_cfg(TASK).scene.terrain.terrain_generator
    kinds = column_kinds(generator)
    self.assertEqual(kinds, [0] * 3 + [1] * 4 + [2] * 4 + [3] * 2 + [4] * 2 + [-1] * 5)
    terrain = NS(size=generator.size, num_rows=generator.num_rows, num_cols=generator.num_cols)
    col_y = lambda c: (c - 10 + .5) * 8.  # noqa: E731  column centre
    points = torch.tensor([[0., col_y(0)], [0., col_y(4)], [0., col_y(8)], [-7., col_y(11) - 2.],
                           [-1., col_y(11) + 2.], [0., col_y(16)], [0., 90.]])
    material, valid = material_at(points, terrain, torch.as_tensor(LAYOUT), torch.tensor(kinds))
    # Mixed tile: x cell 0 / y cell 0 is rigid; x cell 3 / y cell 1 is loose.
    self.assertEqual(material.tolist(), [0, 1, 2, 0, 2, 0, 0])
    self.assertEqual(valid.tolist(), [True] * 5 + [False, False])

  def test_material_query_matches_compiled_support(self):
    terrain = NS(size=(8., 8.), num_rows=1, num_cols=4)
    layout = torch.as_tensor(LAYOUT)
    for col in range(4):
      spec = mujoco.MjSpec()
      spec.worldbody.add_body(name="terrain")
      tile = LiftTile(kind=col, proportion=1., size=terrain.size)
      tile.function(1., spec, np.random.default_rng(0))
      model = spec.compile()
      data = mujoco.MjData(model)
      mujoco.mj_forward(model, data)
      for i, j in np.ndindex(4, 2):
        origin = np.array([(i + .5) * 2, (j + .5) * 4, 1.])
        geom = np.empty(1, dtype=np.int32)
        distance = mujoco.mj_ray(model, data, origin, np.array([0., 0., -1.]), None, True, -1, geom)
        global_xy = torch.from_numpy(origin[:2] + np.array([-4., (col - 2) * 8.]))
        material, valid = material_at(global_xy, terrain, layout, torch.arange(4))
        self.assertTrue(valid)
        self.assertAlmostEqual(1. - distance, MATERIALS[material.item()][0])
        # Litter is soft at full difficulty and outranks the foot; rigid floor does not.
        if material.item() == 0:
          self.assertEqual(model.geom_priority[geom[0]], 0)
        else:
          self.assertEqual(model.geom_priority[geom[0]], 2)
          self.assertAlmostEqual(model.geom_solref[geom[0], 0], .05)
          self.assertAlmostEqual(model.geom_solimp[geom[0], 0], .6)

  def test_split_level_heights_match_compiled_strips(self):
    generator = load_env_cfg(TASK).scene.terrain.terrain_generator
    heights, widths = split_heights(generator)
    split_cols = [c for c, k in enumerate(column_kinds(generator)) if k == 4]
    self.assertEqual(len(split_cols), 2)
    self.assertAlmostEqual(heights[0, split_cols[0]], .02)
    self.assertAlmostEqual(heights[-1, split_cols[0]], .07, places=6)
    self.assertTrue((np.diff(heights[:, split_cols[0]]) > 0).all())
    self.assertTrue((heights[:, [c for c in range(generator.num_cols) if c not in split_cols]] == 0).all())
    terrain = NS(size=generator.size, num_rows=generator.num_rows, num_cols=generator.num_cols)
    tile = SplitTile(proportion=1., size=generator.size, levels=generator.num_rows)
    n, width = tile.strips()
    self.assertAlmostEqual(width, widths[split_cols[0]])
    for row in (0, 5, 9):
      spec = mujoco.MjSpec()
      spec.worldbody.add_body(name="terrain")
      origin = tile.function((row + .5) / generator.num_rows, spec, np.random.default_rng(0)).origin
      model = spec.compile()
      data = mujoco.MjData(model)
      mujoco.mj_forward(model, data)
      self.assertEqual(origin[2], 0.)  # spawn on a low strip
      local = np.array([(2., (k + .5) * width) for k in range(n)])
      tops = [1. - mujoco.mj_ray(model, data, np.array([x, y, 1.]), np.array([0., 0., -1.]), None, True, -1,
                                 np.empty(1, dtype=np.int32)) for x, y in local]
      offset = np.array([(row - generator.num_rows / 2) * 8., (split_cols[0] - generator.num_cols / 2) * 8.])
      analytic = split_offset(torch.tensor(local + offset, dtype=torch.float32), terrain,
                              torch.as_tensor(heights), torch.as_tensor(widths))
      np.testing.assert_allclose(analytic.numpy(), tops, atol=1e-5)
      self.assertTrue(set(np.round(tops, 5)) == {0., round(float(heights[row, split_cols[0]]), 5)})

  def test_pull_schedule_release_and_anchor(self):
    # One pull per robot in the 3 s window: it starts at 0.1 s, the next gap is 5 s.
    cfg = lift.PullForceCfg(entity_name="robot", duration_range=(1., 1.), gap_range=(5., 5.))
    self.assertEqual((cfg.max_force, cfg.release_force), (100., 40.))
    term = object.__new__(lift.PullForce)
    term.cfg = cfg
    lean = NS(active=torch.zeros(4, dtype=torch.bool))
    height = torch.full((4, 1), .78)
    term._env = NS(step_dt=.02, num_envs=4, device="cpu", command_manager=NS(
      active_terms=["lean", "base_height"], get_term=lambda _: lean, get_command=lambda _: height))
    term.scale = 1.
    term.body_ids = [0, 1]
    for name, value in (("target", torch.zeros(4, 3)), ("force", torch.zeros(4, 3)),
                        ("body", torch.zeros(4, dtype=torch.long)), ("elapsed", torch.zeros(4)),
                        ("duration", torch.zeros(4)), ("gap", torch.full((4,), .1)),
                        ("pulling", torch.zeros(4, dtype=torch.bool)), ("since_strong", torch.full((4,), 1e3)),
                        ("wrench", torch.zeros(4, 2, 3))):
      setattr(term, name, value)
    torch.manual_seed(0)
    peak, released, wrench_ok = torch.zeros(4), [], True
    for _ in range(150):  # 3 s: gap, a 1 s pull, settle
      term.process_actions(None)
      peak = torch.maximum(peak, term.force.norm(dim=-1))
      self.assertTrue((term.force[:, 2] == 0).all())
      wrench_ok &= bool(torch.allclose(term.wrench.sum(1), term.force))
      released.append(term.released.clone())
    self.assertTrue(wrench_ok)
    self.assertTrue((peak <= 100. + 1e-4).all() and (peak > 0).all())
    strong = peak > cfg.release_force
    released = torch.stack(released)
    # Released through the pull and for about settle_time after it.
    torch.testing.assert_close(released.any(0), strong)
    self.assertTrue(((released.sum(0)[strong] * .02) >= 1. - 1e-6).all())
    # No pull starts while a lean or a deep squat is commanded.
    term.pulling[:] = False
    term.gap[:] = 0.
    lean.active[:2] = True
    height[2:] = .40
    term.process_actions(None)
    self.assertFalse(term.pulling.any())
    lean.active[:] = False
    height[:] = .78
    term.scale = 0.
    term.pulling[:] = False
    term.gap[:] = 0.
    term.since_strong[:] = 1e3
    term.process_actions(None)
    self.assertEqual(term.force.abs().sum().item(), 0.)
    self.assertFalse(term.released.any())

  def test_drag_and_command_capped_plow(self):
    pos = torch.tensor([[[0., 0., .12]]])
    rot = torch.tensor([[[[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]]])
    tips = torch.tensor([[[-.05, 0., 0.], [.07, 0., 0.]]])
    points, vel = point_kinematics(pos, rot, torch.ones_like(pos), torch.zeros_like(pos),
                                   tips, torch.tensor([[.01, .01]]))
    torch.testing.assert_close(points[..., 2].amin(-1), torch.tensor([[.04]]))
    # Raised support under the toe: burial 5 mm, dissipative xy drag, zero vertical force.
    depth = torch.tensor([[[0., .015]]])
    bed = torch.tensor([[[0., .030]]])
    force, _ = litter_wrench(points, vel, pos, depth, torch.tensor([800.]), 45., floor_height=bed)
    torch.testing.assert_close(force, torch.tensor([[[-2., -2., 0.]]]))
    # Sole points 3 cm above the bed, moving at 1 m/s through 8 cm of litter.
    s = dict(layer=torch.full((1, 2, 2), .08), points=torch.zeros(1, 2, 2, 3),
             velocity=torch.tensor([1., 0., 0.]).expand(1, 2, 2, 3), active=torch.ones(1, 2, dtype=torch.bool))
    s["points"][..., 2] = .03
    command = torch.full((1, 2), .11)
    env = NS(action_manager=NS(get_term=lambda _: NS(sample=lambda: s)),
             command_manager=NS(get_command=lambda _: command))
    torch.testing.assert_close(lift.loose_plow(env), torch.tensor([2 * .05]))
    command.fill_(.03)  # travelling at the commanded 3 cm obeys the command
    self.assertEqual(lift.loose_plow(env).item(), 0)
    command.fill_(.05)
    torch.testing.assert_close(lift.loose_plow(env), torch.tensor([2 * .02]))
    s["layer"].zero_()
    self.assertEqual(lift.loose_plow(env).item(), 0)

  def test_apex_latch_heel_first_and_landing_window(self):
    # Run the actual cached parent and child swing state machines with synthetic contact/geometry.
    surface = object.__new__(lift.LiftSurface)
    control = torch.tensor([[.04, .11]])
    twist = torch.tensor([[.4, 0., 0.]])
    contact = torch.zeros(1, 2, dtype=torch.bool)
    sensor = NS(found=contact, last_air_time=torch.full_like(control, .14), current_air_time=torch.zeros_like(control))
    env = NS(scene={"feet_ground_contact": NS(data=sensor)}, common_step_counter=0,
             _sim_step_counter=0, step_dt=.02, episode_length_buf=torch.zeros(1), device="cpu",
             command_manager=NS(get_command=lambda name: control if name == "clearance" else twist))
    env.action_manager = NS(get_term=lambda _: surface)
    surface._env = env
    surface.cfg = NS(period=.75, stance_fraction=.56)
    surface._sample_key = None
    for field in ("peak", "last_apex", "last_liftoff_angle", "swing_target", "air_pitch",
                  "swing_bed_lo", "swing_bed_hi"):
      setattr(surface, field, torch.zeros_like(control))
    surface.last_bed = torch.zeros(1, 2, 14)
    for field in ("was_air", "complete_swing", "has_landed", "previous_contact", "tracking"):
      setattr(surface, field, torch.zeros_like(contact))
    points = torch.zeros(1, 2, 14, 3)
    velocity = torch.zeros_like(points)
    rotation = torch.eye(3).repeat(1, 2, 1, 1)
    def step(height, planted, theta=0., vz=0.):
      env.common_step_counter += 1
      points[..., 2] = height
      velocity[..., 2] = vz
      rotation[:] = pitch(theta)
      contact.fill_(planted)
      return surface.sample()
    geometry = (points, velocity, rotation, torch.ones_like(points[..., 2], dtype=torch.bool), points[..., 2] * 0)
    with patch.object(surface, "geometry", return_value=geometry):
      step(.12, False)
      self.assertFalse(step(.0, True)["tracked_landing"].any())
      step(.01, False)
      control.fill_(.09)
      step(.07, False)
      self.assertTrue(step(.0, True)["tracked_landing"].all())
      torch.testing.assert_close(surface.swing_target, torch.tensor([[.04, .11]]))
      torch.testing.assert_close(surface.last_apex, torch.full_like(control, .07))
      self.assertTrue(surface.sample()["tracked_landing"].all())
      step(.03, False)
      sensor.last_air_time.fill_(.02)
      self.assertFalse(step(.0, True)["tracked_landing"].any())
      sensor.last_air_time.fill_(.14)
      # Heel first: 10 deg toe-up while descending through the last centimetre.
      step(.01, False)
      step(.08, False, -.17)
      self.assertEqual(lift.touchdown_tilt(env).item(), 0)  # high: outside the window
      step(.01, False, -.17, -.3)
      tilt = lift.touchdown_tilt(env)
      self.assertGreater(tilt.item(), 2 * 1.6)  # both feet, ~1.7 each
      s = step(.0, True, 0.)
      self.assertTrue(s["heel_first"].all())
      self.assertLess(lift.landing_pitch_sum_deg(env).item(), -19)
      # A level landing is free and not heel-first.
      step(.01, False)
      step(.08, False, 0.)
      step(.01, False, 0., -.3)
      self.assertEqual(lift.touchdown_tilt(env).item(), 0)
      self.assertFalse(step(.0, True)["heel_first"].any())
      # A swing that steps up 5 cm is a completed swing but is not scored for apex.
      step(.01, False)
      surface.last_bed[:, :, 7:] = .05
      step(.10, False)
      surface.last_bed[:] = .05
      s = step(.0, True)
      self.assertTrue(s["tracked_landing"].all() and not s["level_landing"].any())
      self.assertEqual(lift.swing_apex(env).item(), 0)
      step(.06, False)
      self.assertTrue(step(.0, True)["level_landing"].all())
      surface.last_bed.zero_()
      step(.03, False)
      twist.zero_()
      step(.08, False)
      self.assertFalse(step(.0, True)["tracked_landing"].any())
    for target in (.04, .075, .11):
      desired = torch.tensor(target)
      self.assertEqual(lift.apex_error(desired, desired).item(), 0.)
      self.assertLess(lift.apex_error(desired + .01, desired), lift.apex_error(desired + .03, desired))

  def test_lateral_tracking(self):
    command = torch.tensor([[0., .3, 0.], [0., 0., 0.]])
    env = NS(command_manager=NS(get_command=lambda _: command),
             scene={"robot": NS(data=NS(root_link_lin_vel_b=torch.tensor([[0., .3, 0.], [0., .5, 0.]])))})
    torch.testing.assert_close(lift.track_lateral_velocity(env, std=.5),
                               torch.tensor([1., math.exp(-1.)]))

  def test_touchdown_load_follows_robot_weight(self):
    # Two robots, bare (32.1 kg) and with 5.5 kg payload, both landing at 1.1 x their own weight.
    mass = torch.tensor([[32.1], [37.6]])
    peak = (1.1 * mass * 9.81).expand(2, 2)
    sensor = NS(current_contact_time=torch.full((2, 2), .02), last_air_time=torch.full((2, 2), .2),
                force_history=torch.stack((torch.zeros(2, 2, 4), torch.zeros(2, 2, 4), peak[..., None].expand(2, 2, 4)), -1))
    env = NS(scene={"feet_ground_contact": NS(data=sensor), "robot": NS(data=NS(indexing=NS(body_ids=torch.tensor([0]))))},
             sim=NS(model=NS(body_mass=mass)),
             command_manager=NS(get_term=lambda _: NS(moving=torch.ones(2, dtype=torch.bool))))
    cost = lift.touchdown_load(env)
    torch.testing.assert_close(cost, 2 * .1 * mass[:, 0] * 9.81 / 150)
    sensor.force_history[..., 2] = mass[:, :, None] * 9.81 * .99  # below body weight: free
    self.assertEqual(lift.touchdown_load(env).abs().sum().item(), 0.)

  def test_yaw_tracking_keeps_roll_pitch_factor(self):
    from src.tasks.common.mdp.rewards import track_angular_velocity
    w = torch.tensor([[.3, -.2, .5], [0., 0., .1]])
    command = torch.tensor([[0., 0., .2], [0., 0., .1]])
    env = NS(command_manager=NS(get_command=lambda _: command),
             scene={"robot": NS(data=NS(root_link_ang_vel_b=w))})
    old = track_angular_velocity(env, std=math.sqrt(.5), command_name="twist")
    new = lift.track_yaw_rate(env, std=.5, xy_std=math.sqrt(.5) / math.sqrt(.05))
    xy = w[:, :2].square().sum(-1)
    # Same roll/pitch factor as the 47k term; yaw kernel exp(-4 e^2) instead of exp(-2 e^2).
    torch.testing.assert_close(new / old, torch.exp(-4 * (w[:, 2] - command[:, 2]) ** 2)
                               / torch.exp(-2 * (w[:, 2] - command[:, 2]) ** 2))
    torch.testing.assert_close(new[1], torch.tensor(1.))
    self.assertGreater(xy[0].item(), 0)

  def test_registered_scratch_contract(self):
    cfg = load_env_cfg(TASK)
    base = load_env_cfg(BASE)
    self.assertNotIn("clearance", base.commands)
    self.assertEqual(cfg.actions["joint_pos"], base.actions["joint_pos"])
    self.assertEqual(cfg.actions["upper_body_pose"], base.actions["upper_body_pose"])
    self.assertEqual(cfg.commands["base_height"].floor_stages[-1][1], .22)
    self.assertEqual(cfg.commands["lean"].max_lean_stages[0][1], 0.)
    self.assertEqual(cfg.commands["lean"].max_lean_stages[-1][1], 1.25)
    # Every event (spawn, friction, pushes, payloads, delays, gains) is the 47k task's,
    # plus turning seam spawns along the seam, after reset_base.
    events = dict(cfg.events)
    self.assertIs(events.pop("face_along_seams").func, lift.face_along_seams)
    self.assertEqual(events, base.events)
    names = list(cfg.events)
    self.assertGreater(names.index("face_along_seams"), names.index("reset_base"))
    for key in ("terrain_levels", "command_vel"):
      self.assertEqual(cfg.curriculum[key], base.curriculum[key])
    surface, old = cfg.actions["poultry_surface"], base.actions["poultry_surface"]
    self.assertEqual((surface.drag_strength_range, surface.max_drag_force),
                     (old.drag_strength_range, old.max_drag_force))
    self.assertEqual(cfg.curriculum["litter_depth"].params["stages"][-1], (LOOSE_DEPTH_STAGES[-1][0] * ITER, .08))
    generator, old_generator = cfg.scene.terrain.terrain_generator, base.scene.terrain.terrain_generator
    self.assertEqual((generator.size, generator.num_rows, generator.num_cols, generator.curriculum),
                     (old_generator.size, old_generator.num_rows, old_generator.num_cols, True))
    for name, sub in old_generator.sub_terrains.items():
      if name not in ("flat", "soft_floor"):
        self.assertEqual(generator.sub_terrains[name], sub)
    self.assertAlmostEqual(sum(TILE_PROPORTIONS.values()), old_generator.sub_terrains["flat"].proportion
                           + old_generator.sub_terrains["soft_floor"].proportion)
    self.assertNotIn("foot_clearance", cfg.rewards)
    self.assertNotIn("weight_poultry_foot_level", cfg.curriculum)
    self.assertIsInstance(cfg.commands["twist"], lift.PoultryTwistCfg)
    self.assertIs(cfg.rewards["track_angular_velocity"].func, lift.track_yaw_rate)
    self.assertEqual(cfg.rewards["track_angular_velocity"].params["std"], .5)
    self.assertEqual(cfg.observations["actor"].history_length, 6)
    self.assertEqual(list(cfg.observations["actor"].terms), [*base.observations["actor"].terms, "clearance_commands"])
    self.assertEqual(list(cfg.observations["critic"].terms),
                     [*base.observations["critic"].terms, "clearance_commands", "litter_truth", "pull_state"])
    # Stand-still terms let go during strong pulls; weights and parameters unchanged.
    for name in ("idle_position_anchor", "idle_feet_still", "idle_base_motion", "stand_feet_square",
                 "pose", "track_linear_velocity"):
      self.assertIs(cfg.rewards[name].func.base, base.rewards[name].func)
      self.assertEqual((cfg.rewards[name].weight, cfg.rewards[name].params),
                       (base.rewards[name].weight, base.rewards[name].params))
    self.assertEqual(cfg.curriculum["pull_scale"].params["stages"][-1][1], 1.)
    self.assertEqual(_TERM_RULES["pull_state"][1], [1.0, -1.0, 1.0, 1.0])
    for name in ("poultry_clearance", "poultry_swing_height", "poultry_plow", "leg_power",
                 "touchdown_load", "touchdown_tilt"):
      self.assertEqual(cfg.rewards[name].weight, 0.)
      self.assertLess(cfg.curriculum["weight_" + name].params["stages"][-1][1], 0.)
    self.assertEqual(cfg.curriculum["weight_touchdown_tilt"].params["stages"][-1][1], -2.)
    runner = load_rl_cfg(TASK)
    self.assertFalse(runner.resume)
    self.assertEqual(runner.experiment_name, "Adaptive_Lift")
    self.assertGreater(runner.max_iterations, cfg.commands["lean"].max_lean_stages[-1][0] // ITER)
