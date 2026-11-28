import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import mujoco
import numpy as np
import torch
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from src.tasks.common.mdp.symmetry import _TERM_RULES
from src.tasks.duet.config.g1_23dof.adaptive_lift import TASK, TERRAIN_STAGES, reset_tile
from src.tasks.duet.duet_env_cfg import ITER
from src.tasks.duet.mdp import adaptive_lift as lift
from src.tasks.duet.mdp.lift_terrain import LAYOUT, MATERIALS, LiftTile, material_at
from src.tasks.duet.mdp.poultry_surface import litter_wrench, point_kinematics


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

  def test_terrain_stages_and_spatial_materials(self):
    terrain = NS(size=(12., 8.), num_rows=3, num_cols=4, terrain_origins=torch.randn(3, 4, 3),
                 env_origins=torch.empty(256, 3), terrain_types=torch.zeros(256, dtype=torch.long),
                 terrain_levels=torch.zeros(256, dtype=torch.long))
    env = NS(scene=NS(terrain=terrain), device="cpu")
    for stage, iteration in enumerate((0, *TERRAIN_STAGES)):
      env.common_step_counter = iteration * ITER
      reset_tile(env, torch.arange(256), False)
      self.assertEqual(terrain.terrain_types.unique().tolist(), list(range(stage + 1)))
      torch.testing.assert_close(terrain.env_origins, terrain.terrain_origins[1, terrain.terrain_types])
    points = torch.tensor([[0., -12.], [0., -4.], [0., 4.], [-5., 10.], [-1., 14.], [0., 20.]])
    material, valid = material_at(points, terrain, torch.as_tensor(LAYOUT))
    self.assertEqual(material.tolist(), [0, 1, 2, 0, 2, 0])
    self.assertEqual(valid.tolist(), [True] * 5 + [False])
    self.assertLess(max(m[1] for m in MATERIALS) * 1.2, .04)

  def test_material_query_matches_compiled_support(self):
    terrain = NS(size=(12., 8.), num_rows=1, num_cols=4)
    layout = torch.as_tensor(LAYOUT)
    # Cast a vertical ray at every material cell in each tile of the compiled floor.
    for col in range(4):
      spec = mujoco.MjSpec()
      spec.worldbody.add_body(name="terrain")
      tile = LiftTile(kind=col, proportion=1., size=terrain.size)
      tile.function(0., spec, np.random.default_rng(0))
      model = spec.compile()
      data = mujoco.MjData(model)
      mujoco.mj_forward(model, data)
      for i, j in np.ndindex(4, 2):
        origin = np.array([(i + .5) * 3, (j + .5) * 4, 1.])
        distance = mujoco.mj_ray(model, data, origin, np.array([0., 0., -1.]), None, True,
                                 -1, np.empty(1, dtype=np.int32))
        global_xy = torch.from_numpy(origin[:2] + np.array([-6., (col - 2) * 8.]))
        material, valid = material_at(global_xy, terrain, layout)
        self.assertTrue(valid)
        self.assertAlmostEqual(1. - distance, MATERIALS[material.item()][0])

  def test_sole_pitch_and_support_relative_drag(self):
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
    s = dict(layer=depth, clearance=points[..., 2]-bed-depth, velocity=vel, active=torch.tensor([[True]]))
    env = NS(action_manager=NS(get_term=lambda _: NS(sample=lambda: s)))
    self.assertGreater(lift.loose_plow(env).item(), 0)
    s["layer"] = torch.zeros_like(depth)
    self.assertEqual(lift.loose_plow(env).item(), 0)

  def test_apex_latches_once_and_ignores_spawn_and_idle(self):
    # Run the actual cached parent and child swing state machines with synthetic contact/geometry.
    surface = object.__new__(lift.LiftSurface)
    control = torch.tensor([[.04, .11]])
    twist = torch.tensor([[.4, 0., 0.]])
    contact = torch.zeros(1, 2, dtype=torch.bool)
    env = NS(scene={"feet_ground_contact": NS(data=NS(found=contact, last_air_time=torch.full_like(control, .14)))}, common_step_counter=0,
             _sim_step_counter=0, step_dt=.02, episode_length_buf=torch.zeros(1), device="cpu",
             command_manager=NS(get_command=lambda name: control if name == "clearance" else twist))
    surface._env = env
    surface.cfg = NS(period=.75, stance_fraction=.56)
    surface._sample_key = None
    for field in ("peak", "last_apex", "last_liftoff_angle", "swing_target"):
      setattr(surface, field, torch.zeros_like(control))
    for field in ("was_air", "complete_swing", "has_landed", "previous_contact", "tracking"):
      setattr(surface, field, torch.zeros_like(contact))
    points = torch.zeros(1, 2, 14, 3)
    def step(height, planted):
      env.common_step_counter += 1
      points[..., 2] = height
      contact.fill_(planted)
      return surface.sample()
    geometry = (points, points * 0, torch.eye(3).expand(1, 2, 3, 3),
                torch.ones_like(points[..., 2], dtype=torch.bool), points[..., 2] * 0)
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
      env.scene["feet_ground_contact"].data.last_air_time.fill_(.02)
      self.assertFalse(step(.0, True)["tracked_landing"].any())
      env.scene["feet_ground_contact"].data.last_air_time.fill_(.14)
      step(.03, False)
      twist.zero_()
      step(.08, False)
      self.assertFalse(step(.0, True)["tracked_landing"].any())
    for target in (.04, .075, .11):
      desired = torch.tensor(target)
      self.assertEqual(lift.apex_error(desired, desired).item(), 0.)
      self.assertLess(lift.apex_error(desired + .01, desired), lift.apex_error(desired + .03, desired))

  def test_registered_scratch_contract(self):
    cfg = load_env_cfg(TASK)
    base = load_env_cfg("Unitree-G1-23Dof-Duet-PoultrySurface-Lean")
    self.assertNotIn("clearance", base.commands)
    self.assertEqual(cfg.actions["joint_pos"], base.actions["joint_pos"])
    self.assertEqual(cfg.actions["upper_body_pose"], base.actions["upper_body_pose"])
    self.assertEqual(cfg.commands["base_height"].floor_stages[-1][1], .22)
    self.assertEqual(cfg.commands["lean"].max_lean_stages[0][1], 0.)
    self.assertEqual(cfg.commands["lean"].max_lean_stages[-1][1], 1.25)
    for key in ("hand_payload", "torso_payload", "hand_com", "actuator_delay"):
      self.assertEqual(cfg.events[key], base.events[key])
    self.assertNotIn("foot_clearance", cfg.rewards)
    self.assertNotIn("weight_poultry_foot_level", cfg.curriculum)
    self.assertNotIn("litter_depth", cfg.curriculum)
    self.assertEqual(cfg.observations["actor"].history_length, 6)
    for group in ("actor", "critic"):
      self.assertEqual(list(cfg.observations[group].terms), [*base.observations[group].terms, "clearance_commands"])
    for name in ("poultry_clearance", "poultry_swing_height", "poultry_plow", "leg_power", "touchdown_load"):
      self.assertEqual(cfg.rewards[name].weight, 0.)
      self.assertLess(cfg.curriculum["weight_" + name].params["stages"][-1][1], 0.)
    runner = load_rl_cfg(TASK)
    self.assertFalse(runner.resume)
    self.assertEqual(runner.experiment_name, "Adaptive_Lift")
    self.assertGreater(runner.max_iterations, cfg.commands["lean"].max_lean_stages[-1][0] // ITER)
