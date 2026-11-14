"""Egg-tray transport: an arm-only residual RL stabiliser on top of frozen DUET.

The DUET checkpoint ``poultry_v3_squat_91500`` keeps sole control of the 12 leg
joints and waist_yaw (``frozen_duet.py``); a new policy outputs bounded
residuals on the 10 arm joints around a carrying pose (``mdp/actions.py``) to
keep a tray of individually simulated eggs steady while DUET walks on the
poultry-litter floor. See ``docs/eggtray_rl.md``.
"""
