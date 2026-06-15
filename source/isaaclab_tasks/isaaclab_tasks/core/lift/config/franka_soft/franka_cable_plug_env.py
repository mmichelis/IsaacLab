# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cable-plug environment with a guard against non-finite solver output."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import VecEnvStepReturn


class FrankaCablePlugEnv(ManagerBasedRLEnv):
    """Cable-plug env that keeps NaN/Inf solver blow-ups from reaching the RL algorithm.

    The coupled MJWarp + VBD solve can diverge to NaN/Inf during plug-grasp contact. The
    ``velocity_divergence`` termination catches such envs (it tests finiteness of every state
    that the observations are built from) and resets them, so their observations are recomputed
    clean. The reward, however, is computed from the pre-reset state in
    :meth:`ManagerBasedRLEnv.step`, so the diverging step still returns a NaN reward. This
    subclass zeros those non-finite rewards so no NaN reaches the learner.
    """

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # The no-cable reset places the plug at the gripper, read from the articulation's forward
        # kinematics (see ``reset_plug_uniform``). FK is only live after the simulation's first
        # ``forward``, so a throwaway reset here warms it and the first user reset places the plug
        # at the gripper rather than at the (arm-up) spawn pose.
        if not cfg.with_cable:
            self.reset()

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        obs, reward, terminated, time_outs, extras = super().step(action)
        # The diverged env is reset by the divergence termination, but its reward was computed
        # before that reset; zero any non-finite values so the policy update stays well-defined.
        torch.nan_to_num_(reward, nan=0.0, posinf=0.0, neginf=0.0)
        # Guard observations too: a non-finite obs that slips past the divergence reset would
        # poison the policy/critic gradients (and the action std) and crash the learner.
        for group in obs.values():
            torch.nan_to_num_(group, nan=0.0, posinf=0.0, neginf=0.0)
        return obs, reward, terminated, time_outs, extras
