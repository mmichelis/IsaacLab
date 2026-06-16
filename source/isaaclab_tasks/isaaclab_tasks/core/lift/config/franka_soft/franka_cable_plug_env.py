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
        if not cfg.with_cable:
            self._cache_plug_grasp_frame()

    def _cache_plug_grasp_frame(self) -> None:
        """Cache the gripper grasp frame at the default arm config for the no-cable plug reset.

        The plug spawns centered on the gripper (see :func:`mdp.reset_plug_uniform`), but the arm
        always resets to the same default joints, so the grasp frame is a per-env constant. The
        coupled solver only refreshes the arm FK after a real step (not during a reset event), so
        drive the arm to its default config and step once to make the FK current, then cache the
        ``panda_hand`` pose. Without this the event would read the stale pre-reset arm pose and
        spawn the plug wherever the arm wandered during the previous episode.
        """
        robot = self.scene["robot"]
        env_ids = torch.arange(self.num_envs, device=self.device)
        robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch, env_ids=env_ids)
        self.step(torch.zeros(self.num_envs, self.action_manager.total_action_dim, device=self.device))
        hand_idx = robot.find_bodies("panda_hand")[0][0]
        self.plug_grasp_hand_pos_w = robot.data.body_link_pos_w.torch[:, hand_idx].clone()
        self.plug_grasp_hand_quat_w = robot.data.body_link_quat_w.torch[:, hand_idx].clone()

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
