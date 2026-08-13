# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.envs import ManagerBasedEnv

from isaaclab_tasks.core.lift.mdp.events import grasp_travel_distance

from .events_cfg import GraspTravelOpeningCfg


class grasp_travel_opening(grasp_travel_distance):
    """Measure grasp distance, travel distance, and total gripper opening."""

    cfg: GraspTravelOpeningCfg

    def __init__(self, cfg: GraspTravelOpeningCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._gripper_joint_ids = self._robot.find_joints(cfg.gripper_joint_names)[0]

    def __call__(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> torch.Tensor:
        feature = super().__call__(env, env_ids)
        opening = self._robot.data.joint_pos.torch[env_ids][:, self._gripper_joint_ids].abs().sum(-1, keepdim=True)
        return torch.cat([feature, opening], dim=-1)
