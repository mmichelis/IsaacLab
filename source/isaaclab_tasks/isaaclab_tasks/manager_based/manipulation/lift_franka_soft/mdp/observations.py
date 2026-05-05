# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation functions for the Franka deformable lifting environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, DeformableObject
    from isaaclab.envs import ManagerBasedRLEnv


def deformable_com_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Position of the deformable object's COM in the robot's root frame [m].

    The COM is the mean of the deformable's nodal positions (see
    :attr:`~isaaclab.assets.DeformableObject.data.root_pos_w`).

    Returns:
        Tensor of shape ``(num_envs, 3)``.
    """
    asset: DeformableObject = env.scene[asset_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    com_w = wp.to_torch(asset.data.root_pos_w)
    com_b, _ = subtract_frame_transforms(wp.to_torch(robot.data.root_pos_w), wp.to_torch(robot.data.root_quat_w), com_w)
    return com_b
