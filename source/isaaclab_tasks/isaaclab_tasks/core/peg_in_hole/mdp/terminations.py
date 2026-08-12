# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination functions for the rigid lift tasks.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def joint_vel_out_of_sim_limit(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when joint velocities exceed actuator simulator limits [m/s or rad/s, depending on joint type]."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
    limits = torch.full_like(asset.data.joint_vel.torch, torch.inf)
    for actuator in asset.actuators.values():
        limits[:, actuator.joint_indices] = actuator.velocity_limit_sim
    return torch.any(torch.abs(asset.data.joint_vel.torch[:, joint_ids]) > limits[:, joint_ids], dim=1)


def object_outside_bounds(
    env: ManagerBasedRLEnv,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Terminate if the rigid object's center leaves the workspace bounds.

    Args:
        env: The environment instance.
        x_bounds: Allowed x-position range in the environment frame [m].
        y_bounds: Allowed y-position range in the environment frame [m].
        z_bounds: Allowed z-position range in the environment frame [m].
        asset_cfg: The rigid object entity.

    Returns:
        Boolean tensor with shape ``(num_envs,)``.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    pos = asset.data.root_pos_w.torch[:, :3] - env.scene.env_origins[:, :3]
    outside_x = (pos[:, 0] < x_bounds[0]) | (pos[:, 0] > x_bounds[1])
    outside_y = (pos[:, 1] < y_bounds[0]) | (pos[:, 1] > y_bounds[1])
    outside_z = (pos[:, 2] < z_bounds[0]) | (pos[:, 2] > z_bounds[1])
    return outside_x | outside_y | outside_z
