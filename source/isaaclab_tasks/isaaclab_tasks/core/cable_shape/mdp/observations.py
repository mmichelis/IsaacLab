# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import CableObject
    from isaaclab.envs import ManagerBasedEnv


def cable_segment_positions_in_env_frame(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("cable")
) -> torch.Tensor:
    """Flattened cable segment positions in the environment frame [m]."""
    asset: CableObject = env.scene[asset_cfg.name]
    positions = asset.data.segment_pose_w.torch[..., :3] - env.scene.env_origins.unsqueeze(1)
    return positions.flatten(1)


def cable_segment_velocities(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("cable")
) -> torch.Tensor:
    """Flattened world-frame cable segment velocities [m/s, rad/s]."""
    asset: CableObject = env.scene[asset_cfg.name]
    return asset.data.segment_velocity_w.torch.flatten(1)
