# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv


def asset_pose_b(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    reference_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return an asset root pose in a reference asset's root frame.

    Args:
        env: The environment.
        asset_cfg: Asset whose root pose is observed.
        reference_cfg: Reference asset.

    Returns:
        Pose ``(x, y, z, qx, qy, qz, qw)`` with shape ``(num_envs, 7)``. Position is in [m].
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    reference: Articulation | RigidObject = env.scene[reference_cfg.name]
    position, orientation = subtract_frame_transforms(
        reference.data.root_pos_w.torch,
        reference.data.root_quat_w.torch,
        asset.data.root_pos_w.torch,
        asset.data.root_quat_w.torch,
    )
    return torch.cat((position, orientation), dim=-1)
