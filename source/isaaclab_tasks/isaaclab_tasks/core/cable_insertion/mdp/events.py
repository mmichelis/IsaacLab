# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset events for cable insertion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import CableObject, RigidObject
    from isaaclab.envs import ManagerBasedEnv


def reset_cable_from_object(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
) -> None:
    """Reset the cable shape while keeping its attached end aligned with the object."""
    cable: CableObject = env.scene[cable_cfg.name]
    object_asset: RigidObject = env.scene[object_cfg.name]
    segment_pose = cable.data.default_segment_pose_w.torch[env_ids].clone()
    segment_velocity = cable.data.default_segment_velocity_w.torch[env_ids].clone()
    default_object_pos = object_asset.data.default_root_pose.torch[env_ids, :3] + env.scene.env_origins[env_ids]
    object_offset = object_asset.data.root_link_pose_w.torch[env_ids, :3] - default_object_pos
    segment_pose[..., :3] += object_offset.unsqueeze(1)
    cable.write_segment_pose_to_sim_index(segment_pose=segment_pose, env_ids=env_ids)
    cable.write_segment_velocity_to_sim_index(segment_velocity=segment_velocity, env_ids=env_ids)
