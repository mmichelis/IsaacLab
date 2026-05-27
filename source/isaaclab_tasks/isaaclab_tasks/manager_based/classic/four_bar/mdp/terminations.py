# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .observations import _newton_state_views

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def invalid_newton_state(
    env: ManagerBasedRLEnv,
    max_body_distance: float = 10.0,
    max_body_velocity: float = 100.0,
    max_joint_velocity: float = 100.0,
) -> torch.Tensor:
    """Terminate envs whose Newton state is non-finite or outside conservative bounds.

    Args:
        env: The manager-based RL environment.
        max_body_distance: Maximum body distance from the environment origin [m].
        max_body_velocity: Maximum absolute body linear or angular velocity [m/s or rad/s].
        max_joint_velocity: Maximum absolute joint velocity [rad/s].

    Returns:
        Boolean tensor indicating environments with invalid state.
    """
    body_pos, body_vel, joint_pos, joint_vel = _newton_state_views(env)
    body_pos_rel = body_pos[:, :, :3] - env.scene.env_origins[:, None, :]

    non_finite = (
        ~torch.isfinite(body_pos).flatten(start_dim=1).all(dim=1)
        | ~torch.isfinite(body_vel).flatten(start_dim=1).all(dim=1)
        | ~torch.isfinite(joint_pos).flatten(start_dim=1).all(dim=1)
        | ~torch.isfinite(joint_vel).flatten(start_dim=1).all(dim=1)
    )
    body_too_far = body_pos_rel.abs().flatten(start_dim=1).amax(dim=1) > max_body_distance
    body_too_fast = body_vel.abs().flatten(start_dim=1).amax(dim=1) > max_body_velocity
    joints_too_fast = joint_vel.abs().flatten(start_dim=1).amax(dim=1) > max_joint_velocity

    return non_finite | body_too_far | body_too_fast | joints_too_fast
