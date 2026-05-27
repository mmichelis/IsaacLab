# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _sanitize_tensor(value: torch.Tensor, limit: float) -> torch.Tensor:
    """Return a finite tensor clipped to the configured absolute limit."""
    return torch.nan_to_num(value, nan=0.0, posinf=limit, neginf=-limit).clamp(min=-limit, max=limit)


def _newton_state_views(env: ManagerBasedEnv) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return four-bar body and joint state tensors."""
    from isaaclab_newton.physics import NewtonManager  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415

    state = NewtonManager.get_state_0()
    body_pos = wp.to_torch(state.body_q).view(env.num_envs, -1, 7)
    body_vel = wp.to_torch(state.body_qd).view(env.num_envs, -1, 6)
    joint_pos = wp.to_torch(state.joint_q).view(env.num_envs, -1)
    joint_vel = wp.to_torch(state.joint_qd).view(env.num_envs, -1)
    return body_pos, body_vel, joint_pos, joint_vel


def four_bar_body_pos_rel(env: ManagerBasedEnv) -> torch.Tensor:
    """Body positions relative to each environment origin [m]."""
    body_pos, _, _, _ = _newton_state_views(env)
    return _sanitize_tensor(body_pos[:, :, :3] - env.scene.env_origins[:, None, :], limit=10.0).flatten(start_dim=1)


def four_bar_body_lin_vel(env: ManagerBasedEnv) -> torch.Tensor:
    """Body linear velocities in world frame [m/s]."""
    _, body_vel, _, _ = _newton_state_views(env)
    return _sanitize_tensor(body_vel[:, :, :3], limit=100.0).flatten(start_dim=1)


def four_bar_body_ang_vel(env: ManagerBasedEnv) -> torch.Tensor:
    """Body angular velocities in world frame [rad/s]."""
    _, body_vel, _, _ = _newton_state_views(env)
    return _sanitize_tensor(body_vel[:, :, 3:], limit=100.0).flatten(start_dim=1)


def four_bar_joint_pos(env: ManagerBasedEnv) -> torch.Tensor:
    """Four-bar joint positions [rad]."""
    _, _, joint_pos, _ = _newton_state_views(env)
    return _sanitize_tensor(joint_pos, limit=10.0)


def four_bar_joint_vel(env: ManagerBasedEnv) -> torch.Tensor:
    """Four-bar joint velocities [rad/s]."""
    _, _, _, joint_vel = _newton_state_views(env)
    return _sanitize_tensor(joint_vel, limit=100.0)


def gait_phase(env: ManagerBasedEnv, period_s: float) -> torch.Tensor:
    """Periodic gait phase signal."""
    phase = (
        2.0
        * torch.pi
        * env.episode_length_buf.to(env.device, dtype=torch.float32)
        * env.step_dt
        / period_s
    )
    return torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1)
