# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _four_bar_center_x(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the mean body x-position relative to each environment origin [m]."""
    from isaaclab_newton.physics import NewtonManager  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415

    state = NewtonManager.get_state_0()
    body_pos = wp.to_torch(state.body_q).view(env.num_envs, -1, 7)
    center_x = body_pos[:, :, 0].mean(dim=1) - env.scene.env_origins[:, 0]
    return torch.nan_to_num(center_x, nan=0.0, posinf=10.0, neginf=-10.0).clamp(min=-10.0, max=10.0)


def root_lin_vel_x(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward mean body velocity along the world x-axis [m/s].

    Args:
        env: The manager-based RL environment.

    Returns:
        The mean four-bar body linear velocity along world x [m/s], shape ``(num_envs,)``.
    """
    from isaaclab_newton.physics import NewtonManager  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415

    state = NewtonManager.get_state_0()
    body_vel = wp.to_torch(state.body_qd).view(env.num_envs, -1, 6)
    reward = body_vel[:, :, 0].mean(dim=1)
    return torch.nan_to_num(reward, nan=0.0, posinf=100.0, neginf=-100.0).clamp(min=-100.0, max=100.0)


def body_progress_x(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward center-of-linkage progress along the world x-axis [m/s].

    Args:
        env: The manager-based RL environment.

    Returns:
        The finite difference center-of-linkage x velocity [m/s], shape ``(num_envs,)``.
    """
    center_x = _four_bar_center_x(env)
    if not hasattr(env, "_four_bar_previous_center_x") or env._four_bar_previous_center_x.shape != center_x.shape:
        env._four_bar_previous_center_x = center_x.clone()
        return torch.zeros_like(center_x)

    progress = (center_x - env._four_bar_previous_center_x) / env.step_dt
    env._four_bar_previous_center_x[:] = center_x
    return torch.nan_to_num(progress, nan=0.0, posinf=10.0, neginf=-10.0).clamp(min=-10.0, max=10.0)


def gait_action_alignment(env: ManagerBasedRLEnv, period_s: float) -> torch.Tensor:
    """Reward actions that match the nominal one-cycle four-bar gait."""
    phase = (
        2.0
        * torch.pi
        * env.episode_length_buf.to(env.device, dtype=torch.float32)
        * env.step_dt
        / period_s
    )
    desired_actions = torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1)
    clipped_actions = torch.clamp(env.action_manager.action, min=-1.0, max=1.0)
    return (clipped_actions * desired_actions).sum(dim=-1)


def invalid_state_penalty(
    env: ManagerBasedRLEnv,
    max_body_distance: float = 10.0,
    max_body_velocity: float = 100.0,
    max_joint_velocity: float = 100.0,
) -> torch.Tensor:
    """Return one for envs with invalid Newton state."""
    from .terminations import invalid_newton_state  # noqa: PLC0415

    return invalid_newton_state(
        env,
        max_body_distance=max_body_distance,
        max_body_velocity=max_body_velocity,
        max_joint_velocity=max_joint_velocity,
    ).float()
