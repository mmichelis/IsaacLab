# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def set_four_bar_material(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    friction: float,
    restitution: float,
) -> None:
    """Set Newton shape contact material for the four-bar scene.

    Args:
        env: The manager-based environment.
        env_ids: Unused environment indices.
        friction: Shape friction coefficient.
        restitution: Shape restitution coefficient.
    """
    from isaaclab_newton.physics import NewtonManager  # noqa: PLC0415
    from newton.solvers import SolverNotifyFlags  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415

    model = NewtonManager.get_model()
    if model is None or model.shape_material_mu is None:
        return

    wp.to_torch(model.shape_material_mu).fill_(friction)
    if model.shape_material_restitution is not None:
        wp.to_torch(model.shape_material_restitution).fill_(restitution)
    NewtonManager.add_model_change(SolverNotifyFlags.SHAPE_PROPERTIES)


def reset_four_bar_configuration(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    joint_angle_range: tuple[float, float],
    joint_velocity_range: tuple[float, float],
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
) -> None:
    """Reset the linkage to a random closed-chain configuration.

    The joint configuration is sampled as a parallelogram pattern
    ``[theta, -theta, theta, -theta]`` so the four-bar starts near a feasible
    closed-loop pose.

    Args:
        env: The manager-based environment.
        env_ids: Environment indices to reset.
        joint_angle_range: Range for the sampled linkage angle [rad].
        joint_velocity_range: Range for sampled joint velocities [rad/s].
        pose_range: Root pose offsets. Position keys are ``x``, ``y``, ``z`` [m],
            and rotation keys are ``roll``, ``pitch``, ``yaw`` [rad].
        velocity_range: Root velocity offsets. Linear keys are ``x``, ``y``, ``z`` [m/s],
            and angular keys are ``roll``, ``pitch``, ``yaw`` [rad/s].
    """
    from isaaclab_newton.physics import NewtonManager  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415

    state = NewtonManager.get_state_0()
    solver = NewtonManager._solver
    if solver is None:
        return

    joint_pos = wp.to_torch(state.joint_q).view(env.num_envs, -1)
    joint_vel = wp.to_torch(state.joint_qd).view(env.num_envs, -1)
    if joint_pos.shape[1] != 4:
        raise ValueError(f"reset_four_bar_configuration expects four joints, got {joint_pos.shape[1]}.")

    theta = math_utils.sample_uniform(*joint_angle_range, (len(env_ids), 1), device=env.device)
    joint_pos[env_ids] = torch.cat([theta, -theta, theta, -theta], dim=-1)
    joint_vel[env_ids] = math_utils.sample_uniform(*joint_velocity_range, (len(env_ids), 4), device=env.device)

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=env.device)
    pose_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device)

    base_pose = torch.zeros((env.num_envs, 7), device=env.device)
    base_pose[:, 6] = 1.0
    base_pose[env_ids, :3] = env.scene.env_origins[env_ids] + pose_samples[:, :3]
    base_pose[env_ids, 3:7] = math_utils.quat_from_euler_xyz(
        pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    )

    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=env.device)
    velocity_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device)

    base_vel = torch.zeros((env.num_envs, 6), device=env.device)
    base_vel[env_ids, :3] = velocity_samples[:, :3]
    base_vel[env_ids, 3:] = velocity_samples[:, 3:]

    world_mask = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    world_mask[env_ids] = 1
    solver.reset(
        state_out=state,
        world_mask=wp.from_torch(world_mask, dtype=wp.int32),
        joint_q=state.joint_q,
        joint_u=state.joint_qd,
        base_q=wp.from_torch(base_pose, dtype=wp.transformf),
        base_u=wp.from_torch(base_vel, dtype=wp.spatial_vectorf),
    )

    if hasattr(env, "_four_bar_previous_center_x"):
        body_pos = wp.to_torch(state.body_q).view(env.num_envs, -1, 7)
        center_x = body_pos[:, :, 0].mean(dim=1) - env.scene.env_origins[:, 0]
        env._four_bar_previous_center_x[env_ids] = torch.nan_to_num(
            center_x[env_ids], nan=0.0, posinf=10.0, neginf=-10.0
        ).clamp(min=-10.0, max=10.0)
