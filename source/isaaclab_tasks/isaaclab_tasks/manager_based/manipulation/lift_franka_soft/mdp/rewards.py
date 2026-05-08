# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward and termination functions for the Franka deformable lifting environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

from .observations import scripted_grasp_action_target

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, DeformableObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import FrameTransformer


def _finite_reward(reward: torch.Tensor, replacement: float = 0.0) -> torch.Tensor:
    """Replace non-finite reward values with a bounded fallback."""
    return torch.nan_to_num(reward, nan=replacement, posinf=replacement, neginf=replacement)


def _deformable_nearest_ee_distance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Distance from the end-effector frame to the nearest deformable node [m]."""
    asset: DeformableObject = env.scene[asset_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    nodal_pos_w = wp.to_torch(asset.data.nodal_pos_w)
    ee_w = wp.to_torch(ee_frame.data.target_pos_w)[..., 0, :]
    return torch.linalg.norm(nodal_pos_w - ee_w.unsqueeze(1), dim=2).min(dim=1).values


def _deformable_com_ee_distance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    grasp_height_offset: float = 0.0,
) -> torch.Tensor:
    """Distance from the end-effector frame to the deformable COM [m]."""
    asset: DeformableObject = env.scene[asset_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    com_w = wp.to_torch(asset.data.root_pos_w).clone()
    com_w[:, 2] += grasp_height_offset
    ee_w = wp.to_torch(ee_frame.data.target_pos_w)[..., 0, :]
    return torch.linalg.norm(com_w - ee_w, dim=1)


def _gripper_is_closing(env: ManagerBasedRLEnv, action_name: str) -> torch.Tensor:
    """Whether the binary gripper action is commanding closure."""
    gripper_action = env.action_manager.get_term(action_name).raw_actions
    return torch.any(gripper_action < 0.0, dim=1).float()


def deformable_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> torch.Tensor:
    """Reward if the deformable COM is above a minimum height.

    Args:
        env: The environment instance.
        minimal_height: Minimum COM height [m].
        asset_cfg: The deformable object entity.

    Returns:
        Reward tensor with shape ``(num_envs,)``.
    """
    asset: DeformableObject = env.scene[asset_cfg.name]
    com_z = wp.to_torch(asset.data.root_pos_w)[:, 2]
    return _finite_reward(torch.where(com_z > minimal_height, 1.0, 0.0))


def deformable_lift_height(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    height_scale: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> torch.Tensor:
    """Reward deformable COM height above a minimum height.

    Args:
        env: The environment instance.
        minimal_height: Height at which lift credit starts [m].
        height_scale: Height span used to normalize the reward [m].
        asset_cfg: The deformable object entity.

    Returns:
        Reward tensor with shape ``(num_envs,)``.
    """
    asset: DeformableObject = env.scene[asset_cfg.name]
    com_z = wp.to_torch(asset.data.root_pos_w)[:, 2]
    return _finite_reward(torch.clamp((com_z - minimal_height) / height_scale, min=0.0, max=1.0))


def deformable_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    grasp_height_offset: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward reaching a grasp point near the deformable COM with the end-effector.

    Args:
        env: The environment instance.
        std: The tanh kernel standard deviation [m].
        grasp_height_offset: Vertical offset from the COM to the grasp point [m].
        asset_cfg: The deformable object entity.
        ee_frame_cfg: The end-effector frame entity.

    Returns:
        Reward tensor with shape ``(num_envs,)``.
    """
    distance = _deformable_com_ee_distance(env, asset_cfg, ee_frame_cfg, grasp_height_offset)
    return _finite_reward(1.0 - torch.tanh(distance / std))


def deformable_nearest_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward reaching the nearest deformable node with the end-effector.

    Args:
        env: The environment instance.
        std: The tanh kernel standard deviation [m].
        asset_cfg: The deformable object entity.
        ee_frame_cfg: The end-effector frame entity.

    Returns:
        Reward tensor with shape ``(num_envs,)``.
    """
    distance = _deformable_nearest_ee_distance(env, asset_cfg, ee_frame_cfg)
    return _finite_reward(1.0 - torch.tanh(distance / std))


def gripper_close_near_deformable(
    env: ManagerBasedRLEnv,
    std: float,
    far_penalty_scale: float,
    grasp_height_offset: float = 0.0,
    action_name: str = "gripper_action",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward closing the gripper near the deformable and penalize closing far away.

    Args:
        env: The environment instance.
        std: The tanh kernel standard deviation [m].
        far_penalty_scale: Scale applied to the far-from-object close penalty.
        grasp_height_offset: Vertical offset from the COM to the grasp point [m].
        action_name: Name of the gripper action term.
        asset_cfg: The deformable object entity.
        ee_frame_cfg: The end-effector frame entity.

    Returns:
        Reward tensor with shape ``(num_envs,)``.
    """
    distance = _deformable_com_ee_distance(env, asset_cfg, ee_frame_cfg, grasp_height_offset)
    near = 1.0 - torch.tanh(distance / std)
    closing = _gripper_is_closing(env, action_name)
    return _finite_reward(closing * (near - far_penalty_scale * (1.0 - near)))


def gripper_goal_near_deformable(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    grasp_height_offset: float = 0.0,
    action_name: str = "gripper_action",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward moving the closed gripper toward the goal while it remains near the deformable.

    Args:
        env: The environment instance.
        std: The tanh kernel standard deviation [m].
        command_name: Name of the generated pose command.
        grasp_height_offset: Vertical offset from the COM to the grasp point [m].
        action_name: Name of the gripper action term.
        robot_cfg: The robot entity providing the root frame.
        asset_cfg: The deformable object entity.
        ee_frame_cfg: The end-effector frame entity.

    Returns:
        Reward tensor with shape ``(num_envs,)``.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    command = env.command_manager.get_command(command_name)
    goal_w, _ = combine_frame_transforms(
        wp.to_torch(robot.data.root_pos_w), wp.to_torch(robot.data.root_quat_w), command[:, :3]
    )
    ee_w = wp.to_torch(ee_frame.data.target_pos_w)[..., 0, :]
    goal_distance = torch.linalg.norm(goal_w - ee_w, dim=1)
    object_distance = _deformable_com_ee_distance(env, asset_cfg, ee_frame_cfg, grasp_height_offset)
    near = 1.0 - torch.tanh(object_distance / std)
    closing = _gripper_is_closing(env, action_name)
    return _finite_reward(closing * near * (1.0 - torch.tanh(goal_distance / std)))


def gripper_lift_near_deformable(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    height_scale: float,
    grasp_height_offset: float = 0.0,
    action_name: str = "gripper_action",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward lifting the closed gripper while it remains near the deformable.

    Args:
        env: The environment instance.
        std: The tanh kernel standard deviation [m].
        minimal_height: End-effector height at which lift credit starts [m].
        height_scale: Height span used to normalize the lift credit [m].
        grasp_height_offset: Vertical offset from the COM to the grasp point [m].
        action_name: Name of the gripper action term.
        asset_cfg: The deformable object entity.
        ee_frame_cfg: The end-effector frame entity.

    Returns:
        Reward tensor with shape ``(num_envs,)``.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_z = wp.to_torch(ee_frame.data.target_pos_w)[..., 0, 2] - env.scene.env_origins[:, 2]
    lift = torch.clamp((ee_z - minimal_height) / height_scale, min=0.0, max=1.0)
    distance = _deformable_com_ee_distance(env, asset_cfg, ee_frame_cfg, grasp_height_offset)
    near = 1.0 - torch.tanh(distance / std)
    closing = _gripper_is_closing(env, action_name)
    return _finite_reward(closing * near * lift)


def end_effector_lift_action_near_deformable(
    env: ManagerBasedRLEnv,
    std: float,
    action_scale_z: float,
    grasp_height_offset: float = 0.0,
    action_name: str = "arm_action",
    gripper_action_name: str = "gripper_action",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward upward IK commands while the closed gripper is near the deformable.

    Args:
        env: The environment instance.
        std: The tanh kernel standard deviation for gripper-to-object distance [m].
        action_scale_z: Maximum processed upward IK displacement used for normalization [m].
        grasp_height_offset: Vertical offset from the COM to the grasp point [m].
        action_name: Name of the arm action term.
        gripper_action_name: Name of the gripper action term.
        asset_cfg: The deformable object entity.
        ee_frame_cfg: The end-effector frame entity.

    Returns:
        Reward tensor with shape ``(num_envs,)``.
    """
    distance = _deformable_com_ee_distance(env, asset_cfg, ee_frame_cfg, grasp_height_offset)
    near = 1.0 - torch.tanh(distance / std)
    closing = _gripper_is_closing(env, gripper_action_name)
    lift_action = env.action_manager.get_term(action_name).processed_actions[:, 2]
    upward = torch.clamp(lift_action / action_scale_z, min=0.0, max=1.0)
    return _finite_reward(closing * near * upward)


def end_effector_action_to_deformable(
    env: ManagerBasedRLEnv,
    std: float,
    grasp_height_offset: float = 0.0,
    action_name: str = "arm_action",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward horizontal end-effector commands that point toward the deformable grasp point.

    Args:
        env: The environment instance.
        std: Distance scale used to fade this reward near the grasp point [m].
        grasp_height_offset: Vertical offset from the COM to the grasp point [m].
        action_name: Name of the arm action term.
        asset_cfg: The deformable object entity.
        ee_frame_cfg: The end-effector frame entity.

    Returns:
        Reward tensor with shape ``(num_envs,)``.
    """
    asset: DeformableObject = env.scene[asset_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    grasp_w = wp.to_torch(asset.data.root_pos_w).clone()
    grasp_w[:, 2] += grasp_height_offset
    ee_w = wp.to_torch(ee_frame.data.target_pos_w)[..., 0, :]
    to_grasp = grasp_w - ee_w
    to_grasp[:, 2] = 0.0
    distance = torch.linalg.norm(to_grasp, dim=1)

    arm_action = env.action_manager.get_term(action_name).processed_actions[:, :3].clone()
    arm_action[:, 2] = 0.0
    action_norm = torch.linalg.norm(arm_action, dim=1)
    alignment = torch.sum(arm_action * to_grasp, dim=1) / (torch.clamp(action_norm * distance, min=1e-6))
    alignment = torch.clamp(alignment, min=0.0, max=1.0)
    return _finite_reward(alignment * torch.tanh(distance / std))


def end_effector_grasp_command_tracking(
    env: ManagerBasedRLEnv,
    std: float,
    action_scale: tuple[float, float, float],
    fade_std: float,
    grasp_height_offset: float = 0.0,
    action_name: str = "arm_action",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward IK position commands that move directly toward the deformable grasp point.

    The desired command is the clamped per-step displacement from the end-effector to
    the grasp point. The reward fades out near the object so it does not penalize the
    later lift motion.

    Args:
        env: The environment instance.
        std: The tanh kernel standard deviation for command error [m].
        action_scale: Per-axis maximum processed IK displacement [m].
        fade_std: Distance scale used to fade the reward near the grasp point [m].
        grasp_height_offset: Vertical offset from the COM to the grasp point [m].
        action_name: Name of the arm action term.
        asset_cfg: The deformable object entity.
        ee_frame_cfg: The end-effector frame entity.

    Returns:
        Reward tensor with shape ``(num_envs,)``.
    """
    asset: DeformableObject = env.scene[asset_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    grasp_w = wp.to_torch(asset.data.root_pos_w).clone()
    grasp_w[:, 2] += grasp_height_offset
    ee_w = wp.to_torch(ee_frame.data.target_pos_w)[..., 0, :]
    to_grasp = grasp_w - ee_w

    scale = torch.tensor(action_scale, device=to_grasp.device, dtype=to_grasp.dtype).unsqueeze(0)
    desired_action = torch.clamp(to_grasp, min=-scale, max=scale)
    arm_action = env.action_manager.get_term(action_name).processed_actions[:, :3]
    action_error = torch.linalg.norm(arm_action - desired_action, dim=1)
    distance = torch.linalg.norm(to_grasp, dim=1)

    tracking = 1.0 - torch.tanh(action_error / std)
    far_from_grasp = torch.tanh(distance / fade_std)
    return _finite_reward(tracking * far_from_grasp)


def scripted_grasp_action_tracking(
    env: ManagerBasedRLEnv,
    std: float,
    gripper_std: float,
    command_name: str,
    action_scale: tuple[float, float, float],
    grasp_height_offset: float = 0.0,
    hover_height_offset: float = 0.12,
    xy_close_distance: float = 0.04,
    close_distance: float = 0.06,
    closed_finger_position: float = 0.035,
    lift_height: float = 0.065,
    vertical_action_limit: float = 1.0,
    arm_action_name: str = "arm_action",
    gripper_action_name: str = "gripper_action",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward matching the scripted reach-close-lift target action.

    Args:
        env: The environment instance.
        std: Exponential kernel scale for raw arm action mean squared error.
        gripper_std: Exponential kernel scale for raw gripper action squared error.
        command_name: Name of the generated pose command.
        action_scale: Per-axis IK action scale used to normalize position targets [m].
        grasp_height_offset: Vertical offset from the COM to the grasp point [m].
        hover_height_offset: Vertical offset from the COM for the pre-grasp hover point [m].
        xy_close_distance: Horizontal distance at which the target switches from hover to descent [m].
        close_distance: Distance at which the target switches from reaching to closing [m].
        closed_finger_position: Finger joint position treated as ready to lift [m].
        lift_height: Deformable COM height at which the target switches to goal tracking [m].
        vertical_action_limit: Absolute raw z action limit for scripted arm targets.
        arm_action_name: Name of the arm action term.
        gripper_action_name: Name of the gripper action term.
        asset_cfg: The deformable object entity.
        ee_frame_cfg: The end-effector frame entity.
        robot_cfg: The robot entity.

    Returns:
        Reward tensor with shape ``(num_envs,)``.
    """
    target_action = scripted_grasp_action_target(
        env,
        command_name=command_name,
        action_scale=action_scale,
        grasp_height_offset=grasp_height_offset,
        hover_height_offset=hover_height_offset,
        xy_close_distance=xy_close_distance,
        close_distance=close_distance,
        closed_finger_position=closed_finger_position,
        lift_height=lift_height,
        vertical_action_limit=vertical_action_limit,
        asset_cfg=asset_cfg,
        ee_frame_cfg=ee_frame_cfg,
        robot_cfg=robot_cfg,
    )
    arm_action = env.action_manager.get_term(arm_action_name).raw_actions
    gripper_action = env.action_manager.get_term(gripper_action_name).raw_actions
    action = torch.cat((arm_action, gripper_action), dim=1)
    arm_error = torch.mean(torch.square(action[:, :3] - target_action[:, :3]), dim=1)
    gripper_error = torch.square(action[:, 3] - target_action[:, 3])
    arm_loss = arm_error / std
    gripper_loss = gripper_error / gripper_std
    tracking = 1.0 - (0.8 * arm_loss + 0.2 * gripper_loss)
    return _finite_reward(torch.clamp(tracking, min=-1.0, max=1.0))


def end_effector_low_height_penalty(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    margin: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Penalty for the end-effector approaching the table from below a safe height.

    Args:
        env: The environment instance.
        minimum_height: Height below which the full penalty is applied [m].
        margin: Height band above :paramref:`minimum_height` where the penalty ramps down [m].
        ee_frame_cfg: The end-effector frame entity.

    Returns:
        Penalty tensor with shape ``(num_envs,)``.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_z = wp.to_torch(ee_frame.data.target_pos_w)[..., 0, 2] - env.scene.env_origins[:, 2]
    return _finite_reward(torch.clamp((minimum_height + margin - ee_z) / margin, min=0.0, max=1.0))


def end_effector_grasp_height(
    env: ManagerBasedRLEnv,
    std: float,
    grasp_height_offset: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward matching the end-effector height to a safe deformable grasp height.

    Args:
        env: The environment instance.
        std: The tanh kernel standard deviation [m].
        grasp_height_offset: Vertical offset from the deformable COM to the grasp height [m].
        asset_cfg: The deformable object entity.
        ee_frame_cfg: The end-effector frame entity.

    Returns:
        Reward tensor with shape ``(num_envs,)``.
    """
    asset: DeformableObject = env.scene[asset_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    grasp_z = wp.to_torch(asset.data.root_pos_w)[:, 2] + grasp_height_offset
    ee_z = wp.to_torch(ee_frame.data.target_pos_w)[..., 0, 2]
    return _finite_reward(1.0 - torch.tanh(torch.abs(ee_z - grasp_z) / std))


def deformable_com_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> torch.Tensor:
    """Reward tracking of the goal position by the deformable's COM (tanh kernel).

    Only credits when the COM is above ``minimal_height`` (i.e. the object is lifted).
    The command is interpreted as ``[x, y, z, qw, qx, qy, qz]`` in the robot's root frame.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    asset: DeformableObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        wp.to_torch(robot.data.root_pos_w), wp.to_torch(robot.data.root_quat_w), des_pos_b
    )
    com_w = wp.to_torch(asset.data.root_pos_w)
    distance = torch.linalg.norm(des_pos_w - com_w, dim=1)
    return _finite_reward((com_w[:, 2] > minimal_height) * (1.0 - torch.tanh(distance / std)))


def deformable_goal_reached(
    env: ManagerBasedRLEnv,
    command_name: str,
    distance_threshold: float,
    minimal_height: float,
    minimum_steps: int = 0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> torch.Tensor:
    """Termination signal when the deformable COM reaches the lifted goal.

    Args:
        env: The environment instance.
        command_name: Name of the generated pose command.
        distance_threshold: Goal-distance threshold [m].
        minimal_height: Minimum COM height [m].
        minimum_steps: Minimum episode age before success can be reported.
        robot_cfg: The robot entity providing the root frame.
        asset_cfg: The deformable object entity.

    Returns:
        Boolean tensor with shape ``(num_envs,)``.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    asset: DeformableObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    goal_w, _ = combine_frame_transforms(
        wp.to_torch(robot.data.root_pos_w), wp.to_torch(robot.data.root_quat_w), command[:, :3]
    )
    com_w = wp.to_torch(asset.data.root_pos_w)
    distance = torch.linalg.norm(goal_w - com_w, dim=1)
    finite = torch.isfinite(distance) & torch.isfinite(com_w[:, 2])
    old_enough = env.episode_length_buf >= minimum_steps
    return old_enough & finite & (com_w[:, 2] > minimal_height) & (distance < distance_threshold)


def gripper_close_action(env: ManagerBasedRLEnv, action_name: str = "gripper_action") -> torch.Tensor:
    """Penalty signal for commanding the gripper to close.

    The binary gripper action uses negative float actions for close commands and
    non-negative actions for open commands.

    Args:
        env: The environment instance.
        action_name: Name of the gripper action term.

    Returns:
        Tensor with shape ``(num_envs,)`` containing ``1`` when the gripper is
        commanded closed and ``0`` otherwise.
    """
    gripper_action = env.action_manager.get_term(action_name).raw_actions
    return _finite_reward(torch.any(gripper_action < 0.0, dim=1).float())


def deformable_com_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> torch.Tensor:
    """Termination signal when the deformable's COM falls below ``minimum_height`` [m]."""
    asset: DeformableObject = env.scene[asset_cfg.name]
    com_z = wp.to_torch(asset.data.root_pos_w)[:, 2]
    return com_z < minimum_height


def deformable_outside_table_bounds(
    env: ManagerBasedRLEnv,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> torch.Tensor:
    """Terminate if any deformable nodal point leaves the table footprint.

    Args:
        env: The environment instance.
        x_bounds: Allowed x-position range in the environment frame [m].
        y_bounds: Allowed y-position range in the environment frame [m].
        asset_cfg: The deformable object entity.

    Returns:
        Boolean tensor with shape ``(num_envs,)``.
    """
    asset: DeformableObject = env.scene[asset_cfg.name]
    nodal_pos = wp.to_torch(asset.data.nodal_pos_w) - env.scene.env_origins.unsqueeze(1)
    outside_x = (nodal_pos[..., 0] < x_bounds[0]) | (nodal_pos[..., 0] > x_bounds[1])
    outside_y = (nodal_pos[..., 1] < y_bounds[0]) | (nodal_pos[..., 1] > y_bounds[1])
    return torch.any(outside_x | outside_y, dim=1)


def deformable_state_nonfinite(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> torch.Tensor:
    """Terminate environments with non-finite deformable state."""
    asset: DeformableObject = env.scene[asset_cfg.name]
    root_finite = torch.isfinite(wp.to_torch(asset.data.root_pos_w)).all(dim=1)
    nodal_finite = torch.isfinite(wp.to_torch(asset.data.nodal_pos_w).reshape(env.num_envs, -1)).all(dim=1)
    return ~(root_finite & nodal_finite)


def ee_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Termination signal when the end-effector falls below ``minimum_height`` [m].

    Height is measured in the environment frame (``z`` of the EE position with the env
    origin subtracted), so the threshold is independent of the environment's xy offset.

    Args:
        env: The environment instance.
        minimum_height: Minimum allowed EE height in the environment frame [m].
        ee_frame_cfg: The end-effector frame entity.

    Returns:
        Boolean tensor with shape ``(num_envs,)``.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_z = wp.to_torch(ee_frame.data.target_pos_w)[..., 0, 2] - env.scene.env_origins[:, 2]
    return ee_z < minimum_height


def robot_state_nonfinite(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Terminate environments with non-finite robot state.

    Args:
        env: The environment instance.
        robot_cfg: The robot entity.
        ee_frame_cfg: The end-effector frame entity.

    Returns:
        Boolean tensor with shape ``(num_envs,)``.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    joint_state_finite = torch.isfinite(robot.data.joint_pos.torch).all(dim=1) & torch.isfinite(
        robot.data.joint_vel.torch
    ).all(dim=1)
    ee_finite = torch.isfinite(wp.to_torch(ee_frame.data.target_pos_w).reshape(env.num_envs, -1)).all(dim=1)
    return ~(joint_state_finite & ee_finite)
