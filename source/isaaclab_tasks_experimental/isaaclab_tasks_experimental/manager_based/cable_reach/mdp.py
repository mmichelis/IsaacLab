# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific MDP helpers for the cable-reach environment.

The handle is the root body of the cable articulation (the ``<freejoint/>`` sits on the
handle in the generated MJCF), so the handle pose/velocity are read via the asset's
``root_*_w`` buffers rather than ``body_link_*_w[body_ids]`` — the root buffers are
always populated at reset time, whereas link-level forward kinematics may not be
propagated yet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_error_magnitude,
    subtract_frame_transforms,
)

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import FrameTransformer


##
# Internal helpers
##


def _handle_pos_w(env: ManagerBasedRLEnv, cable_cfg: SceneEntityCfg) -> torch.Tensor:
    cable: Articulation = env.scene[cable_cfg.name]
    return wp.to_torch(cable.data.root_pos_w)


def _handle_quat_w(env: ManagerBasedRLEnv, cable_cfg: SceneEntityCfg) -> torch.Tensor:
    cable: Articulation = env.scene[cable_cfg.name]
    return wp.to_torch(cable.data.root_quat_w)


def _target_pose_w(
    env: ManagerBasedRLEnv, command_name: str, robot_cfg: SceneEntityCfg
) -> tuple[torch.Tensor, torch.Tensor]:
    robot: Articulation = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_quat_b = command[:, 3:7]
    root_pos_w = wp.to_torch(robot.data.root_pos_w)
    root_quat_w = wp.to_torch(robot.data.root_quat_w)
    des_pos_w, des_quat_w = combine_frame_transforms(
        root_pos_w, root_quat_w, des_pos_b, des_quat_b
    )
    return des_pos_w, des_quat_w


##
# Observations
##


def handle_pose_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
) -> torch.Tensor:
    """Handle pose expressed in the robot root frame. Returns ``[pos(3), quat(x,y,z,w)]``."""
    robot: Articulation = env.scene[robot_cfg.name]
    handle_pos_w = _handle_pos_w(env, cable_cfg)
    handle_quat_w = _handle_quat_w(env, cable_cfg)
    root_pos_w = wp.to_torch(robot.data.root_pos_w)
    root_quat_w = wp.to_torch(robot.data.root_quat_w)
    handle_pos_b, handle_quat_b = subtract_frame_transforms(
        root_pos_w, root_quat_w, handle_pos_w, handle_quat_w
    )
    return torch.cat([handle_pos_b, handle_quat_b], dim=-1)


def handle_velocity(
    env: ManagerBasedRLEnv,
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
) -> torch.Tensor:
    """Handle linear and angular velocity in world frame. Returns ``[lin(3), ang(3)]``."""
    cable: Articulation = env.scene[cable_cfg.name]
    lin_vel = wp.to_torch(cable.data.root_lin_vel_w)
    ang_vel = wp.to_torch(cable.data.root_ang_vel_w)
    return torch.cat([lin_vel, ang_vel], dim=-1)


def ee_to_handle_position(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
) -> torch.Tensor:
    """Vector from the end-effector to the handle in world frame. Shape ``(N, 3)``."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_w = wp.to_torch(ee_frame.data.target_pos_w)[..., 0, :]
    handle_pos_w = _handle_pos_w(env, cable_cfg)
    return handle_pos_w - ee_w


def handle_to_target_position(
    env: ManagerBasedRLEnv,
    command_name: str = "handle_pose",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
) -> torch.Tensor:
    """Vector from the handle to the commanded target in world frame. Shape ``(N, 3)``."""
    des_pos_w, _ = _target_pose_w(env, command_name, robot_cfg)
    handle_pos_w = _handle_pos_w(env, cable_cfg)
    return des_pos_w - handle_pos_w


##
# Rewards
##


def ee_to_handle_distance_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
) -> torch.Tensor:
    """Stage 1: reward the end-effector for approaching the handle (tanh kernel)."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_w = wp.to_torch(ee_frame.data.target_pos_w)[..., 0, :]
    handle_pos_w = _handle_pos_w(env, cable_cfg)
    distance = torch.linalg.norm(handle_pos_w - ee_w, dim=1)
    return 1.0 - torch.tanh(distance / std)


def is_grasped(
    env: ManagerBasedRLEnv,
    handle_capture_radius: float = 0.04,
    finger_close_threshold: float = 0.025,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"]),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
) -> torch.Tensor:
    """Geometric grasp detector: handle is near the EE, fingers are closed.

    Returns a ``(num_envs,)`` float tensor of 0.0 / 1.0. Intended both as a gate for
    later-stage rewards and as a small standalone bonus.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    handle_pos_w = _handle_pos_w(env, cable_cfg)
    ee_pos_w = wp.to_torch(ee_frame.data.target_pos_w)[..., 0, :]
    distance = torch.linalg.norm(handle_pos_w - ee_pos_w, dim=1)
    near = distance < handle_capture_radius

    joint_pos = wp.to_torch(robot.data.joint_pos)[:, robot_cfg.joint_ids]
    closed = (joint_pos < finger_close_threshold).all(dim=-1)
    return (near & closed).float()


def is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    handle_capture_radius: float = 0.04,
    finger_close_threshold: float = 0.025,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"]),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
) -> torch.Tensor:
    """Stage 3: handle lifted above the table while being grasped. Returns 0/1."""
    grasped = is_grasped(
        env, handle_capture_radius, finger_close_threshold, robot_cfg, ee_frame_cfg, cable_cfg
    )
    handle_pos_w = _handle_pos_w(env, cable_cfg)
    above = (handle_pos_w[:, 2] > minimal_height).float()
    return grasped * above


def handle_target_position_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str = "handle_pose",
    handle_capture_radius: float = 0.04,
    finger_close_threshold: float = 0.025,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"]),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
) -> torch.Tensor:
    """Stage 4 (position): tanh reward for tracking the 3D target, gated by is_lifted."""
    lifted = is_lifted(
        env,
        minimal_height,
        handle_capture_radius,
        finger_close_threshold,
        robot_cfg,
        ee_frame_cfg,
        cable_cfg,
    )
    handle_pos_w = _handle_pos_w(env, cable_cfg)
    des_pos_w, _ = _target_pose_w(env, command_name, robot_cfg)
    distance = torch.linalg.norm(des_pos_w - handle_pos_w, dim=1)
    return lifted * (1.0 - torch.tanh(distance / std))


def handle_target_orientation_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str = "handle_pose",
    handle_capture_radius: float = 0.04,
    finger_close_threshold: float = 0.025,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"]),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
) -> torch.Tensor:
    """Stage 4 (orientation): tanh reward for matching target orientation, gated by is_lifted."""
    lifted = is_lifted(
        env,
        minimal_height,
        handle_capture_radius,
        finger_close_threshold,
        robot_cfg,
        ee_frame_cfg,
        cable_cfg,
    )
    handle_quat_w = _handle_quat_w(env, cable_cfg)
    _, des_quat_w = _target_pose_w(env, command_name, robot_cfg)
    angle = quat_error_magnitude(handle_quat_w, des_quat_w)
    return lifted * (1.0 - torch.tanh(angle / std))


def success_bonus(
    env: ManagerBasedRLEnv,
    pos_threshold: float = 0.02,
    rot_threshold: float = 0.1,
    command_name: str = "handle_pose",
    minimal_height: float = 0.04,
    handle_capture_radius: float = 0.04,
    finger_close_threshold: float = 0.025,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"]),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
) -> torch.Tensor:
    """Sparse +1 when lifted AND within pos/rot thresholds of the target."""
    lifted = is_lifted(
        env,
        minimal_height,
        handle_capture_radius,
        finger_close_threshold,
        robot_cfg,
        ee_frame_cfg,
        cable_cfg,
    )
    handle_pos_w = _handle_pos_w(env, cable_cfg)
    handle_quat_w = _handle_quat_w(env, cable_cfg)
    des_pos_w, des_quat_w = _target_pose_w(env, command_name, robot_cfg)
    pos_err = torch.linalg.norm(des_pos_w - handle_pos_w, dim=1)
    rot_err = quat_error_magnitude(handle_quat_w, des_quat_w)
    within = ((pos_err < pos_threshold) & (rot_err < rot_threshold)).float()
    return lifted * within
