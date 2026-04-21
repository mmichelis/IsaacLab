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


def _safe_pos(pos: torch.Tensor) -> torch.Tensor:
    """Replace NaN/Inf in a position tensor with 0.0.

    This is a last-resort sanitizer on the *terminal* observation of a failing episode
    so the RSL-RL ``check_nan`` pre-loop check does not crash the whole run. The real
    handling lives in :func:`invalid_cable_state`, which fires an actual failure
    termination so the framework resets the offending env next step.
    """
    return torch.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_quat(quat: torch.Tensor) -> torch.Tensor:
    """Replace NaN/Inf quats with identity (x=0, y=0, z=0, w=1) and renormalize."""
    quat = torch.nan_to_num(quat, nan=0.0, posinf=0.0, neginf=0.0)
    # Detect zero-norm (all zeros after nan_to_num) and substitute identity.
    norm = torch.linalg.norm(quat, dim=-1, keepdim=True)
    identity = torch.zeros_like(quat)
    identity[..., 3] = 1.0  # (x, y, z, w) identity
    quat = torch.where(norm > 1e-6, quat, identity)
    # Renormalize for numerical cleanliness before feeding to math ops.
    return quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp(min=1e-6)


def _handle_pos_w(env: ManagerBasedRLEnv, cable_cfg: SceneEntityCfg) -> torch.Tensor:
    cable: Articulation = env.scene[cable_cfg.name]
    return _safe_pos(wp.to_torch(cable.data.root_pos_w))


def _handle_quat_w(env: ManagerBasedRLEnv, cable_cfg: SceneEntityCfg) -> torch.Tensor:
    cable: Articulation = env.scene[cable_cfg.name]
    return _safe_quat(wp.to_torch(cable.data.root_quat_w))


def _robot_root_pose_w(
    env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg
) -> tuple[torch.Tensor, torch.Tensor]:
    robot: Articulation = env.scene[robot_cfg.name]
    return (
        _safe_pos(wp.to_torch(robot.data.root_pos_w)),
        _safe_quat(wp.to_torch(robot.data.root_quat_w)),
    )


def _target_pose_w(
    env: ManagerBasedRLEnv, command_name: str, robot_cfg: SceneEntityCfg
) -> tuple[torch.Tensor, torch.Tensor]:
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_quat_b = _safe_quat(command[:, 3:7])
    root_pos_w, root_quat_w = _robot_root_pose_w(env, robot_cfg)
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
    handle_pos_w = _handle_pos_w(env, cable_cfg)
    handle_quat_w = _handle_quat_w(env, cable_cfg)
    root_pos_w, root_quat_w = _robot_root_pose_w(env, robot_cfg)
    handle_pos_b, handle_quat_b = subtract_frame_transforms(
        root_pos_w, root_quat_w, handle_pos_w, handle_quat_w
    )
    return torch.cat([_safe_pos(handle_pos_b), _safe_quat(handle_quat_b)], dim=-1)


def handle_velocity(
    env: ManagerBasedRLEnv,
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
    velocity_clamp: float = 50.0,
) -> torch.Tensor:
    """Handle linear and angular velocity in world frame. Returns ``[lin(3), ang(3)]``.

    Clamped to +-``velocity_clamp`` per component to prevent obs blowups when the
    solver produces extreme values during early-training cable whips.
    """
    cable: Articulation = env.scene[cable_cfg.name]
    lin_vel = _safe_pos(wp.to_torch(cable.data.root_lin_vel_w))
    ang_vel = _safe_pos(wp.to_torch(cable.data.root_ang_vel_w))
    return torch.cat([lin_vel, ang_vel], dim=-1).clamp(-velocity_clamp, velocity_clamp)


def ee_to_handle_position(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
) -> torch.Tensor:
    """Vector from the end-effector to the handle in world frame. Shape ``(N, 3)``."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_w = _safe_pos(wp.to_torch(ee_frame.data.target_pos_w)[..., 0, :])
    handle_pos_w = _handle_pos_w(env, cable_cfg)
    return _safe_pos(handle_pos_w - ee_w)


def handle_to_target_position(
    env: ManagerBasedRLEnv,
    command_name: str = "handle_pose",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
) -> torch.Tensor:
    """Vector from the handle to the commanded target in world frame. Shape ``(N, 3)``."""
    des_pos_w, _ = _target_pose_w(env, command_name, robot_cfg)
    handle_pos_w = _handle_pos_w(env, cable_cfg)
    return _safe_pos(des_pos_w - handle_pos_w)


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


##
# Terminations
##


def invalid_cable_state(
    env: ManagerBasedRLEnv,
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_position: float = 50.0,
    max_linear_velocity: float = 100.0,
    max_angular_velocity: float = 500.0,
) -> torch.Tensor:
    """Per-env failure termination when the physics state is invalid.

    Fires when any of the following is true for an env:

    * Cable root pose/velocity contains NaN or Inf.
    * Cable handle is further than ``max_position`` m from the origin (solver blew
      up and launched the cable across the universe).
    * Cable linear velocity magnitude exceeds ``max_linear_velocity`` m/s.
    * Cable angular velocity magnitude exceeds ``max_angular_velocity`` rad/s.
    * Robot root pose/velocity contains NaN or Inf.

    Register with ``time_out=False`` so the episode is counted as a failure rather
    than a clean truncation. The reset event will re-initialize the offending env;
    the rest of the envs keep training.

    Thresholds are intentionally generous — they catch genuine solver blowups but
    leave normal dynamic behavior untouched.
    """
    cable: Articulation = env.scene[cable_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    cable_pos = wp.to_torch(cable.data.root_pos_w)
    cable_lin = wp.to_torch(cable.data.root_lin_vel_w)
    cable_ang = wp.to_torch(cable.data.root_ang_vel_w)
    cable_quat = wp.to_torch(cable.data.root_quat_w)
    robot_pos = wp.to_torch(robot.data.root_pos_w)
    robot_quat = wp.to_torch(robot.data.root_quat_w)

    def _bad(x: torch.Tensor) -> torch.Tensor:
        # Collapse all non-batch dims to a single ``(num_envs,)`` bool.
        invalid = ~torch.isfinite(x)
        return invalid.view(invalid.shape[0], -1).any(dim=-1)

    invalid = (
        _bad(cable_pos)
        | _bad(cable_lin)
        | _bad(cable_ang)
        | _bad(cable_quat)
        | _bad(robot_pos)
        | _bad(robot_quat)
        | (torch.linalg.norm(cable_pos, dim=-1) > max_position)
        | (torch.linalg.norm(cable_lin, dim=-1) > max_linear_velocity)
        | (torch.linalg.norm(cable_ang, dim=-1) > max_angular_velocity)
    )
    return invalid
