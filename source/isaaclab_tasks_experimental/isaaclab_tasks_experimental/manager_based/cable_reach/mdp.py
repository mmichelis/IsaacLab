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
    axis_angle_from_quat,
    combine_frame_transforms,
    quat_conjugate,
    quat_error_magnitude,
    quat_mul,
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
    """Sanitize a quaternion and canonicalize to the w>=0 hemisphere.

    Steps: replace NaN/Inf with identity, renormalize, then flip sign so w>=0. The
    canonicalization collapses the double-cover (q and -q represent the same rotation)
    to a single hemisphere — critical when observations are not normalized, because
    otherwise the network sees sign-flipped quats for the same physical orientation.
    """
    quat = torch.nan_to_num(quat, nan=0.0, posinf=0.0, neginf=0.0)
    norm = torch.linalg.norm(quat, dim=-1, keepdim=True)
    identity = torch.zeros_like(quat)
    identity[..., 3] = 1.0  # (x, y, z, w) identity
    quat = torch.where(norm > 1e-6, quat, identity)
    quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp(min=1e-6)
    # Flip to w>=0 hemisphere.
    sign = torch.where(quat[..., 3:4] < 0.0, -1.0, 1.0)
    return quat * sign


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
    velocity_clamp: float = 10.0,
) -> torch.Tensor:
    """Handle linear and angular velocity in world frame. Returns ``[lin(3), ang(3)]``.

    Clamped to +-``velocity_clamp`` per component to prevent obs blowups when the
    solver produces extreme values during early-training cable whips. The default is
    intentionally conservative (10 rad/s or m/s): larger spikes would dominate the
    first-layer activations of an unnormalized MLP.
    """
    cable: Articulation = env.scene[cable_cfg.name]
    lin_vel = _safe_pos(wp.to_torch(cable.data.root_lin_vel_w))
    ang_vel = _safe_pos(wp.to_torch(cable.data.root_ang_vel_w))
    return torch.cat([lin_vel, ang_vel], dim=-1).clamp(-velocity_clamp, velocity_clamp)


def ee_pose_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """End-effector pose in robot root frame. Returns ``[pos(3), quat(x,y,z,w)]``."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = _safe_pos(wp.to_torch(ee_frame.data.target_pos_w)[..., 0, :])
    ee_quat_w = _safe_quat(wp.to_torch(ee_frame.data.target_quat_w)[..., 0, :])
    root_pos_w, root_quat_w = _robot_root_pose_w(env, robot_cfg)
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
    )
    return torch.cat([_safe_pos(ee_pos_b), _safe_quat(ee_quat_b)], dim=-1)


def ee_to_target_position(
    env: ManagerBasedRLEnv,
    command_name: str = "handle_pose",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Vector from the end-effector to the commanded target, in world frame. Shape ``(N, 3)``.

    Equivalent to robot-root-frame deltas because the Franka base is non-rotating; kept
    in world frame to avoid an extra rotation that numerically cancels to identity.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = _safe_pos(wp.to_torch(ee_frame.data.target_pos_w)[..., 0, :])
    des_pos_w, _ = _target_pose_w(env, command_name, robot_cfg)
    return _safe_pos(des_pos_w - ee_pos_w)


def target_orientation_error(
    env: ManagerBasedRLEnv,
    command_name: str = "handle_pose",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
) -> torch.Tensor:
    """Axis-angle rotation from current handle orientation to the target. Shape ``(N, 3)``.

    Gives the network a *direct* rotation-error signal (analogous to how
    :func:`handle_to_target_position` supplies the position error), rather than
    requiring it to infer the error from two raw quaternions.
    """
    handle_quat_w = _handle_quat_w(env, cable_cfg)
    _, des_quat_w = _target_pose_w(env, command_name, robot_cfg)
    # q_err = q_target * q_current^-1 — applied to the world-frame handle orientation,
    # gives the rotation that takes current into target.
    q_err = _safe_quat(quat_mul(des_quat_w, quat_conjugate(handle_quat_w)))
    return _safe_pos(axis_angle_from_quat(q_err))


def grasp_indicator(
    env: ManagerBasedRLEnv,
    handle_capture_radius: float = 0.08,
    finger_close_threshold: float = 0.035,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"]),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
) -> torch.Tensor:
    """Exposes the binary grasp detector (shape ``(N, 1)``) as an observation.

    Mirrors the staged-reward gate so the critic can attribute the discrete reward
    jumps at grasp transitions, instead of re-inferring the condition from finger
    joints and ee-to-handle distance.
    """
    g = is_grasped(
        env,
        handle_capture_radius=handle_capture_radius,
        finger_close_threshold=finger_close_threshold,
        robot_cfg=robot_cfg,
        ee_frame_cfg=ee_frame_cfg,
        cable_cfg=cable_cfg,
    )
    return g.unsqueeze(-1)


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
    handle_capture_radius: float = 0.08,
    finger_close_threshold: float = 0.035,
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
    handle_capture_radius: float = 0.08,
    finger_close_threshold: float = 0.035,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"]),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
) -> torch.Tensor:
    """Binary: handle lifted above the table while being grasped. Returns 0/1.

    Used by :func:`success_bonus` and as an optional helper; the staged reward no
    longer gates the target-tracking rewards on this because the cliff at
    ``minimal_height`` kills the gradient before the policy discovers lifting.
    """
    grasped = is_grasped(
        env, handle_capture_radius, finger_close_threshold, robot_cfg, ee_frame_cfg, cable_cfg
    )
    handle_pos_w = _handle_pos_w(env, cable_cfg)
    above = (handle_pos_w[:, 2] > minimal_height).float()
    return grasped * above


def lift_progress(
    env: ManagerBasedRLEnv,
    rest_height: float = 0.02,
    max_lift: float = 0.3,
    handle_capture_radius: float = 0.08,
    finger_close_threshold: float = 0.035,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"]),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
) -> torch.Tensor:
    """Continuous lift reward — replaces the sparse binary ``is_lifted`` bonus.

    Returns ``is_grasped * clamp((handle_z - rest_height) / max_lift, 0, 1)``. Once the
    handle is grasped, this climbs linearly from 0 at rest (~2 cm above the table) to
    1 at ``rest_height + max_lift``, giving the policy a smooth gradient pulling the
    handle upward instead of a step-function bonus it has to stumble onto.

    Assumes the environment's ground z=0 plane coincides with the table surface, so
    world-frame handle z is effectively "height above the table."
    """
    grasped = is_grasped(
        env, handle_capture_radius, finger_close_threshold, robot_cfg, ee_frame_cfg, cable_cfg
    )
    handle_pos_w = _handle_pos_w(env, cable_cfg)
    progress = torch.clamp((handle_pos_w[:, 2] - rest_height) / max_lift, 0.0, 1.0)
    return grasped * progress


def handle_target_position_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "handle_pose",
    handle_capture_radius: float = 0.08,
    finger_close_threshold: float = 0.035,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"]),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
) -> torch.Tensor:
    """Stage 4 (position): tanh reward for tracking the 3D target, gated by is_grasped.

    Gated on grasp (not on a lift height) so that once the policy is holding the
    handle, it has an immediate smooth signal pulling the handle toward the target.
    Targets always sit ≥15 cm above the table, so the gradient naturally requires
    lifting without a hardcoded height threshold.
    """
    grasped = is_grasped(
        env, handle_capture_radius, finger_close_threshold, robot_cfg, ee_frame_cfg, cable_cfg
    )
    handle_pos_w = _handle_pos_w(env, cable_cfg)
    des_pos_w, _ = _target_pose_w(env, command_name, robot_cfg)
    distance = torch.linalg.norm(des_pos_w - handle_pos_w, dim=1)
    return grasped * (1.0 - torch.tanh(distance / std))


def handle_target_orientation_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "handle_pose",
    handle_capture_radius: float = 0.08,
    finger_close_threshold: float = 0.035,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"]),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
) -> torch.Tensor:
    """Stage 4 (orientation): tanh reward for matching target orientation, gated by is_grasped."""
    grasped = is_grasped(
        env, handle_capture_radius, finger_close_threshold, robot_cfg, ee_frame_cfg, cable_cfg
    )
    handle_quat_w = _handle_quat_w(env, cable_cfg)
    _, des_quat_w = _target_pose_w(env, command_name, robot_cfg)
    angle = quat_error_magnitude(handle_quat_w, des_quat_w)
    return grasped * (1.0 - torch.tanh(angle / std))


def success_bonus(
    env: ManagerBasedRLEnv,
    pos_threshold: float = 0.05,
    rot_threshold: float = 0.3,
    command_name: str = "handle_pose",
    minimal_height: float = 0.04,
    handle_capture_radius: float = 0.08,
    finger_close_threshold: float = 0.035,
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
    max_distance_from_env_origin: float = 10.0,
    max_linear_velocity: float = 100.0,
    max_angular_velocity: float = 500.0,
) -> torch.Tensor:
    """Per-env failure termination when the physics state is invalid.

    Fires when any of the following is true for an env:

    * Cable root pose / velocity or robot root pose / quat contains NaN / Inf.
    * Cable handle is further than ``max_distance_from_env_origin`` m from the env's
      origin — a blown-up solver will launch the handle far from its ~0.5 m workspace.
    * Cable linear velocity magnitude exceeds ``max_linear_velocity`` m/s.
    * Cable angular velocity magnitude exceeds ``max_angular_velocity`` rad/s.

    Register with ``time_out=False`` so the episode is counted as a failure rather
    than a clean truncation.

    Thresholds are intentionally generous — they catch real solver blowups but leave
    normal dynamic behavior untouched. Crucially, the position check is in the **env-
    local frame** (handle pos minus env origin) rather than the world frame, because
    the env grid itself can span 100+ m for large ``num_envs``.
    """
    cable: Articulation = env.scene[cable_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    cable_pos_w = wp.to_torch(cable.data.root_pos_w)
    cable_lin = wp.to_torch(cable.data.root_lin_vel_w)
    cable_ang = wp.to_torch(cable.data.root_ang_vel_w)
    cable_quat = wp.to_torch(cable.data.root_quat_w)
    robot_pos_w = wp.to_torch(robot.data.root_pos_w)
    robot_quat = wp.to_torch(robot.data.root_quat_w)

    # Handle position relative to this env's origin — cancels out the env-grid offset
    # so the position check is meaningful regardless of how many parallel envs there are.
    cable_pos_local = cable_pos_w - env.scene.env_origins

    def _bad(x: torch.Tensor) -> torch.Tensor:
        invalid = ~torch.isfinite(x)
        return invalid.view(invalid.shape[0], -1).any(dim=-1)

    invalid = (
        _bad(cable_pos_w)
        | _bad(cable_lin)
        | _bad(cable_ang)
        | _bad(cable_quat)
        | _bad(robot_pos_w)
        | _bad(robot_quat)
        | (torch.linalg.norm(cable_pos_local, dim=-1) > max_distance_from_env_origin)
        | (torch.linalg.norm(cable_lin, dim=-1) > max_linear_velocity)
        | (torch.linalg.norm(cable_ang, dim=-1) > max_angular_velocity)
    )
    return invalid
