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
    finger_close_threshold: float = 0.020,
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
    """Stage 1: reward the end-effector for approaching the handle (tanh kernel).

    Distance is measured from the EE point (mid-fingertip, via the 10.3 cm z-
    offset from ``panda_hand``) to the handle's root body centre. A perfect
    grasp puts the EE at the handle's centre, so distance → 0 is the peak.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_w = wp.to_torch(ee_frame.data.target_pos_w)[..., 0, :]
    handle_pos_w = _handle_pos_w(env, cable_cfg)
    distance = torch.linalg.norm(handle_pos_w - ee_w, dim=1)
    return 1.0 - torch.tanh(distance / std)


def ee_below_threshold(
    env: ManagerBasedRLEnv,
    min_z: float = 0.01,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Penalty for the EE point sagging onto the table.

    Returns ``max(0, min_z − ee_z)`` — a positive number measuring HOW FAR below
    the threshold the EE is. Zero when the EE is above ``min_z``. Pair with a
    NEGATIVE weight in the reward config. Using linear distance-below rather
    than a binary threshold gives a smooth gradient that tells the policy to
    lift the arm progressively, not just "get off the table by 1 mm."
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_z = wp.to_torch(ee_frame.data.target_pos_w)[..., 0, 2]
    return torch.clamp(min_z - ee_z, min=0.0)


def gripper_closed_without_grasp(
    env: ManagerBasedRLEnv,
    # Threshold below which the gripper counts as "starting to close". 0.035 is
    # slightly below the fully-open position (0.04), so a freshly-opened gripper
    # has zero penalty while any commanded close starts accumulating it.
    open_threshold: float = 0.035,
    # Same geometric-grasp parameters as :func:`is_grasped_geometric` so the
    # gate matches exactly — the penalty disengages iff ``is_grasped_geometric``
    # would fire (handle near EE AND fingers in the clamp range).
    handle_capture_radius: float = 0.07,
    finger_close_threshold: float = 0.035,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"]),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
) -> torch.Tensor:
    """Penalty for having the gripper fingers closed while NOT grasping the
    handle.

    Returns ``(how-closed) × (not-grasping)`` as a scalar in ``[0, 1]`` per env;
    pair with a negative weight in the reward cfg. Both terms are smooth:

    * ``how_closed = clamp((open_threshold − avg_finger_pos) / open_threshold, 0, 1)``
      — 0 when the gripper is fully open, rising linearly to 1 when fully closed.
    * ``not_grasping = 1 − is_grasped_geometric`` — 0 when the gripper is
      actually on the handle (so a real grasp incurs no penalty), 1 otherwise.

    Intended to force the policy to **keep the gripper open during approach** and
    only close when it's genuinely over the handle — addresses the "closes
    prematurely and then can't recover" behaviour we've been seeing.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    joint_pos = wp.to_torch(robot.data.joint_pos)[:, robot_cfg.joint_ids]
    avg_finger = joint_pos.mean(dim=-1)
    how_closed = torch.clamp((open_threshold - avg_finger) / open_threshold, 0.0, 1.0)

    # Geometric grasp gate, inlined instead of calling is_grasped_geometric to
    # avoid re-reading the scene entities twice.
    handle_pos_w = _handle_pos_w(env, cable_cfg)
    ee_pos_w = wp.to_torch(ee_frame.data.target_pos_w)[..., 0, :]
    near = torch.linalg.norm(handle_pos_w - ee_pos_w, dim=1) < handle_capture_radius
    gripping = ((joint_pos > 0.010) & (joint_pos < finger_close_threshold)).all(dim=-1)
    grasped = (near & gripping).float()

    return how_closed * (1.0 - grasped)


_HAND_BODY_IDX_CACHE: dict[int, int] = {}


def _panda_hand_body_id(robot: "Articulation") -> int:
    """Resolve and cache the panda_hand body index in the robot articulation.

    Avoids the need for callers to thread a resolved ``SceneEntityCfg`` through
    every reward-term's ``params`` dict.
    """
    key = id(robot)
    cached = _HAND_BODY_IDX_CACHE.get(key)
    if cached is not None:
        return cached
    ids, _ = robot.find_bodies(["panda_hand"])
    if not ids:
        raise RuntimeError("Robot articulation has no body named 'panda_hand'.")
    _HAND_BODY_IDX_CACHE[key] = ids[0]
    return ids[0]


def is_grasped_geometric(
    env: ManagerBasedRLEnv,
    # The 8 × 6 × 6 cm box has a 4 cm half-length along its long axis, so a
    # legitimately-gripping EE can be up to 4 cm from the box centre. 7 cm
    # proximity gives comfortable margin for the EE to sit anywhere along the
    # box while still rejecting "fingers closed 10+ cm away."
    handle_capture_radius: float = 0.07,
    # Fingers resting on the 6 cm-wide box settle at ~0.030 each (half the box
    # width). 0.030 as a strict upper bound was failing exactly at the boundary;
    # 0.035 gives 5 mm of slack for slight compression or squeeze.
    finger_close_threshold: float = 0.035,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"]),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
) -> torch.Tensor:
    """Loose geometric grasp check: handle near EE + fingers in gripping range.

    Thresholds sized for the current 8 × 6 × 6 cm rigid box. Used to gate the
    ``grasp`` bonus and the shaping terms (``handle_above_table``, ``is_lifted``)
    so the policy gets a clean signal the moment it closes on the handle. The
    stricter velocity-correlation variant lives in :func:`is_grasped` and is
    reserved for ``handle_tracks_gripper`` and ``success_bonus``, where we
    actually want to verify physical co-motion (not just geometric contact).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    handle_pos_w = _handle_pos_w(env, cable_cfg)
    ee_pos_w = wp.to_torch(ee_frame.data.target_pos_w)[..., 0, :]
    near = torch.linalg.norm(handle_pos_w - ee_pos_w, dim=1) < handle_capture_radius
    joint_pos = wp.to_torch(robot.data.joint_pos)[:, robot_cfg.joint_ids]
    gripping = ((joint_pos > 0.010) & (joint_pos < finger_close_threshold)).all(dim=-1)
    return (near & gripping).float()


def is_grasped(
    env: ManagerBasedRLEnv,
    # Proximity tolerance. 5 cm: lenient enough that the handle doesn't pop out
    # of the check during brief contact dynamics, tight enough that a real grip
    # always passes (the 4 cm handle is within 3 cm of the EE when clamped).
    handle_capture_radius: float = 0.05,
    # Upper bound of the gripping range. For the 4 cm wide handle, each Panda finger
    # rests at ~0.02 m from center when actually clamped. Anything above this means
    # the gripper hasn't closed on the handle yet. The lower bound (0.010) is
    # hardcoded below: closing on empty air drops the joints all the way to 0, which
    # should NOT count as a grasp.
    finger_close_threshold: float = 0.030,
    # Max relative LINEAR velocity. 0.15 m/s is loose enough to pass through the
    # transient when the fingers first contact the handle (brief shock) but still
    # tight enough to reject sustained "pushing the handle across the table"
    # behaviours — those have handle lagging the gripper by 0.3-0.5 m/s.
    max_lin_vel_mismatch: float = 0.15,
    # Max relative ANGULAR velocity. Kept as a filter against "flick" exploits:
    # a flicked handle spins at 5-20 rad/s about its inertia while the gripper
    # barely rotates, so even a generous 2 rad/s tolerance still rejects flicks.
    # 0.5 rad/s was too tight for normal gripper rotations during lift.
    max_ang_vel_mismatch: float = 2.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"]),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
) -> torch.Tensor:
    """Physical grasp detector: handle is near the EE, fingers are in the gripping
    range, AND the handle moves AS A RIGID BODY with the gripper (linear AND
    angular velocity correlation).

    Returns a ``(num_envs,)`` float tensor of 0.0 / 1.0.

    All four conditions must hold:

    * **Proximity**: handle within ``handle_capture_radius`` of the EE
    * **Fingers**: both finger joints in the gripping range (not open, not on air)
    * **Linear velocity match**: ``|v_handle − v_hand| < max_lin_vel_mismatch``
    * **Angular velocity match**: ``|ω_handle − ω_hand| < max_ang_vel_mismatch``

    The angular-velocity check is the key addition over a pure linear-velocity
    test: during a flick, linear velocities can match for a frame as the gripper
    accelerates the handle, but the handle spins freely about its own axis while
    the gripper frame does not. Requiring BOTH match means the handle truly has
    to be moving as if rigidly attached to the gripper, not just "being pushed in
    the same direction for an instant."
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cable: Articulation = env.scene[cable_cfg.name]

    # Proximity check
    handle_pos_w = _handle_pos_w(env, cable_cfg)
    ee_pos_w = wp.to_torch(ee_frame.data.target_pos_w)[..., 0, :]
    near = torch.linalg.norm(handle_pos_w - ee_pos_w, dim=1) < handle_capture_radius

    # Finger-gap check
    joint_pos = wp.to_torch(robot.data.joint_pos)[:, robot_cfg.joint_ids]
    gripping = ((joint_pos > 0.010) & (joint_pos < finger_close_threshold)).all(dim=-1)

    # Linear & angular velocity correlation.
    hand_body_id = _panda_hand_body_id(robot)
    hand_lin_vel_w = wp.to_torch(robot.data.body_link_vel_w)[:, hand_body_id, :3]
    hand_ang_vel_w = wp.to_torch(robot.data.body_link_vel_w)[:, hand_body_id, 3:6]
    handle_lin_vel_w = _safe_pos(wp.to_torch(cable.data.root_lin_vel_w))
    handle_ang_vel_w = _safe_pos(wp.to_torch(cable.data.root_ang_vel_w))

    lin_match = torch.linalg.norm(handle_lin_vel_w - hand_lin_vel_w, dim=1) < max_lin_vel_mismatch
    ang_match = torch.linalg.norm(handle_ang_vel_w - hand_ang_vel_w, dim=1) < max_ang_vel_mismatch

    return (near & gripping & lin_match & ang_match).float()


def is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    handle_capture_radius: float = 0.07,
    finger_close_threshold: float = 0.035,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"]),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
) -> torch.Tensor:
    """Binary: handle lifted above the table while being grasped. Returns 0/1.

    Gate uses the LOOSE geometric grasp check (:func:`is_grasped_geometric`, not
    the strict velocity-correlation :func:`is_grasped`) — if the box is up off
    the table and the fingers are clamped on it near the EE, that's a real lift,
    regardless of velocity transients.
    """
    grasped = is_grasped_geometric(
        env=env,
        handle_capture_radius=handle_capture_radius,
        finger_close_threshold=finger_close_threshold,
        robot_cfg=robot_cfg,
        ee_frame_cfg=ee_frame_cfg,
        cable_cfg=cable_cfg,
    )
    handle_pos_w = _handle_pos_w(env, cable_cfg)
    above = (handle_pos_w[:, 2] > minimal_height).float()
    return grasped * above


def handle_tracks_gripper(
    env: ManagerBasedRLEnv,
    # Tanh kernel std for the linear-velocity mismatch. 0.1 m/s places a ~50%
    # reward drop at vel_diff = 0.1 m/s. Tighter means the policy is penalized
    # for any slip; looser means the policy tolerates transient mismatches.
    lin_vel_std: float = 0.10,
    # Gate the reward on the gripper actually being in motion. Without this, a
    # stationary grip earns full reward (both velocities are 0 → perfect match),
    # which would let the policy farm it by grabbing and sitting still. Reward
    # fades in linearly from 0 (gripper static) to 1 (gripper moving ≥ ``motion_scale``).
    motion_scale: float = 0.05,
    # Proximity + finger gate (so "close fingers on empty air" doesn't fire).
    # Sized for the 8 × 6 × 6 cm box (see :func:`is_grasped_geometric`).
    handle_capture_radius: float = 0.07,
    finger_close_threshold: float = 0.035,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"]),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
) -> torch.Tensor:
    """Velocity-tracking reward: rewards the cable handle for moving with the
    gripper at the same velocity (proper rigid-body grip behaviour).

    Structure:

    1. **Match term** ``1 − tanh(|v_handle − v_hand| / lin_vel_std)`` — peaks at 1
       when the velocities are identical, decays smoothly as they diverge.
    2. **Motion gate** ``clamp(|v_hand| / motion_scale, 0, 1)`` — the reward
       fades to 0 when the gripper isn't moving, so the policy can't farm this
       by grabbing the handle and sitting still.
    3. **Geometric gate** — handle near EE AND fingers in the gripping range,
       so "close fingers on empty air while gripper flies around" earns nothing.

    A rigid-body grip satisfies all three: the gripper is actively moving, its
    velocity is matched by the handle, and the fingers are clamped on something
    near the EE. A push/flick can match (1) briefly but fails the motion gate's
    sustained check and the geometric gate once the handle separates.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cable: Articulation = env.scene[cable_cfg.name]

    # Velocity match term.
    hand_body_id = _panda_hand_body_id(robot)
    hand_lin_vel_w = wp.to_torch(robot.data.body_link_vel_w)[:, hand_body_id, :3]
    handle_lin_vel_w = _safe_pos(wp.to_torch(cable.data.root_lin_vel_w))
    vel_mismatch = torch.linalg.norm(handle_lin_vel_w - hand_lin_vel_w, dim=1)
    match = 1.0 - torch.tanh(vel_mismatch / lin_vel_std)

    # Motion gate: how fast is the hand moving?
    hand_speed = torch.linalg.norm(hand_lin_vel_w, dim=1)
    motion = torch.clamp(hand_speed / motion_scale, 0.0, 1.0)

    # Geometric gate: handle within reach, fingers clamped on something.
    handle_pos_w = _handle_pos_w(env, cable_cfg)
    ee_pos_w = wp.to_torch(ee_frame.data.target_pos_w)[..., 0, :]
    near = torch.linalg.norm(handle_pos_w - ee_pos_w, dim=1) < handle_capture_radius
    joint_pos = wp.to_torch(robot.data.joint_pos)[:, robot_cfg.joint_ids]
    gripping = ((joint_pos > 0.010) & (joint_pos < finger_close_threshold)).all(dim=-1)
    geom = (near & gripping).float()

    return motion * geom * match


def handle_above_table(
    env: ManagerBasedRLEnv,
    rest_height: float = 0.02,
    max_lift: float = 0.20,
    handle_capture_radius: float = 0.07,
    finger_close_threshold: float = 0.035,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"]),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
) -> torch.Tensor:
    """Reward for the handle sitting above the table while grasped.

    Returns ``is_grasped_geometric * clamp((handle_z − rest_height) / max_lift,
    0, 1)``. Uses the LOOSE geometric grasp gate (proximity + fingers in range)
    rather than the strict velocity-correlation :func:`is_grasped`, so real
    lifts register even though finger contact transiently mismatches velocities.

    Scale: ``max_lift = 0.20 m`` saturates at 20 cm above the table, roughly
    the minimum target altitude (target z range 15-40 cm).
    """
    grasped = is_grasped_geometric(
        env=env,
        handle_capture_radius=handle_capture_radius,
        finger_close_threshold=finger_close_threshold,
        robot_cfg=robot_cfg,
        ee_frame_cfg=ee_frame_cfg,
        cable_cfg=cable_cfg,
    )
    handle_pos_w = _handle_pos_w(env, cable_cfg)
    height_above = torch.clamp(handle_pos_w[:, 2] - rest_height, min=0.0)
    progress = torch.clamp(height_above / max_lift, 0.0, 1.0)
    return grasped * progress


# Kept for reference; the env cfg no longer uses this.
def lift_progress(
    env: ManagerBasedRLEnv,
    rest_height: float = 0.02,
    max_lift: float = 0.3,
    handle_capture_radius: float = 0.08,
    finger_close_threshold: float = 0.020,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"]),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
) -> torch.Tensor:
    """Z-only lift reward. Deprecated in favour of :func:`handle_above_table`."""
    grasped = is_grasped(
        env=env, handle_capture_radius=handle_capture_radius, finger_close_threshold=finger_close_threshold, robot_cfg=robot_cfg, ee_frame_cfg=ee_frame_cfg, cable_cfg=cable_cfg
    )
    handle_pos_w = _handle_pos_w(env, cable_cfg)
    progress = torch.clamp((handle_pos_w[:, 2] - rest_height) / max_lift, 0.0, 1.0)
    return grasped * progress


def handle_target_position_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "handle_pose",
    handle_capture_radius: float = 0.08,
    finger_close_threshold: float = 0.020,
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
        env=env, handle_capture_radius=handle_capture_radius, finger_close_threshold=finger_close_threshold, robot_cfg=robot_cfg, ee_frame_cfg=ee_frame_cfg, cable_cfg=cable_cfg
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
    finger_close_threshold: float = 0.020,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"]),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
) -> torch.Tensor:
    """Stage 4 (orientation): tanh reward for matching target orientation, gated by is_grasped."""
    grasped = is_grasped(
        env=env, handle_capture_radius=handle_capture_radius, finger_close_threshold=finger_close_threshold, robot_cfg=robot_cfg, ee_frame_cfg=ee_frame_cfg, cable_cfg=cable_cfg
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
    finger_close_threshold: float = 0.020,
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
