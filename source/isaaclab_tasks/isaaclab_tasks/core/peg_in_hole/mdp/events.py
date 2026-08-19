# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply, sample_uniform

from isaaclab_tasks.contrib.franka_pour.geometry import oriented_boxes_overlap
from isaaclab_tasks.core.lift.mdp.events import grasp_travel_distance, reset_joints_shared_offset

from .events_cfg import GraspTravelOpeningCfg


def reset_hole_from_target(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    part_offsets: dict[str, tuple[float, float, float]],
    depth_range: tuple[float, float],
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> None:
    """Place the hole behind the target along its local z-axis.

    Args:
        env: The environment.
        env_ids: Environments to reset.
        part_offsets: Part offsets at the fully inserted target pose [m].
        depth_range: Target-to-hole pose offset range [m].
        target_cfg: Target asset defining the hole frame.
    """
    lower, upper = depth_range
    if lower > upper:
        raise ValueError("depth_range lower bound must not exceed upper bound.")

    target = env.scene[target_cfg.name]
    target_pose = target.data.root_link_pose_w.torch[env_ids]
    depth = sample_uniform(lower, upper, (len(env_ids),), device=env.device)
    velocity = torch.zeros(len(env_ids), 6, device=env.device, dtype=target_pose.dtype)
    for name, offset in part_offsets.items():
        local_offset = torch.tensor(offset, device=env.device, dtype=target_pose.dtype).expand(len(env_ids), 3).clone()
        local_offset[:, 2] -= depth
        pose = target_pose.clone()
        pose[:, :3] += quat_apply(target_pose[:, 3:7], local_offset)
        part = env.scene[name]
        part.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        part.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=env_ids)


def rigid_object_box_clearance(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    object_name: str,
    obstacle_names: list[str],
    min_clearance: float,
) -> torch.Tensor:
    """Return whether a cuboid rigid object clears every cuboid obstacle."""
    obj = env.scene[object_name]
    object_pose = obj.data.root_link_pose_w.torch[env_ids]
    object_half = torch.as_tensor(obj.cfg.spawn.size, device=env.device, dtype=object_pose.dtype) / 2
    valid = torch.ones(len(env_ids), dtype=torch.bool, device=env.device)
    for name in obstacle_names:
        obstacle = env.scene[name]
        obstacle_pose = obstacle.data.root_link_pose_w.torch[env_ids]
        obstacle_half = torch.as_tensor(obstacle.cfg.spawn.size, device=env.device, dtype=object_pose.dtype) / 2
        valid &= ~oriented_boxes_overlap(
            object_pose[:, :3],
            object_pose[:, 3:7],
            object_half,
            obstacle_pose[:, :3],
            obstacle_pose[:, 3:7],
            obstacle_half,
            min_clearance,
        )
    return valid


def rigid_objects_above_plane(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    object_names: list[str],
    height: float,
    min_clearance: float,
) -> torch.Tensor:
    """Return whether oriented cuboids clear a horizontal plane."""
    valid = torch.ones(len(env_ids), dtype=torch.bool, device=env.device)
    origin_z = env.scene.env_origins[env_ids, 2]
    for name in object_names:
        obj = env.scene[name]
        pose = obj.data.root_link_pose_w.torch[env_ids]
        half = torch.as_tensor(obj.cfg.spawn.size, device=env.device, dtype=pose.dtype) / 2
        vertical_radius = (matrix_from_quat(pose[:, 3:7]).abs()[:, 2] * half).sum(-1)
        valid &= pose[:, 2] - origin_z - vertical_radius >= height + min_clearance
    return valid


def reset_object_and_target_in_gripper(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    probability: float,
    hand_offset: tuple[float, float, float],
    gripper_position_range: tuple[float, float],
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    object_bottom_height_range: tuple[float, float] = (0.001, 0.005),
) -> None:
    """Place the object and target together at a hand-relative pose.

    Args:
        env: The environment.
        env_ids: Environments to reset.
        probability: Independent selection probability per environment.
        hand_offset: Object position in the hand frame [m].
        gripper_position_range: Finger offsets from their default positions [m].
        robot_cfg: Robot and hand body to use as the reference.
        object_cfg: Object asset to place.
        target_cfg: Target asset to place at the same pose.
        object_bottom_height_range: Allowed object bottom height above the table [m].
    """
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    target = env.scene[target_cfg.name]
    hand_pos = robot.data.body_pos_w.torch[env_ids, robot_cfg.body_ids[0]]
    hand_quat = robot.data.body_quat_w.torch[env_ids, robot_cfg.body_ids[0]]
    offset = torch.tensor(hand_offset, device=env.device, dtype=hand_pos.dtype).expand(len(env_ids), 3)
    root_pos = hand_pos + quat_apply(hand_quat, offset)
    half_extents = torch.as_tensor(obj.cfg.spawn.size, device=env.device, dtype=hand_pos.dtype) / 2
    vertical_radius = (matrix_from_quat(hand_quat).abs()[:, 2] * half_extents).sum(-1)
    bottom_height = root_pos[:, 2] - env.scene.env_origins[env_ids, 2] - vertical_radius
    lower, upper = object_bottom_height_range
    eligible = (bottom_height >= lower) & (bottom_height <= upper)
    picked_local = torch.nonzero(eligible & (torch.rand(len(env_ids), device=env.device) < probability)).view(-1)
    if len(picked_local) == 0:
        return
    picked = env_ids[picked_local]
    reset_joints_shared_offset(env, picked, gripper_position_range, robot_cfg)
    root_pose = torch.cat([root_pos[picked_local], hand_quat[picked_local]], dim=-1)
    root_velocity = torch.zeros(len(picked), 6, device=env.device, dtype=hand_pos.dtype)
    for asset in (obj, target):
        asset.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=picked)
        asset.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=picked)


class grasp_travel_opening(grasp_travel_distance):
    """Measure grasp distance, travel distance, and total gripper opening."""

    cfg: GraspTravelOpeningCfg

    def __init__(self, cfg: GraspTravelOpeningCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._gripper_joint_ids = self._robot.find_joints(cfg.gripper_joint_names)[0]

    def __call__(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> torch.Tensor:
        feature = super().__call__(env, env_ids)
        opening = self._robot.data.joint_pos.torch[env_ids][:, self._gripper_joint_ids].abs().sum(-1, keepdim=True)
        return torch.cat([feature, opening], dim=-1)
