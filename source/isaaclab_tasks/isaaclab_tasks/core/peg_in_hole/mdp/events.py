# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply

from isaaclab_tasks.core.lift.mdp.events import grasp_travel_distance, reset_joints_shared_offset

from .events_cfg import GraspTravelOpeningCfg


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
