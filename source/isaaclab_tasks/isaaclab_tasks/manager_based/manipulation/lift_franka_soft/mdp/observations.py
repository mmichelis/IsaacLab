# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation functions for the Franka deformable lifting environment."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, DeformableObject
    from isaaclab.envs import ManagerBasedRLEnv


def finite_tensor(obs: torch.Tensor, kwargs: dict[str, float] | None = None) -> torch.Tensor:
    """Replace non-finite observation values with finite fallbacks.

    Args:
        obs: Observation tensor.
        kwargs: Optional replacement values keyed by ``nan``, ``posinf``, and ``neginf``.

    Returns:
        Sanitized observation tensor.
    """
    kwargs = kwargs or {}
    return torch.nan_to_num(
        obs,
        nan=kwargs.get("nan", 0.0),
        posinf=kwargs.get("posinf", 0.0),
        neginf=kwargs.get("neginf", 0.0),
    )


def deformable_com_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Position of the deformable object's COM in the robot's root frame [m].

    The COM is the mean of the deformable's nodal positions (see
    :attr:`~isaaclab.assets.DeformableObject.data.root_pos_w`).

    Returns:
        Tensor of shape ``(num_envs, 3)``.
    """
    asset: DeformableObject = env.scene[asset_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    com_w = wp.to_torch(asset.data.root_pos_w)
    com_b, _ = subtract_frame_transforms(wp.to_torch(robot.data.root_pos_w), wp.to_torch(robot.data.root_quat_w), com_w)
    return com_b


def end_effector_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Position of the end-effector frame in the robot's root frame [m].

    Args:
        env: The environment instance.
        ee_frame_cfg: The end-effector frame entity.
        robot_cfg: The robot entity providing the reference frame.

    Returns:
        Tensor of shape ``(num_envs, 3)``.
    """
    ee_frame = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w.torch[..., 0, :]
    ee_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, ee_pos_w)
    return ee_pos_b


def end_effector_to_deformable_com(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Vector from the end-effector frame to the deformable COM in the robot's root frame [m].

    Args:
        env: The environment instance.
        asset_cfg: The deformable object entity.
        ee_frame_cfg: The end-effector frame entity.
        robot_cfg: The robot entity providing the reference frame.

    Returns:
        Tensor of shape ``(num_envs, 3)``.
    """
    return deformable_com_in_robot_root_frame(env, asset_cfg, robot_cfg) - end_effector_position_in_robot_root_frame(
        env, ee_frame_cfg, robot_cfg
    )


def deformable_com_to_goal(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Vector from the deformable COM to the commanded goal in the robot's root frame [m].

    Args:
        env: The environment instance.
        command_name: Name of the generated pose command.
        asset_cfg: The deformable object entity.
        robot_cfg: The robot entity providing the reference frame.

    Returns:
        Tensor of shape ``(num_envs, 3)``.
    """
    command = env.command_manager.get_command(command_name)
    return command[:, :3] - deformable_com_in_robot_root_frame(env, asset_cfg, robot_cfg)


def scripted_grasp_action_target(
    env: ManagerBasedRLEnv,
    command_name: str,
    action_scale: tuple[float, float, float],
    grasp_height_offset: float = 0.0,
    hover_height_offset: float = 0.12,
    xy_close_distance: float = 0.04,
    close_distance: float = 0.06,
    closed_finger_position: float = 0.035,
    lift_height: float = 0.065,
    vertical_action_limit: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Scripted reach-close-lift target action for policy guidance.

    Args:
        env: The environment instance.
        command_name: Name of the generated pose command.
        action_scale: Per-axis IK action scale used to normalize position targets [m].
        grasp_height_offset: Vertical offset from the COM to the grasp point [m].
        hover_height_offset: Vertical offset from the COM for the pre-grasp hover point [m].
        xy_close_distance: Horizontal distance at which the target switches from hover to descent [m].
        close_distance: Distance at which the target switches from reaching to closing [m].
        closed_finger_position: Finger joint position treated as ready to lift [m].
        lift_height: Deformable COM height at which the target switches to goal tracking [m].
        vertical_action_limit: Absolute raw z action limit for scripted arm targets.
        asset_cfg: The deformable object entity.
        ee_frame_cfg: The end-effector frame entity.
        robot_cfg: The robot entity.

    Returns:
        Tensor of shape ``(num_envs, 4)`` containing target raw arm xyz and
        gripper actions.
    """
    asset: DeformableObject = env.scene[asset_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]

    root_pos_w = wp.to_torch(robot.data.root_pos_w)
    root_quat_w = wp.to_torch(robot.data.root_quat_w)
    com_w = wp.to_torch(asset.data.root_pos_w)
    com_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, com_w)
    grasp_b = com_b.clone()
    grasp_b[:, 2] += grasp_height_offset
    hover_b = com_b.clone()
    hover_b[:, 2] += hover_height_offset
    ee_w = wp.to_torch(ee_frame.data.target_pos_w)[..., 0, :]
    ee_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, ee_w)
    to_grasp = grasp_b - ee_b
    to_hover = hover_b - ee_b
    xy_distance = torch.linalg.norm(to_grasp[:, :2], dim=1)
    grasp_distance = torch.linalg.norm(to_grasp, dim=1)

    command = env.command_manager.get_command(command_name)
    to_goal = command[:, :3] - ee_b

    scale = torch.tensor(action_scale, device=to_grasp.device, dtype=to_grasp.dtype).unsqueeze(0)
    reach_target = torch.where((xy_distance > xy_close_distance).unsqueeze(1), to_hover, to_grasp)
    reach_action = torch.clamp(reach_target / scale, min=-1.0, max=1.0)
    lift_action = torch.zeros_like(reach_action)
    lift_action[:, 2] = 1.0
    goal_action = torch.clamp(to_goal / scale, min=-1.0, max=1.0)
    reach_action[:, 2] = torch.clamp(reach_action[:, 2], min=-vertical_action_limit, max=vertical_action_limit)
    lift_action[:, 2] = torch.clamp(lift_action[:, 2], min=-vertical_action_limit, max=vertical_action_limit)
    goal_action[:, 2] = torch.clamp(goal_action[:, 2], min=-vertical_action_limit, max=vertical_action_limit)

    finger_ids, _ = robot.find_joints(["panda_finger.*"])
    finger_position = robot.data.joint_pos.torch[:, finger_ids].mean(dim=1)
    near = grasp_distance < close_distance
    ready_to_lift = near & (finger_position < closed_finger_position)
    lifted = com_w[:, 2] > lift_height

    arm_action = reach_action
    arm_action = torch.where((ready_to_lift & ~lifted).unsqueeze(1), lift_action, arm_action)
    arm_action = torch.where(lifted.unsqueeze(1), goal_action, arm_action)

    gripper_action = torch.where(near | lifted, -torch.ones_like(grasp_distance), torch.ones_like(grasp_distance))
    return torch.cat((arm_action, gripper_action.unsqueeze(1)), dim=1)


class DeformableSampledPointsInRobotRootFrame(ManagerTermBase):
    """Sampled deformable nodal points expressed in the robot's root frame.

    The point indices are sampled on reset, then reused within the episode so
    each observed point follows the same material node over time.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("deformable"))
        self.robot_cfg: SceneEntityCfg = cfg.params.get("robot_cfg", SceneEntityCfg("robot"))
        self.num_points: int = cfg.params.get("num_points", 20)

        asset: DeformableObject = env.scene[self.asset_cfg.name]
        self.num_nodes = asset.data.nodal_pos_w.shape[1]
        self.node_ids = torch.empty(env.num_envs, self.num_points, dtype=torch.long, device=env.device)
        self.reset()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Resample observed deformable nodes for the selected environments."""
        if env_ids is None:
            env_ids = slice(None)
            num_envs = self.num_envs
        else:
            num_envs = len(env_ids)

        if self.num_points <= self.num_nodes:
            self.node_ids[env_ids] = torch.rand((num_envs, self.num_nodes), device=self.device).topk(
                self.num_points, dim=1
            ).indices
        else:
            self.node_ids[env_ids] = torch.randint(self.num_nodes, (num_envs, self.num_points), device=self.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        num_points: int = 20,
    ) -> torch.Tensor:
        """Sample deformable nodal positions in the robot's root frame.

        Args:
            env: The environment instance.
            asset_cfg: The deformable object entity.
            robot_cfg: The robot entity providing the reference frame.
            num_points: Number of sampled points.

        Returns:
            Flattened tensor of shape ``(num_envs, 3 * num_points)`` with sampled
            point positions [m] in the robot root frame.
        """
        asset: DeformableObject = env.scene[asset_cfg.name]
        robot: Articulation = env.scene[robot_cfg.name]
        if num_points != self.num_points:
            raise ValueError(
                f"Requested {num_points} deformable points, but this term was initialized with {self.num_points}."
            )

        nodal_pos_w = asset.data.nodal_pos_w.torch
        sampled_points_w = nodal_pos_w.gather(1, self.node_ids.unsqueeze(-1).expand(-1, -1, 3))

        flat_sampled_points_w = sampled_points_w.reshape(-1, 3)
        root_pos_w = robot.data.root_pos_w.torch.unsqueeze(1).expand(-1, num_points, -1)
        root_quat_w = robot.data.root_quat_w.torch.unsqueeze(1).expand(-1, num_points, -1)
        sampled_points_b, _ = subtract_frame_transforms(
            root_pos_w.reshape(-1, 3),
            root_quat_w.reshape(-1, 4),
            flat_sampled_points_w,
        )
        return sampled_points_b.view(env.num_envs, -1)
