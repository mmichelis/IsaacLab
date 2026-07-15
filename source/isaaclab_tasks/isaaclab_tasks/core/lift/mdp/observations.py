# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation functions for the rigid and deformable lift tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_from_matrix, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, DeformableObject, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w.torch[:, :3]
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, object_pos_w)
    return object_pos_b


def object_orientation_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The orientation of the object in the robot's root frame as a quaternion ``(x, y, z, w)``."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    _, object_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w.torch,
        robot.data.root_quat_w.torch,
        object.data.root_pos_w.torch[:, :3],
        object.data.root_quat_w.torch,
    )
    return object_quat_b


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


class DeformableSampledPointsInRobotRootFrame(ManagerTermBase):
    """Sampled deformable nodal points expressed in the robot's root frame.

    The point indices are sampled on reset, then reused within the episode so
    each observed point follows the same material node over time.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
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
            self.node_ids[env_ids] = (
                torch.rand((num_envs, self.num_nodes), device=self.device).topk(self.num_points, dim=1).indices
            )
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


class DeformableOrientationInRobotRootFrame(ManagerTermBase):
    """Best-fit orientation of a deformable object in the robot's root frame as a quaternion.

    Reconstructs a rigid orientation for a (near-rigid) deformable body by fitting the rotation
    that best aligns a fixed set of sampled material vertices from their rest configuration to
    their current positions (Kabsch / orthogonal Procrustes). This lets a policy trained on a
    rigid object's ``object_orientation`` observation be reused with a deformable object.

    The sampled node indices and their rest offsets are cached on construction and shared across
    environments (the rest shape is identical per env), so each observed vertex tracks the same
    material node over time. More points give a more robust fit.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("object"))
        self.robot_cfg: SceneEntityCfg = cfg.params.get("robot_cfg", SceneEntityCfg("robot"))
        self.num_points: int = cfg.params.get("num_points", 8)

        asset: DeformableObject = env.scene[self.asset_cfg.name]
        num_nodes = asset.data.nodal_pos_w.shape[1]
        if not 4 <= self.num_points <= num_nodes:
            raise ValueError(
                f"DeformableOrientationInRobotRootFrame needs 4..{num_nodes} points, got {self.num_points}."
            )
        # Fixed sampled nodes shared across envs (the rest shape is identical per env).
        self.node_ids = torch.randperm(num_nodes, device=env.device)[: self.num_points]
        # Rest offsets of the sampled nodes about their centroid (rest orientation is identity).
        rest_points = asset.data.default_nodal_state_w.torch[0, self.node_ids, :3]
        self.rest_offsets = rest_points - rest_points.mean(dim=0, keepdim=True)  # (num_points, 3)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        num_points: int = 8,
    ) -> torch.Tensor:
        """Best-fit deformable orientation in the robot's root frame.

        Args:
            env: The environment instance.
            asset_cfg: The deformable object entity.
            robot_cfg: The robot entity providing the reference frame.
            num_points: Number of sampled vertices used to fit the frame.

        Returns:
            Quaternion ``(x, y, z, w)`` of shape ``(num_envs, 4)``.
        """
        asset: DeformableObject = env.scene[asset_cfg.name]
        robot: Articulation = env.scene[robot_cfg.name]
        if num_points != self.num_points:
            raise ValueError(
                f"Requested {num_points} deformable points, but this term was initialized with {self.num_points}."
            )

        # Current sampled vertices centered on their centroid: (num_envs, num_points, 3).
        sampled_w = asset.data.nodal_pos_w.torch[:, self.node_ids, :]
        cur_offsets = sampled_w - sampled_w.mean(dim=1, keepdim=True)

        # Kabsch: rotation aligning rest offsets to current offsets. cov = sum_k rest_k cur_k^T.
        cov = torch.einsum("ki,nkj->nij", self.rest_offsets, cur_offsets)
        u, _, vh = torch.linalg.svd(cov)
        v = vh.transpose(-2, -1)
        ut = u.transpose(-2, -1)
        # Proper-rotation (det +1) correction: flip the sign of the last column of V.
        signs = torch.ones_like(u[:, 0, :])
        signs[:, -1] = torch.sign(torch.linalg.det(torch.matmul(v, ut)))
        rot = torch.matmul(v * signs.unsqueeze(1), ut)
        x = torch.nn.functional.normalize(rot[:, :, 0], dim=-1)
        y = rot[:, :, 1]
        y = torch.nn.functional.normalize(y - torch.sum(x * y, dim=-1, keepdim=True) * x, dim=-1)
        rot = torch.stack((x, y, torch.linalg.cross(x, y)), dim=-1)
        object_quat_w = quat_from_matrix(rot)

        _, object_quat_b = subtract_frame_transforms(
            wp.to_torch(robot.data.root_pos_w),
            wp.to_torch(robot.data.root_quat_w),
            wp.to_torch(asset.data.root_pos_w),
            object_quat_w,
        )
        return object_quat_b
