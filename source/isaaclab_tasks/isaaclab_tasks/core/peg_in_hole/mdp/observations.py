# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply, subtract_frame_transforms

from isaaclab_tasks.core.lift.mdp.utils import sample_object_point_cloud
from isaaclab_tasks.core.utils import cuboid_corner_offsets

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import ObservationTermCfg


def asset_pose_b(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    reference_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return an asset root pose in a reference asset's root frame.

    Args:
        env: The environment.
        asset_cfg: Asset whose root pose is observed.
        reference_cfg: Reference asset.

    Returns:
        Pose ``(x, y, z, qx, qy, qz, qw)`` with shape ``(num_envs, 7)``. Position is in [m].
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    reference: Articulation | RigidObject = env.scene[reference_cfg.name]
    position, orientation = subtract_frame_transforms(
        reference.data.root_pos_w.torch,
        reference.data.root_quat_w.torch,
        asset.data.root_pos_w.torch,
        asset.data.root_quat_w.torch,
    )
    return torch.cat((position, orientation), dim=-1)


class cuboid_corners_b(ManagerTermBase):
    """Ordered cuboid corners expressed in a reference root frame."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        reference_cfg: SceneEntityCfg = cfg.params.get("reference_cfg", SceneEntityCfg("robot"))
        self._asset: RigidObject = env.scene[asset_cfg.name]
        self._reference: Articulation | RigidObject = env.scene[reference_cfg.name]
        self._corners_local = cuboid_corner_offsets(self._asset.cfg.spawn.size, env.device)
        self._visualizer = None
        if cfg.params.get("visualize", True):
            from isaaclab.markers import VisualizationMarkers
            from isaaclab.markers.config import RAY_CASTER_MARKER_CFG

            marker_cfg = RAY_CASTER_MARKER_CFG.replace(prim_path=f"/Visuals/{asset_cfg.name.title()}ObservationCorners")
            marker_cfg.markers["hit"].radius = 0.0025
            self._visualizer = VisualizationMarkers(marker_cfg)

    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg,
        reference_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        flatten: bool = False,
        visualize: bool = True,
    ) -> torch.Tensor:
        """Return ordered cuboid corners [m], optionally flattened."""
        num_corners = self._corners_local.shape[0]
        corners_local = self._corners_local.unsqueeze(0).expand(env.num_envs, -1, -1)
        asset_quat_w = self._asset.data.root_quat_w.torch.unsqueeze(1).expand(-1, num_corners, -1)
        corners_w = quat_apply(asset_quat_w, corners_local) + self._asset.data.root_pos_w.torch.unsqueeze(1)
        reference_pos_w = self._reference.data.root_pos_w.torch.unsqueeze(1).expand(-1, num_corners, -1)
        reference_quat_w = self._reference.data.root_quat_w.torch.unsqueeze(1).expand(-1, num_corners, -1)
        corners_b, _ = subtract_frame_transforms(reference_pos_w, reference_quat_w, corners_w, None)
        if visualize and self._visualizer is not None:
            self._visualizer.visualize(translations=corners_w.reshape(-1, 3))
        return corners_b.reshape(env.num_envs, -1) if flatten else corners_b


class hole_structure_point_cloud_b(ManagerTermBase):
    """Hole surface points with cached world geometry expressed in a reference root frame."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        part_cfgs: list[SceneEntityCfg] = cfg.params["part_cfgs"]
        num_points: int = cfg.params.get("num_points", 32)
        if not part_cfgs or num_points < len(part_cfgs):
            raise ValueError("num_points must provide at least one point per hole part.")

        reference_cfg: SceneEntityCfg = cfg.params.get("reference_cfg", SceneEntityCfg("robot"))
        self._reference: Articulation | RigidObject = env.scene[reference_cfg.name]
        self._parts: list[RigidObject] = [env.scene[part_cfg.name] for part_cfg in part_cfgs]
        self._visualizer = None
        if cfg.params.get("visualize", True):
            from isaaclab.markers import VisualizationMarkers
            from isaaclab.markers.config import RAY_CASTER_MARKER_CFG

            marker_cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/HoleObservationPointCloud")
            marker_cfg.markers["hit"].radius = 0.0025
            self._visualizer = VisualizationMarkers(marker_cfg)

        points_per_part, remainder = divmod(num_points, len(self._parts))
        self._points_local = []
        for index, part in enumerate(self._parts):
            part_num_points = points_per_part + int(index < remainder)
            self._points_local.append(
                sample_object_point_cloud(
                    env.num_envs,
                    part_num_points,
                    part.cfg.prim_path,
                    device=env.device,
                    per_env=cfg.params.get("per_env", False),
                )
            )
        self._num_points = num_points
        self._points_w = torch.empty(env.num_envs, num_points, 3, device=env.device)
        self._update_cache()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Refresh cached points for reset environments."""
        self._update_cache(env_ids)

    def _update_cache(self, env_ids: Sequence[int] | None = None) -> None:
        """Refresh selected cached point clouds."""
        if env_ids is None:
            ids = torch.arange(self._points_w.shape[0], device=self._points_w.device)
        else:
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self._points_w.device)
        if len(ids) == 0:
            return

        points_w = []
        for part, points_local in zip(self._parts, self._points_local):
            part_num_points = points_local.shape[1]
            part_quat_w = part.data.root_quat_w.torch[ids].unsqueeze(1).expand(-1, part_num_points, -1)
            part_pos_w = part.data.root_pos_w.torch[ids].unsqueeze(1)
            points_w.append(quat_apply(part_quat_w, points_local[ids]) + part_pos_w)
        self._points_w[ids] = torch.cat(points_w, dim=1)

    def __call__(
        self,
        env: ManagerBasedEnv,
        part_cfgs: list[SceneEntityCfg],
        reference_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        num_points: int = 32,
        flatten: bool = False,
        visualize: bool = True,
        per_env: bool = False,
    ) -> torch.Tensor:
        """Return the sampled hole structure point cloud.

        Args:
            env: The environment.
            part_cfgs: Hole scene entities in point concatenation order.
            reference_cfg: Reference asset.
            num_points: Total number of sampled points.
            flatten: Whether to flatten the point dimension.
            visualize: Whether to draw markers for the points.
            per_env: Unused at runtime; sampling occurs during initialization.

        Returns:
            Hole surface points [m], shape ``(num_envs, num_points, 3)`` or
            ``(num_envs, 3 * num_points)``.
        """
        if num_points != self._num_points:
            raise ValueError(f"Expected {self._num_points} points, received {num_points}.")

        reference_pos_w = self._reference.data.root_pos_w.torch.unsqueeze(1).expand(-1, num_points, -1)
        reference_quat_w = self._reference.data.root_quat_w.torch.unsqueeze(1).expand(-1, num_points, -1)
        points_b, _ = subtract_frame_transforms(reference_pos_w, reference_quat_w, self._points_w, None)
        if visualize and self._visualizer is not None:
            self._visualizer.visualize(translations=self._points_w.reshape(-1, 3))

        return points_b.reshape(env.num_envs, -1) if flatten else points_b


class hole_structure_corners_b(ManagerTermBase):
    """Ordered corners of hole cuboids expressed in a reference root frame."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        part_cfgs: list[SceneEntityCfg] = cfg.params["part_cfgs"]
        if not part_cfgs:
            raise ValueError("At least one hole part is required.")

        reference_cfg: SceneEntityCfg = cfg.params.get("reference_cfg", SceneEntityCfg("robot"))
        self._reference: Articulation | RigidObject = env.scene[reference_cfg.name]
        self._parts: list[RigidObject] = [env.scene[part_cfg.name] for part_cfg in part_cfgs]
        self._corners_local = [cuboid_corner_offsets(part.cfg.spawn.size, env.device) for part in self._parts]
        self._num_corners = 8 * len(self._parts)
        self._corners_w = torch.empty(env.num_envs, self._num_corners, 3, device=env.device)
        self._visualizer = None
        if cfg.params.get("visualize", True):
            from isaaclab.markers import VisualizationMarkers
            from isaaclab.markers.config import RAY_CASTER_MARKER_CFG

            marker_cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/HoleObservationCorners")
            marker_cfg.markers["hit"].radius = 0.0025
            self._visualizer = VisualizationMarkers(marker_cfg)
        self._update_cache()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Refresh cached corners for reset environments."""
        self._update_cache(env_ids)

    def _update_cache(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            ids = torch.arange(self._corners_w.shape[0], device=self._corners_w.device)
        else:
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self._corners_w.device)
        if len(ids) == 0:
            return

        corners_w = []
        for part, corners_local in zip(self._parts, self._corners_local):
            part_quat_w = part.data.root_quat_w.torch[ids].unsqueeze(1).expand(-1, 8, -1)
            local = corners_local.unsqueeze(0).expand(len(ids), -1, -1)
            corners_w.append(quat_apply(part_quat_w, local) + part.data.root_pos_w.torch[ids].unsqueeze(1))
        self._corners_w[ids] = torch.cat(corners_w, dim=1)

    def __call__(
        self,
        env: ManagerBasedEnv,
        part_cfgs: list[SceneEntityCfg],
        reference_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        flatten: bool = False,
        visualize: bool = True,
    ) -> torch.Tensor:
        """Return part-major ordered hole corners [m], optionally flattened."""
        reference_pos_w = self._reference.data.root_pos_w.torch.unsqueeze(1).expand(-1, self._num_corners, -1)
        reference_quat_w = self._reference.data.root_quat_w.torch.unsqueeze(1).expand(-1, self._num_corners, -1)
        corners_b, _ = subtract_frame_transforms(reference_pos_w, reference_quat_w, self._corners_w, None)
        if visualize and self._visualizer is not None:
            self._visualizer.visualize(translations=self._corners_w.reshape(-1, 3))
        return corners_b.reshape(env.num_envs, -1) if flatten else corners_b
