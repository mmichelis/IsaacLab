# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for the rigid lift tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_apply, transform_points

from isaaclab_tasks.core.lift.mdp.rewards import contacts
from isaaclab_tasks.core.lift.mdp.utils import sample_object_point_cloud, symmetric_point_cloud_distance

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.markers import VisualizationMarkersCfg


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*finger"),
) -> torch.Tensor:
    """Reward the selected robot bodies for reaching the object using a tanh kernel."""
    object: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    object_pos_w = _object_position_w(object, object_cfg)
    body_pos_w = robot.data.body_pos_w.torch[:, robot_cfg.body_ids]
    distance = torch.linalg.norm(body_pos_w - object_pos_w[:, None, :], dim=-1).max(dim=-1).values
    return 1.0 - torch.tanh(distance / std)


def object_lifting(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward raising the object above ``minimal_height`` [m] using a tanh kernel with scale ``std`` [m].

    Dense and ungated, so it fills the gradient gap between reaching and goal tracking with a
    smooth vertical signal. Returns ``0`` at or below ``minimal_height`` and saturates toward ``1``.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    height = _object_position_w(obj, object_cfg)[:, 2] - minimal_height
    return torch.tanh(height.clamp(min=0.0) / std)


class object_fingertip_distance(ManagerTermBase):
    """Reward closing the gripper around the object using a tanh kernel with scale ``std`` [m].

    Each selected fingertip body is rewarded for approaching the nearest point on the object's
    surface, so grasping any part of the object is credited rather than only its center (which
    generalizes to non-compact rigid shapes). Surface points are pre-sampled in the object's
    local frame and transformed to world each step; the per-finger nearest-point proximities are
    averaged. Supplies the grasp gradient that the hand-midpoint reach reward lacks.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        object_cfg: SceneEntityCfg = cfg.params.get("object_cfg", SceneEntityCfg("object"))
        num_points = 32
        obj: RigidObject = env.scene[object_cfg.name]
        # surface points in the object's local frame, shape (num_envs, num_points, 3)
        self._points_local = sample_object_point_cloud(env.num_envs, num_points, obj.cfg.prim_path, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        obj: RigidObject = env.scene[object_cfg.name]
        robot: Articulation = env.scene[robot_cfg.name]
        # object surface points in world frame: (num_envs, num_points, 3)
        num_points = self._points_local.shape[1]
        object_quat_w = obj.data.root_quat_w.torch.unsqueeze(1).repeat(1, num_points, 1)
        points_w = quat_apply(object_quat_w, self._points_local) + obj.data.root_pos_w.torch.unsqueeze(1)
        # nearest surface point to each fingertip body: (num_envs, num_fingers)
        finger_pos_w = robot.data.body_pos_w.torch[:, robot_cfg.body_ids]
        distance = torch.linalg.norm(finger_pos_w.unsqueeze(2) - points_w.unsqueeze(1), dim=3)
        nearest = distance.min(dim=2).values
        return (1.0 - torch.tanh(nearest / std)).mean(dim=1)


class _object_goal_distance(ManagerTermBase):
    """Shared command or target distance term."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        target_cfg: SceneEntityCfg | None = cfg.params.get("target_cfg")
        if target_cfg is None:
            return
        object_cfg: SceneEntityCfg = cfg.params.get("object_cfg", SceneEntityCfg("object"))
        num_points: int = cfg.params.get("num_points", 32)
        obj: RigidObject = env.scene[object_cfg.name]
        self._object_points_local = sample_object_point_cloud(
            env.num_envs, num_points, obj.cfg.prim_path, device=env.device
        )
        self._target_points_local = self._object_points_local

    def _object_grasped(
        self,
        env: ManagerBasedRLEnv,
        contact_threshold: float,
        thumb_name: str | None,
        finger_names: list[str] | None,
    ) -> torch.Tensor:
        if thumb_name is None and finger_names is None:
            return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        if thumb_name is None or finger_names is None:
            raise ValueError("Grasp gating requires both thumb_name and finger_names.")
        return contacts(env, contact_threshold, thumb_name, finger_names)

    def _goal_metrics(
        self,
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg,
        command_name: str | None = None,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        target_cfg: SceneEntityCfg | None = None,
    ) -> torch.Tensor:
        obj: RigidObject = env.scene[object_cfg.name]
        object_pos_w = _object_position_w(obj, object_cfg)
        if (target_cfg is None) == (command_name is None):
            raise ValueError("Exactly one of command_name or target_cfg must be provided.")
        if target_cfg is None:
            robot: RigidObject = env.scene[robot_cfg.name]
            command = env.command_manager.get_command(command_name)
            goal_pos_w, _ = combine_frame_transforms(
                robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, command[:, :3]
            )
            return torch.linalg.norm(goal_pos_w - object_pos_w, dim=1)

        target: RigidObject = env.scene[target_cfg.name]
        object_points_w = transform_points(
            self._object_points_local, obj.data.root_pos_w.torch, obj.data.root_quat_w.torch
        )
        target_points_w = transform_points(
            self._target_points_local, target.data.root_pos_w.torch, target.data.root_quat_w.torch
        )
        distance = symmetric_point_cloud_distance(object_points_w, target_points_w)
        return distance


class object_goal_distance(_object_goal_distance):
    """Reward object-to-goal alignment using a tanh kernel.

    If ``success_threshold`` is provided, this also tracks sticky per-episode success.
    Contact sensor names additionally require bilateral object contact.
    ``minimal_height`` is accepted for compatibility and ignored.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._track_success = cfg.params.get("success_threshold") is not None
        if self._track_success:
            self.succeeded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor):
        if self._track_success:
            self._env.extras.setdefault("log", {})["Metrics/success_rate"] = (
                self.succeeded[env_ids].float().mean().item()
            )
            self.succeeded[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        minimal_height: float | None = None,
        command_name: str | None = None,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        success_threshold: float | None = None,
        target_cfg: SceneEntityCfg | None = None,
        num_points: int = 32,
        contact_threshold: float = 0.01,
        thumb_name: str | None = None,
        finger_names: list[str] | None = None,
    ) -> torch.Tensor:
        distance = self._goal_metrics(env, object_cfg, command_name, robot_cfg, target_cfg)
        grasped = self._object_grasped(env, contact_threshold, thumb_name, finger_names)
        if success_threshold is not None:
            self.succeeded |= grasped & (distance < success_threshold)
        return grasped.float() * (1 - torch.tanh(distance / std))


class object_goal_distance_delta(ManagerTermBase):
    """Reward the agent for moving the object closer to the goal each step.

    Returns the per-step decrease in the object-to-goal distance (previous minus current),
    gated so it only credits while the object is lifted above ``minimal_height``. Moving the
    object toward the goal yields a positive reward and away a negative one. The stored
    distance is re-baselined on the first step after reset so the reset teleport does not
    produce a spurious reward. Success tracking matches :class:`object_goal_distance`.
    If ``object_cfg`` selects one body, that body's pose is used instead of the root pose.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._prev_distance = torch.zeros(env.num_envs, device=env.device)
        # baseline the stored distance on the first call after each reset
        self._needs_baseline = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self._track_success = cfg.params.get("success_threshold") is not None
        if self._track_success:
            self._succeeded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def _goal_distance(
        self, env: ManagerBasedRLEnv, command_name: str, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg
    ):
        robot: RigidObject = env.scene[robot_cfg.name]
        obj: RigidObject = env.scene[object_cfg.name]
        command = env.command_manager.get_command(command_name)
        des_pos_w, _ = combine_frame_transforms(
            robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, command[:, :3]
        )
        object_pos_w = _object_position_w(obj, object_cfg)
        distance = torch.linalg.norm(des_pos_w - object_pos_w, dim=1)
        return distance, object_pos_w

    def reset(self, env_ids: torch.Tensor):
        self._needs_baseline[env_ids] = True
        if self._track_success:
            self._env.extras.setdefault("log", {})["Metrics/success_rate"] = (
                self._succeeded[env_ids].float().mean().item()
            )
            self._succeeded[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        minimal_height: float,
        command_name: str,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        success_threshold: float | None = None,
    ) -> torch.Tensor:
        distance, object_pos_w = self._goal_distance(env, command_name, robot_cfg, object_cfg)
        # freshly reset envs baseline here (after command resampling) so their first delta is zero
        self._prev_distance = torch.where(self._needs_baseline, distance, self._prev_distance)
        self._needs_baseline[:] = False
        is_lifted = object_pos_w[:, 2] > minimal_height
        if success_threshold is not None:
            self._succeeded |= is_lifted & (distance < success_threshold)
        delta = self._prev_distance - distance
        self._prev_distance = distance
        return is_lifted.float() * delta


class object_target_point_cloud_reached(_object_goal_distance):
    """Per-step success bonus for aligning the object and target point clouds."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._success_visualizer = None
        self._success_vis_pos_w = None
        asset_name = cfg.params.get("success_vis_asset_name")
        visualizer_cfg = cfg.params.get("success_visualizer_cfg")
        if (asset_name is None) != (visualizer_cfg is None):
            raise ValueError("Success visualization requires both asset name and visualizer configuration.")
        if asset_name is not None:
            asset = env.scene[asset_name]
            if not isinstance(asset, AssetBaseCfg):
                raise TypeError("Success visualization currently requires a static scene asset.")
            offset = torch.tensor(asset.init_state.pos, device=env.device)
            self._success_vis_pos_w = env.scene.env_origins + offset
            from isaaclab.markers import VisualizationMarkers

            self._success_visualizer = VisualizationMarkers(visualizer_cfg)
            self._success_visualizer.set_visibility(True)
            self._success_visualizer.visualize(self._success_vis_pos_w)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        minimal_height: float | None = None,
        success_threshold: float = 0.05,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
        num_points: int = 32,
        success_vis_asset_name: str | None = None,
        success_visualizer_cfg: VisualizationMarkersCfg | None = None,
        contact_threshold: float = 0.01,
        thumb_name: str | None = None,
        finger_names: list[str] | None = None,
    ) -> torch.Tensor:
        distance = self._goal_metrics(env, object_cfg, target_cfg=target_cfg)
        grasped = self._object_grasped(env, contact_threshold, thumb_name, finger_names)
        reached = grasped & (distance < success_threshold)
        if self._success_visualizer is not None:
            self._success_visualizer.visualize(self._success_vis_pos_w, marker_indices=reached.int())
        return reached.float()


def object_goal_reached(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    command_name: str,
    success_threshold: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Per-step success bonus for holding the object at the goal.

    Returns ``1.0`` while the object is within ``success_threshold`` [m] of the commanded
    goal position and lifted above ``minimal_height`` [m], else ``0.0``. Matches the
    condition tracked as ``Metrics/success_rate`` in :class:`object_goal_distance`.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, command[:, :3])
    object_pos_w = _object_position_w(obj, object_cfg)
    distance = torch.linalg.norm(des_pos_w - object_pos_w, dim=1)
    is_lifted = object_pos_w[:, 2] > minimal_height
    return (is_lifted & (distance < success_threshold)).float()


def _object_position_w(object: RigidObject, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the object's root or single selected body position [m]."""
    if object_cfg.body_names is None and object_cfg.body_ids == slice(None):
        return object.data.root_pos_w.torch

    body_pos_w = object.data.body_pos_w.torch[:, object_cfg.body_ids]
    if body_pos_w.shape[1] != 1:
        raise ValueError("Rigid-object rewards require exactly one selected body.")
    return body_pos_w[:, 0]
