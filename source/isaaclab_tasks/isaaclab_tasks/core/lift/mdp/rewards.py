# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error

from .terminations import _deformable_vertices_in_bounds

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, CableObject, DeformableObject, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ManagerTermBaseCfg
    from isaaclab.sensors import ContactSensor, FrameTransformer


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    thumb_name: str,
    finger_names: list[str],
    contact_threshold: float = 1.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward reaching the object using a tanh-kernel on end-effector distance with contact bonus.

    The reward is close to 1 when the distance is small. The reward is scaled by contact:
    - Full reward (1x) when good contact (thumb + finger)
    - Reduced reward (0.1x) when no contact

    Args:
        env: The environment instance.
        std: Standard deviation for tanh kernel.
        thumb_name: Name of the thumb contact sensor.
        finger_names: Names of the finger contact sensors.
        contact_threshold: Contact force magnitude threshold.
        object_cfg: Configuration for the object.
        asset_cfg: Configuration for the robot asset.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    asset_pos = asset.data.body_pos_w.torch[:, asset_cfg.body_ids]
    object_pos = obj.data.root_pos_w.torch
    distance = torch.linalg.norm(asset_pos - object_pos[:, None, :], dim=-1).max(dim=-1).values
    contact_bonus = contacts(env, contact_threshold, thumb_name, finger_names).float().clamp(0.1, 1.0)
    return (1 - torch.tanh(distance / std)) * contact_bonus


def _contact_force_mag(sensor: ContactSensor, num_envs: int) -> torch.Tensor:
    """Extract per-environment contact force magnitude from a sensor's force_matrix_w."""
    force = sensor.data.force_matrix_w.torch.view(num_envs, 3)
    return torch.linalg.norm(force, dim=-1)


def contacts(env: ManagerBasedRLEnv, threshold: float, thumb_name: str, finger_names: list[str]) -> torch.Tensor:
    """Reward for good contact: thumb + at least one finger above threshold.

    Args:
        env: The environment instance.
        threshold: Contact force magnitude threshold.
        thumb_name: Name of the thumb contact sensor in the scene.
        finger_names: Names of the finger contact sensors in the scene.

    Returns:
        Boolean tensor indicating good contact condition per environment.
    """
    thumb_mag = _contact_force_mag(env.scene.sensors[thumb_name], env.num_envs)

    any_finger_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for finger_name in finger_names:
        finger_mag = _contact_force_mag(env.scene.sensors[finger_name], env.num_envs)
        any_finger_contact = any_finger_contact | (finger_mag > threshold)

    return (thumb_mag > threshold) & any_finger_contact


def contact_count(env: ManagerBasedRLEnv, threshold: float, sensor_names: list[str]) -> torch.Tensor:
    """Count the number of contact sensors with force above threshold.

    For each sensor that detects contact above the threshold, add 1 to the total.
    This provides a reward proportional to the number of fingers in contact.

    Args:
        env: The environment instance.
        threshold: Contact force magnitude threshold.
        sensor_names: Names of the contact sensors in the scene.

    Returns:
        Tensor of shape (num_envs,) with the count of sensors in contact per environment.
    """
    count = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    for sensor_name in sensor_names:
        mag = _contact_force_mag(env.scene.sensors[sensor_name], env.num_envs)
        count += (mag > threshold).float()
    return count / len(sensor_names)


class success_reward(ManagerTermBase):
    """Reward success by comparing commanded pose to the object pose using tanh kernels on error.

    The reward is gated by contact: only given when thumb + at least one finger are in contact.

    Maintains a sticky ``succeeded`` boolean tensor per environment that flips to ``True`` once
    the success condition is met during an episode and resets to ``False`` on environment reset.

    Args:
        cfg: Configuration object specifying term parameters.
        env: The manager-based RL environment.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.succeeded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.succeeded[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg,
        align_asset_cfg: SceneEntityCfg,
        pos_std: float,
        thumb_name: str,
        finger_names: list[str],
        contact_threshold: float = 0.01,
        rot_std: float | None = None,
    ) -> torch.Tensor:
        asset: RigidObject = env.scene[asset_cfg.name]
        obj: RigidObject = env.scene[align_asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        des_pos_w, des_quat_w = combine_frame_transforms(
            asset.data.root_pos_w.torch,
            asset.data.root_quat_w.torch,
            command[:, :3],
            command[:, 3:7],
        )
        pos_err, rot_err = compute_pose_error(
            des_pos_w,
            des_quat_w,
            obj.data.root_pos_w.torch,
            obj.data.root_quat_w.torch,
        )
        pos_dist = torch.linalg.norm(pos_err, dim=1)
        contact_mask = contacts(env, contact_threshold, thumb_name, finger_names)

        if rot_std:
            rot_dist = torch.linalg.norm(rot_err, dim=1)
            reward = (1 - torch.tanh(pos_dist / pos_std)) * (1 - torch.tanh(rot_dist / rot_std)) * contact_mask.float()
            self.succeeded |= (pos_dist < pos_std) & (rot_dist < rot_std) & contact_mask
        else:
            reward = ((1 - torch.tanh(pos_dist / pos_std)) ** 2) * contact_mask.float()
            self.succeeded |= (pos_dist < pos_std) & contact_mask

        return reward


def position_command_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    align_asset_cfg: SceneEntityCfg,
    thumb_name: str,
    finger_names: list[str],
    contact_threshold: float = 0.1,
) -> torch.Tensor:
    """Reward tracking of commanded position using tanh kernel, gated by contact presence.

    .. deprecated::
        Use :class:`position_command_progress`, which pays per increment of ground gained on the
        best error so far instead of paying every step the object is near the goal. Replace
        ``std`` with ``min_improvement``.
    """
    warnings.warn(
        "The reward term 'position_command_error_tanh' is deprecated. Use 'position_command_progress' instead,"
        " replacing 'std' with 'min_improvement'.",
        DeprecationWarning,
        stacklevel=2,
    )
    asset: RigidObject = env.scene[asset_cfg.name]
    obj: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w.torch,
        asset.data.root_quat_w.torch,
        command[:, :3],
    )
    distance = torch.linalg.norm(obj.data.root_pos_w.torch - des_pos_w, dim=1)
    return (1 - torch.tanh(distance / std)) * contacts(env, contact_threshold, thumb_name, finger_names).float()


def orientation_command_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    align_asset_cfg: SceneEntityCfg,
    thumb_name: str,
    finger_names: list[str],
    contact_threshold: float = 0.1,
) -> torch.Tensor:
    """Reward tracking of commanded orientation using tanh kernel, gated by contact presence.

    .. deprecated::
        Use :class:`orientation_command_progress`, which pays per increment of ground gained on the
        best error so far instead of paying every step the object is near the goal. Replace
        ``std`` with ``min_improvement``.
    """
    warnings.warn(
        "The reward term 'orientation_command_error_tanh' is deprecated. Use 'orientation_command_progress' instead,"
        " replacing 'std' with 'min_improvement'.",
        DeprecationWarning,
        stacklevel=2,
    )
    asset: RigidObject = env.scene[asset_cfg.name]
    obj: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_quat_w = math_utils.quat_mul(asset.data.root_link_quat_w.torch, command[:, 3:7])
    quat_distance = math_utils.quat_error_magnitude(obj.data.root_quat_w.torch, des_quat_w)
    return (1 - torch.tanh(quat_distance / std)) * contacts(env, contact_threshold, thumb_name, finger_names).float()


class _ProgressReward(ManagerTermBase):
    """Base class for rewards that only pay out when a tracking error reaches a new episode best.

    The term holds a per-environment bar equal to the smallest error credited so far in the episode.
    A fixed reward of ``1.0`` is paid on every step that pushes the error below that bar by at least
    ``min_improvement``, and nothing is paid otherwise. Since the bar moves only when a reward is
    paid, ground that was already credited cannot be farmed again by backing off and re-approaching,
    and the episodic reward sum counts how many improvements the policy actually made.

    The bar is seeded with the error measured on the first step of an episode, so holding the starting
    pose earns nothing. Progress made while the gating condition is false does not move the bar, so it
    stays claimable once the condition is met again. The bar is measured against the command in force,
    so it is re-seeded whenever the command resamples and never carries across goals.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # inf marks an environment whose bar has not been seeded yet against its current command
        self.best_error = torch.full((env.num_envs,), float("inf"), device=env.device)
        self._prev_command: torch.Tensor | None = None

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.best_error[env_ids] = float("inf")

    def _progress(
        self, error: torch.Tensor, gate: torch.Tensor, min_improvement: float, command: torch.Tensor
    ) -> torch.Tensor:
        """Return 1.0 for the environments that beat their best error under the command in force.

        Args:
            error: Current tracking error per environment.
            gate: Environments allowed to be credited this step.
            min_improvement: Amount the error must beat the bar by to be paid again.
            command: Command being tracked; a change re-seeds the bar for that environment.

        Returns:
            Tensor of shape ``(num_envs,)`` that is ``1.0`` where a payout is due.
        """
        # a resampled command changes the error's reference, so the bar it was measured against no
        # longer applies; dropping it to inf re-seeds from the first error under the new command
        if self._prev_command is None:
            self._prev_command = command.clone()
        else:
            self.best_error[(self._prev_command != command).any(dim=1)] = float("inf")
            self._prev_command.copy_(command)
        unseeded = torch.isinf(self.best_error)
        self.best_error[unseeded] = error[unseeded]
        improved = gate & (error < self.best_error - min_improvement)
        self.best_error[improved] = error[improved]
        return improved.float()


class position_command_progress(_ProgressReward):
    """Reward every step that brings the object closer to the commanded position than ever before.

    See :class:`_ProgressReward` for the progress bookkeeping. The reward is gated by contact, so
    only positional gains made while the thumb and at least one finger are touching are credited.
    """

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg,
        align_asset_cfg: SceneEntityCfg,
        thumb_name: str,
        finger_names: list[str],
        contact_threshold: float = 0.1,
        min_improvement: float = 0.01,
    ) -> torch.Tensor:
        """Compute the progress reward.

        Args:
            env: The environment instance.
            command_name: Name of the pose command to track.
            asset_cfg: Configuration for the asset the command is expressed relative to.
            align_asset_cfg: Configuration for the asset that must reach the command.
            thumb_name: Name of the thumb contact sensor.
            finger_names: Names of the finger contact sensors.
            contact_threshold: Contact force magnitude threshold [N].
            min_improvement: Distance [m] the object must gain on the episode best to be paid again.
        """
        asset: RigidObject = env.scene[asset_cfg.name]
        obj: RigidObject = env.scene[align_asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        des_pos_w, _ = combine_frame_transforms(
            asset.data.root_pos_w.torch,
            asset.data.root_quat_w.torch,
            command[:, :3],
        )
        distance = torch.linalg.norm(obj.data.root_pos_w.torch - des_pos_w, dim=1)
        gate = contacts(env, contact_threshold, thumb_name, finger_names)
        return self._progress(distance, gate, min_improvement, command)


class orientation_command_progress(_ProgressReward):
    """Reward every step that brings the object closer to the commanded orientation than ever before.

    See :class:`_ProgressReward` for the progress bookkeeping. The reward is gated by contact, so
    only rotational gains made while the thumb and at least one finger are touching are credited.
    """

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg,
        align_asset_cfg: SceneEntityCfg,
        thumb_name: str,
        finger_names: list[str],
        contact_threshold: float = 0.1,
        min_improvement: float = 0.05,
    ) -> torch.Tensor:
        """Compute the progress reward.

        Args:
            env: The environment instance.
            command_name: Name of the pose command to track.
            asset_cfg: Configuration for the asset the command is expressed relative to.
            align_asset_cfg: Configuration for the asset that must reach the command.
            thumb_name: Name of the thumb contact sensor.
            finger_names: Names of the finger contact sensors.
            contact_threshold: Contact force magnitude threshold [N].
            min_improvement: Angle [rad] the object must gain on the episode best to be paid again.
        """
        asset: RigidObject = env.scene[asset_cfg.name]
        obj: RigidObject = env.scene[align_asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        des_quat_w = math_utils.quat_mul(asset.data.root_link_quat_w.torch, command[:, 3:7])
        quat_distance = math_utils.quat_error_magnitude(obj.data.root_quat_w.torch, des_quat_w)
        gate = contacts(env, contact_threshold, thumb_name, finger_names)
        return self._progress(quat_distance, gate, min_improvement, command)


def deformable_lifting(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> torch.Tensor:
    """Reward lifting the deformable COM above ``minimal_height`` [m] with a tanh kernel (``std`` [m])."""
    asset: DeformableObject = env.scene[asset_cfg.name]
    com_z = asset.data.root_pos_w.torch[:, 2]
    height = (com_z - minimal_height).clamp(min=0.0)
    return torch.tanh(height / std)


def deformable_table_sliding(
    env: ManagerBasedRLEnv,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
    speed_threshold: float,
    max_speed: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> torch.Tensor:
    """Penalize horizontal deformable motion near the table surface.

    Args:
        env: The environment instance.
        x_bounds: Table bounds along x [m].
        y_bounds: Table bounds along y [m].
        z_bounds: Table surface bounds along z [m].
        speed_threshold: Unpenalized horizontal speed [m/s].
        max_speed: Maximum penalized horizontal speed [m/s].
        asset_cfg: Deformable asset configuration.
    """
    asset: DeformableObject = env.scene[asset_cfg.name]
    nodal_pos = asset.data.nodal_pos_w.torch - env.scene.env_origins.unsqueeze(1)
    speed_xy = torch.linalg.vector_norm(asset.data.nodal_vel_w.torch[..., :2], dim=-1)
    on_table = (
        (nodal_pos[..., 0] >= x_bounds[0])
        & (nodal_pos[..., 0] <= x_bounds[1])
        & (nodal_pos[..., 1] >= y_bounds[0])
        & (nodal_pos[..., 1] <= y_bounds[1])
        & (nodal_pos[..., 2] >= z_bounds[0])
        & (nodal_pos[..., 2] <= z_bounds[1])
    )
    sliding_speed = on_table * (speed_xy - speed_threshold).clamp(min=0.0, max=max_speed)
    log = env.extras.setdefault("log", {})
    log["Metrics/deformable_table_contact_fraction"] = on_table.float().mean().item()
    log["Metrics/deformable_table_sliding_speed"] = sliding_speed.mean().item()
    return sliding_speed.mean(dim=1)


def deformable_vertices_in_bounds_event(
    env: ManagerBasedRLEnv,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
    success_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> torch.Tensor:
    """Return a unit-integral event when enough deformable vertices are inside the bounds."""
    asset: DeformableObject = env.scene[asset_cfg.name]
    nodal_pos = asset.data.nodal_pos_w.torch - env.scene.env_origins.unsqueeze(1)
    vertex_fraction = _deformable_vertices_in_bounds(nodal_pos, x_bounds, y_bounds, z_bounds).float().mean(dim=1)
    return (vertex_fraction >= success_threshold).float() / env.step_dt


def deformable_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward end-effector proximity to the nearest deformable node with a tanh kernel (``std`` [m])."""
    asset: DeformableObject = env.scene[asset_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    nodal_pos_w = asset.data.nodal_pos_w.torch
    ee_w = ee_frame.data.target_pos_w.torch[..., 0, :]
    distance = torch.linalg.norm(nodal_pos_w - ee_w.unsqueeze(1), dim=2).min(dim=1).values
    return 1.0 - torch.tanh(distance / std)


def deformable_com_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward end-effector proximity to the deformable COM with a tanh kernel (``std`` [m])."""
    asset: DeformableObject = env.scene[asset_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    com_w = asset.data.root_pos_w.torch
    ee_w = ee_frame.data.target_pos_w.torch[..., 0, :]
    distance = torch.linalg.norm(com_w - ee_w, dim=1)
    return 1.0 - torch.tanh(distance / std)


def deformable_vertex_distance_to_bounds(
    env: ManagerBasedRLEnv,
    std: float,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> torch.Tensor:
    """Reward mean vertex proximity to environment-frame AABB bounds [m].

    The tanh kernel uses ``std`` [m].
    """
    mean_distance = _deformable_mean_vertex_distance_to_bounds(env, x_bounds, y_bounds, z_bounds, asset_cfg)
    return 1.0 - torch.tanh(mean_distance / std)


class DeformableVertexDistanceToBoundsProgress(ManagerTermBase):
    """Reward signed deformable vertex progress toward environment-frame AABB bounds."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._previous_distance = _deformable_mean_vertex_distance_to_bounds(env, **cfg.params)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        selected = slice(None) if env_ids is None else env_ids
        distance = _deformable_mean_vertex_distance_to_bounds(self._env, **self.cfg.params)
        self._previous_distance[selected] = distance[selected]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        z_bounds: tuple[float, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    ) -> torch.Tensor:
        """Return signed mean-distance progress [m/s]."""
        distance = _deformable_mean_vertex_distance_to_bounds(env, x_bounds, y_bounds, z_bounds, asset_cfg)
        progress = self._previous_distance - distance
        self._previous_distance.copy_(distance)
        log = env.extras.setdefault("log", {})
        log["Metrics/deformable_mean_vertex_distance_to_bounds"] = distance.mean().item()
        log["Metrics/deformable_vertex_distance_progress"] = progress.mean().item()
        return progress / env.step_dt


class DeformableVertexFractionInBounds(ManagerTermBase):
    """Reward deformable vertex occupancy inside an AABB and log episode success."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._succeeded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._env.extras.setdefault("log", {})["Metrics/success_rate"] = self._succeeded[env_ids].float().mean().item()
        self._succeeded[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        z_bounds: tuple[float, float],
        success_threshold: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    ) -> torch.Tensor:
        """Return the fraction of vertices inside the inclusive environment-frame bounds."""
        asset: DeformableObject = env.scene[asset_cfg.name]
        nodal_pos = asset.data.nodal_pos_w.torch - env.scene.env_origins.unsqueeze(1)
        vertex_count = _deformable_vertices_in_bounds(nodal_pos, x_bounds, y_bounds, z_bounds).sum(dim=1)
        vertex_fraction = vertex_count.float() / nodal_pos.shape[1]
        self._succeeded |= vertex_fraction >= success_threshold
        log = env.extras.setdefault("log", {})
        log["Metrics/deformable_vertices_in_bounds"] = vertex_count.float().mean().item()
        log["Metrics/deformable_vertex_fraction_in_bounds"] = vertex_fraction.mean().item()
        return vertex_fraction


class DeformableAreaFractionInBounds(ManagerTermBase):
    """Measure lumped reference-area-weighted vertex occupancy inside an AABB."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._output: Literal["fraction", "event", "success"] = cfg.params.get("output", "fraction")
        if self._output not in ("fraction", "event", "success"):
            raise ValueError(f"Unsupported area occupancy output: {self._output!r}.")
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("deformable"))
        asset: DeformableObject = env.scene[asset_cfg.name]
        triangle_indices = _load_surface_triangle_indices(cfg.params["mesh_path"], env.device)
        rest_pos = asset.data.default_nodal_state_w.torch[0, :, :3]
        if triangle_indices.numel() == 0 or triangle_indices.max().item() >= rest_pos.shape[0]:
            raise ValueError("Surface mesh topology does not match the deformable nodal state.")
        triangle_pos = rest_pos[triangle_indices]
        triangle_area = 0.5 * torch.linalg.vector_norm(
            torch.linalg.cross(triangle_pos[:, 1] - triangle_pos[:, 0], triangle_pos[:, 2] - triangle_pos[:, 0], dim=1),
            dim=1,
        )
        self._area_weights = torch.zeros(rest_pos.shape[0], device=env.device)
        self._area_weights.scatter_add_(0, triangle_indices.flatten(), triangle_area.repeat_interleave(3) / 3.0)
        total_area = self._area_weights.sum()
        if total_area <= 0.0:
            raise ValueError("Surface mesh has zero reference area.")
        self._area_weights /= total_area
        self._succeeded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        if self._output == "fraction":
            self._env.extras.setdefault("log", {})["Metrics/success_rate"] = (
                self._succeeded[env_ids].float().mean().item()
            )
        self._succeeded[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        z_bounds: tuple[float, float],
        success_threshold: float,
        mesh_path: str,
        output: Literal["fraction", "event", "success"] = "fraction",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    ) -> torch.Tensor:
        """Return area fraction, unit-integral success event, or success state."""
        del mesh_path, output
        asset: DeformableObject = env.scene[asset_cfg.name]
        nodal_pos = asset.data.nodal_pos_w.torch - env.scene.env_origins.unsqueeze(1)
        in_bounds = _deformable_vertices_in_bounds(nodal_pos, x_bounds, y_bounds, z_bounds)
        area_fraction = (in_bounds * self._area_weights).sum(dim=1)
        succeeded = area_fraction >= success_threshold
        self._succeeded |= succeeded
        if self._output == "event":
            return succeeded.float() / env.step_dt
        if self._output == "success":
            return succeeded
        env.extras.setdefault("log", {})["Metrics/deformable_area_fraction_in_bounds"] = area_fraction.mean().item()
        return area_fraction


def _load_surface_triangle_indices(mesh_path: str, device: str) -> torch.Tensor:
    """Load triangular surface topology from a USD mesh."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(mesh_path)
    if stage is None:
        raise ValueError(f"Failed to open surface mesh USD: {mesh_path}.")
    mesh_prims = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    if len(mesh_prims) != 1:
        raise ValueError(f"Expected one surface mesh in {mesh_path}, found {len(mesh_prims)}.")
    mesh = UsdGeom.Mesh(mesh_prims[0])
    face_counts = mesh.GetFaceVertexCountsAttr().Get()
    if face_counts is None or any(count != 3 for count in face_counts):
        raise ValueError("Area occupancy requires a triangular surface mesh.")
    face_indices = mesh.GetFaceVertexIndicesAttr().Get()
    return torch.tensor(face_indices, dtype=torch.long, device=device).reshape(-1, 3)


def _deformable_mean_vertex_distance_to_bounds(
    env: ManagerBasedRLEnv,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return mean vertex distance to environment-frame AABB bounds [m]."""
    asset: DeformableObject = env.scene[asset_cfg.name]
    nodal_pos = asset.data.nodal_pos_w.torch - env.scene.env_origins.unsqueeze(1)
    lower = nodal_pos.new_tensor([x_bounds[0], y_bounds[0], z_bounds[0]])
    upper = nodal_pos.new_tensor([x_bounds[1], y_bounds[1], z_bounds[1]])
    closest = torch.maximum(torch.minimum(nodal_pos, upper), lower)
    return torch.linalg.norm(nodal_pos - closest, dim=2).mean(dim=1)


def _deformable_com_goal_metrics(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute deformable COM goal distance and lifted state."""
    robot: Articulation = env.scene[robot_cfg.name]
    asset: DeformableObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, command[:, :3])
    com_w = asset.data.root_pos_w.torch
    return torch.linalg.norm(des_pos_w - com_w, dim=1), com_w[:, 2] > minimal_height


class DeformableComGoalDistance(ManagerTermBase):
    """Reward deformable COM goal tracking and log episode success."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._succeeded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._env.extras.setdefault("log", {})["Metrics/success_rate"] = self._succeeded[env_ids].float().mean().item()
        self._succeeded[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        minimal_height: float,
        command_name: str,
        success_threshold: float,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    ) -> torch.Tensor:
        distance, is_lifted = _deformable_com_goal_metrics(env, minimal_height, command_name, robot_cfg, asset_cfg)
        self._succeeded |= is_lifted & (distance < success_threshold)
        return is_lifted.float() * (1.0 - torch.tanh(distance / std))


def deformable_com_goal_reached(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    command_name: str,
    success_threshold: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> torch.Tensor:
    """Reward the deformable COM for reaching the lifted goal."""
    distance, is_lifted = _deformable_com_goal_metrics(env, minimal_height, command_name, robot_cfg, asset_cfg)
    return (is_lifted & (distance < success_threshold)).float()


def gripper_close_action(env: ManagerBasedRLEnv, action_name: str = "gripper_action") -> torch.Tensor:
    """Return one when the binary gripper action commands closing and zero otherwise."""
    gripper_action = env.action_manager.get_term(action_name).raw_actions
    return torch.any(gripper_action < 0.0, dim=1).float()


def cable_lifting(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
) -> torch.Tensor:
    """Reward the average cable height above minimal_height [m] with a tanh kernel (std [m])."""
    asset: CableObject = env.scene[asset_cfg.name]
    mean_z = asset.data.segment_pose_w.torch[..., 2].mean(dim=1)
    height = (mean_z - minimal_height).clamp(min=0.0)
    return torch.tanh(height / std)


def cable_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward end-effector proximity to the nearest cable segment with a tanh kernel (std [m])."""
    asset: CableObject = env.scene[asset_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    segment_pos_w = asset.data.segment_pose_w.torch[..., :3]
    ee_pos_w = ee_frame.data.target_pos_w.torch[..., 0, :]
    distance = torch.linalg.norm(segment_pos_w - ee_pos_w.unsqueeze(1), dim=2).min(dim=1).values
    return 1.0 - torch.tanh(distance / std)


def _cable_segment_goal_metrics(
    env: ManagerBasedRLEnv,
    command_name: str,
    segment_index: int,
    robot_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Compute cable segment goal distance."""
    robot: Articulation = env.scene[robot_cfg.name]
    asset: CableObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    desired_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, command[:, :3]
    )
    segment_pos_w = asset.data.segment_pose_w.torch[:, segment_index, :3]
    return torch.linalg.norm(desired_pos_w - segment_pos_w, dim=1)


class CableSegmentGoalDistance(ManagerTermBase):
    """Reward cable segment goal tracking and log episode success."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._succeeded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._env.extras.setdefault("log", {})["Metrics/success_rate"] = self._succeeded[env_ids].float().mean().item()
        self._succeeded[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        success_threshold: float,
        segment_index: int,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        asset_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
    ) -> torch.Tensor:
        distance = _cable_segment_goal_metrics(env, command_name, segment_index, robot_cfg, asset_cfg)
        self._succeeded |= distance < success_threshold
        return 1.0 - torch.tanh(distance / std)


def cable_segment_goal_reached(
    env: ManagerBasedRLEnv,
    command_name: str,
    success_threshold: float,
    segment_index: int,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
) -> torch.Tensor:
    """Reward a cable segment for reaching the goal."""
    distance = _cable_segment_goal_metrics(env, command_name, segment_index, robot_cfg, asset_cfg)
    return (distance < success_threshold).float()
