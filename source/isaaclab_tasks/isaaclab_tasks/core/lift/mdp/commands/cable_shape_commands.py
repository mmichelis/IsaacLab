# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import CommandTerm
from isaaclab.utils.math import combine_frame_transforms, quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, CableObject
    from isaaclab.envs import ManagerBasedEnv

    from .cable_shape_commands_cfg import CableShapeCommandCfg


class CableShapeCommand(CommandTerm):
    """Planar position targets for every cable segment."""

    cfg: CableShapeCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: CableShapeCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.cable: CableObject = env.scene[cfg.object_name]
        self.success_vis_asset: AssetBaseCfg = env.scene[cfg.success_vis_asset_name]
        self.num_segments = self.cable.num_segments
        if self.num_segments < 2:
            raise ValueError(f"CableShapeCommand requires at least two cable segments, received {self.num_segments}.")
        if cfg.segment_length <= 0.0:
            raise ValueError(f"segment_length must be positive, received {cfg.segment_length}.")
        if cfg.max_turn_angle <= 0.0:
            raise ValueError(f"max_turn_angle must be positive, received {cfg.max_turn_angle}.")
        if cfg.max_sampling_attempts < 1:
            raise ValueError(f"max_sampling_attempts must be positive, received {cfg.max_sampling_attempts}.")
        if cfg.target_z < 0.0:
            raise ValueError(f"target_z must be non-negative, received {cfg.target_z}.")
        for name, bounds in (
            ("pos_x", cfg.ranges.pos_x),
            ("pos_y", cfg.ranges.pos_y),
            ("heading", cfg.ranges.heading),
        ):
            if bounds[0] > bounds[1]:
                raise ValueError(f"{name} range lower bound must not exceed its upper bound, received {bounds}.")
        for name, bounds in zip(("x", "y"), cfg.target_xy_bounds):
            if bounds[0] > bounds[1]:
                raise ValueError(f"target {name} bounds must be increasing, received {bounds}.")

        self.target_positions_b = torch.zeros(self.num_envs, self.num_segments, 3, device=self.device)
        self.target_positions_w = torch.zeros_like(self.target_positions_b)
        self.metrics["mean_position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["max_position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["reset_distance"] = torch.zeros(self.num_envs, device=self.device)
        self._marker_env_ids = torch.arange(self.num_envs, device=self.device).repeat_interleave(self.num_segments)
        success_vis_offset = torch.tensor(self.success_vis_asset.init_state.pos, device=self.device)
        self._success_vis_positions_w = env.scene.env_origins + success_vis_offset

        from isaaclab.markers import VisualizationMarkers

        self.success_visualizer = VisualizationMarkers(self.cfg.success_visualizer_cfg)
        self.success_visualizer.set_visibility(True)
        self.success_visualizer.visualize(self._success_vis_positions_w, environment_ids=self._env.scene._ALL_INDICES)

    @property
    def command(self) -> torch.Tensor:
        """Flattened cable segment target positions [m], shape ``(num_envs, 3 * num_segments)``."""
        return self.target_positions_b.reshape(self.num_envs, -1)

    def _update_metrics(self) -> None:
        self._update_target_positions_w()
        current_positions_w = self.cable.data.segment_pose_w.torch[..., :3]
        position_error = torch.linalg.vector_norm(self.target_positions_w - current_positions_w, dim=-1)
        self.metrics["mean_position_error"][:] = position_error.mean(dim=1)
        self.metrics["max_position_error"][:] = position_error.max(dim=1).values
        self.success_visualizer.visualize(
            self._success_vis_positions_w,
            marker_indices=(self.metrics["max_position_error"] < self.cfg.success_threshold).int(),
            environment_ids=self._env.scene._ALL_INDICES,
        )

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        num_envs = len(env_ids)
        current_positions_b = self._current_cable_positions_b(env_ids)

        targets = torch.empty(num_envs, self.num_segments, 3, device=self.device)
        pending = torch.arange(num_envs, device=self.device)
        for _ in range(self.cfg.max_sampling_attempts):
            if len(pending) == 0:
                break
            difficulty = torch.rand(len(pending), device=self.device)
            candidates = self._sample_targets(current_positions_b[pending], difficulty)
            x_valid = (candidates[..., 0] >= self.cfg.target_xy_bounds[0][0]) & (
                candidates[..., 0] <= self.cfg.target_xy_bounds[0][1]
            )
            y_valid = (candidates[..., 1] >= self.cfg.target_xy_bounds[1][0]) & (
                candidates[..., 1] <= self.cfg.target_xy_bounds[1][1]
            )
            accepted = x_valid.all(dim=1) & y_valid.all(dim=1)
            targets[pending[accepted]] = candidates[accepted]
            pending = pending[~accepted]

        if len(pending) > 0:
            raise RuntimeError(f"Could not sample cable targets after {self.cfg.max_sampling_attempts} attempts.")

        self.target_positions_b[env_ids] = targets
        self.metrics["reset_distance"][env_ids] = torch.linalg.vector_norm(
            targets[..., :2] - current_positions_b[..., :2], dim=-1
        ).mean(dim=1)

    def _sample_targets(self, current_positions_b: torch.Tensor, difficulty: torch.Tensor) -> torch.Tensor:
        """Sample planar target-chain proposals."""
        num_envs = len(current_positions_b)

        random_first_xy = torch.empty(num_envs, 2, device=self.device)
        random_first_xy[:, 0].uniform_(*self.cfg.ranges.pos_x)
        random_first_xy[:, 1].uniform_(*self.cfg.ranges.pos_y)
        first_xy = torch.lerp(current_positions_b[:, 0, :2], random_first_xy, difficulty.unsqueeze(1))

        current_step = current_positions_b[:, 1, :2] - current_positions_b[:, 0, :2]
        current_heading = torch.atan2(current_step[:, 1], current_step[:, 0])
        random_heading = torch.empty(num_envs, device=self.device).uniform_(*self.cfg.ranges.heading)
        heading_delta = torch.atan2(
            torch.sin(random_heading - current_heading), torch.cos(random_heading - current_heading)
        )
        initial_heading = current_heading + difficulty * heading_delta
        turn_angles = torch.empty(num_envs, self.num_segments - 2, device=self.device).uniform_(
            -self.cfg.max_turn_angle, self.cfg.max_turn_angle
        )
        turn_angles *= difficulty.unsqueeze(1)
        heading_offsets = torch.cat((torch.zeros(num_envs, 1, device=self.device), turn_angles), dim=1).cumsum(dim=1)
        headings = initial_heading.unsqueeze(1) + heading_offsets
        steps = self.cfg.segment_length * torch.stack((torch.cos(headings), torch.sin(headings)), dim=-1)

        targets = torch.empty(num_envs, self.num_segments, 3, device=self.device)
        targets[..., 2] = self.cfg.target_z
        targets[:, 0, :2] = first_xy
        targets[:, 1:, :2] = first_xy.unsqueeze(1) + steps.cumsum(dim=1)
        return targets

    def _update_command(self) -> None:
        pass

    def _current_cable_positions_b(self, env_ids: Sequence[int]) -> torch.Tensor:
        """Return cable segment positions in the robot root frame [m]."""
        root_pos_w = self.robot.data.root_pos_w.torch[env_ids].unsqueeze(1)
        root_quat_w = self.robot.data.root_quat_w.torch[env_ids].unsqueeze(1)
        segment_pos_w = self.cable.data.segment_pose_w.torch[env_ids, :, :3]
        root_quat_w = root_quat_w.expand(-1, self.num_segments, -1)
        return quat_apply_inverse(root_quat_w, segment_pos_w - root_pos_w)

    def _update_target_positions_w(self) -> None:
        root_pos_w = self.robot.data.root_pos_w.torch.unsqueeze(1).expand(-1, self.num_segments, -1)
        root_quat_w = self.robot.data.root_quat_w.torch.unsqueeze(1).expand(-1, self.num_segments, -1)
        target_positions_w, _ = combine_frame_transforms(
            root_pos_w.reshape(-1, 3),
            root_quat_w.reshape(-1, 4),
            self.target_positions_b.reshape(-1, 3),
        )
        self.target_positions_w[:] = target_positions_w.view_as(self.target_positions_w)

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if debug_vis:
            if not hasattr(self, "target_visualizer"):
                from isaaclab.markers import VisualizationMarkers

                self.target_visualizer = VisualizationMarkers(self.cfg.target_visualizer_cfg)
                self.current_visualizer = VisualizationMarkers(self.cfg.current_visualizer_cfg)
            self.target_visualizer.set_visibility(True)
            self.current_visualizer.set_visibility(True)
        elif hasattr(self, "target_visualizer"):
            self.target_visualizer.set_visibility(False)
            self.current_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        if not self.robot.is_initialized or not self.cable.is_initialized:
            return
        self._update_target_positions_w()
        self.target_visualizer.visualize(self.target_positions_w.reshape(-1, 3), environment_ids=self._marker_env_ids)
        self.current_visualizer.visualize(
            self.cable.data.segment_pose_w.torch[..., :3].reshape(-1, 3), environment_ids=self._marker_env_ids
        )
