# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class FourBarJointEffortAction(ActionTerm):
    """Apply joint efforts [N·m] to selected four-bar joints."""

    cfg: "FourBarJointEffortActionCfg"

    def __init__(self, cfg: "FourBarJointEffortActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._joint_ids = torch.tensor(cfg.joint_ids, dtype=torch.long, device=self.device)
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._gait_actions = torch.zeros_like(self._raw_actions)

    @property
    def action_dim(self) -> int:
        """Dimension of the action term."""
        return len(self.cfg.joint_ids)

    @property
    def raw_actions(self) -> torch.Tensor:
        """Raw actions passed to this term."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Scaled joint efforts [N·m] applied to the selected joints."""
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        """Scale the input actions into joint efforts [N·m].

        Args:
            actions: Input actions, shape ``(num_envs, action_dim)``.
        """
        self._raw_actions[:] = actions
        clipped_actions = actions
        if self.cfg.action_clip is not None:
            clipped_actions = torch.clamp(actions, min=self.cfg.action_clip[0], max=self.cfg.action_clip[1])
        self._processed_actions[:] = clipped_actions * self.cfg.scale
        if self.cfg.gait_effort != 0.0:
            elapsed_time = self._env.episode_length_buf.to(self.device, dtype=torch.float32) * self._env.step_dt
            phase = 2.0 * torch.pi * elapsed_time / self.cfg.gait_period_s
            gait_scale = 1.0
            if self.cfg.gait_ramp_s > 0.0:
                gait_scale = torch.clamp(elapsed_time / self.cfg.gait_ramp_s, min=0.0, max=1.0).unsqueeze(-1)
            self._gait_actions[:, 0] = torch.sin(phase)
            self._gait_actions[:, 1] = torch.cos(phase)
            self._processed_actions += self.cfg.gait_effort * gait_scale * self._gait_actions
        if self.cfg.effort_clip is not None:
            self._processed_actions.clamp_(min=self.cfg.effort_clip[0], max=self.cfg.effort_clip[1])

    def apply_actions(self) -> None:
        """Apply the processed joint efforts [N·m] to Newton's control buffer."""
        from isaaclab_newton.physics import NewtonManager  # noqa: PLC0415
        import warp as wp  # noqa: PLC0415

        control = NewtonManager.get_control()
        if control is None or control.joint_f is None:
            return

        joint_efforts = wp.to_torch(control.joint_f).view(self.num_envs, -1)
        joint_efforts.zero_()
        joint_efforts[:, self._joint_ids] = self._processed_actions


@configclass
class FourBarJointEffortActionCfg(ActionTermCfg):
    """Configuration for four-bar joint effort actions."""

    class_type: type[FourBarJointEffortAction] | str = "{DIR}.actions:FourBarJointEffortAction"

    joint_ids: tuple[int, ...] = (0, 2)
    """Indices of joints receiving effort actions."""

    scale: float = 1.0
    """Action scale applied to produce joint efforts [N·m]."""

    gait_effort: float = 0.0
    """Nominal phase gait effort added to the selected joints [N·m]."""

    gait_period_s: float = 1.0
    """Period of the nominal phase gait [s]."""

    gait_ramp_s: float = 0.0
    """Time to ramp in the nominal phase gait effort after reset [s]."""

    action_clip: tuple[float, float] | None = None
    """Clip range for raw policy actions before scaling."""

    effort_clip: tuple[float, float] | None = None
    """Clip range for processed joint efforts [N·m]."""
