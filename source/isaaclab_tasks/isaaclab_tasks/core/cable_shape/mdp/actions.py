# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp
from isaaclab_newton.assets import CableObject
from isaaclab_newton.physics import NewtonManager

from isaaclab.managers import ActionTerm

if TYPE_CHECKING:
    from newton import State

    from isaaclab.envs import ManagerBasedEnv

    from .actions_cfg import CableForceActionCfg


@wp.kernel(enable_backward=False)
def _apply_segment_forces(
    segment_force_w: wp.array2d(dtype=wp.vec3f),
    root_body_ids: wp.array(dtype=wp.int32),
    link_body_ids: wp.array2d(dtype=wp.int32),
    body_f: wp.array(dtype=wp.spatial_vectorf),
):
    env_id, segment_id = wp.tid()
    body_id = root_body_ids[env_id]
    if segment_id > 0:
        body_id = link_body_ids[env_id, segment_id - 1]
    force_w = segment_force_w[env_id, segment_id]
    body_f[body_id] += wp.spatial_vector(force_w, wp.vec3f(0.0, 0.0, 0.0), wp.float32)


class ForceControlledCableObject(CableObject):
    """Newton cable with a graph-safe world-frame segment force buffer."""

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if not hasattr(self, "_segment_force_w_torch"):
            return
        if env_ids is None:
            self._segment_force_w_torch.zero_()
        else:
            self._segment_force_w_torch[env_ids] = 0.0

    def set_segment_forces(self, forces: torch.Tensor) -> None:
        """Set world-frame segment forces [N]."""
        if forces.shape != self._segment_force_w_torch.shape:
            raise ValueError(
                f"Expected segment forces with shape {tuple(self._segment_force_w_torch.shape)},"
                f" received {tuple(forces.shape)}."
            )
        self._segment_force_w_torch.copy_(forces)

    def _initialize_impl(self) -> None:
        super()._initialize_impl()
        self._segment_force_w = wp.zeros(
            (self.num_instances, self.num_segments), dtype=wp.vec3f, device=self.device
        )
        self._segment_force_w_torch = wp.to_torch(self._segment_force_w)
        NewtonManager.register_state_force_callback(self._apply_segment_forces)

    def _apply_segment_forces(self, state: State) -> None:
        wp.launch(
            _apply_segment_forces,
            dim=(self.num_instances, self.num_segments),
            inputs=[
                self._segment_force_w,
                self.data._sim_bind_root_body_ids,
                self.data._sim_bind_link_body_ids,
            ],
            outputs=[state.body_f],
            device=self.device,
        )


class CableForceAction(ActionTerm):
    """Apply independent world-frame forces at every cable segment center of mass."""

    cfg: CableForceActionCfg
    _asset: ForceControlledCableObject

    def __init__(self, cfg: CableForceActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if not isinstance(self._asset, ForceControlledCableObject):
            raise TypeError(f"CableForceAction requires ForceControlledCableObject, received {type(self._asset)}.")
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

    @property
    def action_dim(self) -> int:
        return 3 * self._asset.num_segments

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        self._processed_actions[:] = actions.clamp(-1.0, 1.0) * self.cfg.scale

    def apply_actions(self) -> None:
        self._asset.set_segment_forces(self._processed_actions.view(self.num_envs, self._asset.num_segments, 3))

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._asset.reset(env_ids)
