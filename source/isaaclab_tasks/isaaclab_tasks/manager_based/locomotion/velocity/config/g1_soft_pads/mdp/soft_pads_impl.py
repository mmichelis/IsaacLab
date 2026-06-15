# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation of the soft-pad joint-position action term.

Imported lazily (via the string ``class_type`` on
:class:`~...mdp.soft_pads.SoftPadJointPositionActionCfg`) only after the
simulation app has started, because it pulls in :class:`~isaaclab.assets.Articulation`.

Each pad is a free (dynamic) volume deformable simulated by VBD and coupled to
the MJWarp robot by Newton's two-way soft contact. To keep the pad registered
under its foot through the swing phase -- where pure contact would let it drop
away -- the pad's top layer of particles is re-pinned to the foot's pose every
sim-step via a kinematic target (Newton sets a pinned particle's inverse mass to
zero and snaps it to the target). The free lower particles deform against the
ground and push back on the foot through soft contact, mimicking a compliant
shoe sole.

The binding is captured lazily on first use from the pad's spawned (undeformed)
particle layout and the foot's rest pose, so it does not require the pad to be
spawned in exact alignment with the foot.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .soft_pads import SoftPadJointPositionActionCfg


class JointPositionActionWithSoftPads(JointPositionAction):
    """Joint-position action that also keeps a soft pad pinned under each foot.

    Reuses :class:`~isaaclab.envs.mdp.actions.joint_actions.JointPositionAction`
    for the robot's joints and, on every :meth:`apply_actions` (i.e. every
    sim-step), re-pins each pad's top face to its foot. On episode reset the
    whole pad is teleported back under the foot with zeroed velocity.
    """

    cfg: SoftPadJointPositionActionCfg

    def __init__(self, cfg: SoftPadJointPositionActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._robot = self._asset

        # Resolve (pad asset, foot body id) for each configured pair, preserving order.
        self._pads = []
        self._foot_ids: list[int] = []
        for pad_name, foot_body_name in cfg.pad_foot_pairs:
            self._pads.append(env.scene[pad_name])
            ids, _ = self._robot.find_bodies(foot_body_name, preserve_order=True)
            if len(ids) != 1:
                raise ValueError(
                    f"foot body name {foot_body_name!r} for pad {pad_name!r} matched {len(ids)} bodies;"
                    " expected exactly one."
                )
            self._foot_ids.append(ids[0])

        # Lazily-captured per-pad binding (filled by :meth:`_ensure_binding`).
        self._binding_ready = False
        self._rest_local: list[torch.Tensor] = []  # (num_envs, P, 3) particle offset in foot frame
        self._free_flag: list[torch.Tensor] = []  # (num_envs, P) 1.0 = free, 0.0 = pinned
        self._height_above_bottom: list[torch.Tensor] = []  # (num_envs, P) rest height above pad bottom
        # Per-env ground height (flat terrain -> env-origin z). Used to clamp pinned targets so the
        # kinematic constraint can never drag a pad below the floor.
        self._ground_z = env.scene.env_origins[:, 2:3].clone()  # (num_envs, 1)

    def _ensure_binding(self) -> None:
        """Capture each pad's rest layout in its foot's frame (idempotent)."""
        if self._binding_ready:
            return
        for pad in self._pads:
            pad_pos_w = pad.data.nodal_pos_w.torch  # (E, P, 3)

            # Box-local particle coordinates (pad spawns axis-aligned, so world axes == pad axes).
            centroid = pad_pos_w.mean(dim=1, keepdim=True)  # (E, 1, 3)
            pad_local = pad_pos_w - centroid  # (E, P, 3)

            # Express the pad layout directly in the FOOT frame (rest_local is applied at runtime as
            # ``foot_pos + R(foot_quat) @ rest_local``). The spawned box is axis-aligned, so its x is
            # the long (0.18 m) axis and z the thickness; mapping those onto the foot's local x
            # (heel->toe) and z (sole normal) makes the pad a proper sole that follows the foot's
            # heading. (Rotating into the world frame here — the previous behaviour — left the pad
            # world-axis-aligned and thus rotated relative to any foot not pointing along world +x.)
            z_offset = pad_local.new_tensor([0.0, 0.0, self.cfg.pad_z_offset]).view(1, 1, 3)
            rest_local = z_offset + pad_local  # (E, P, 3) in the foot frame
            self._rest_local.append(rest_local)

            # Pin the top ``pin_fraction`` of the pad thickness to the foot; leave the rest free.
            z = pad_local[..., 2]  # (E, P)
            z_min = z.amin(dim=1, keepdim=True)
            z_max = z.amax(dim=1, keepdim=True)
            pin_threshold = z_max - self.cfg.pin_fraction * (z_max - z_min)
            free = (z < pin_threshold).float()  # 1.0 = free, 0.0 = pinned (top layer)
            self._free_flag.append(free)

            # Rest height of each particle above the pad bottom (>= 0). When the foot drives the pad
            # toward/through the floor, pinned targets are clamped to (ground + this height) so the
            # pad rests flat on the ground instead of being dragged through it.
            self._height_above_bottom.append((z - z_min).clamp(min=0.0))  # (E, P)

        self._binding_ready = True

    def _pad_world_targets(self, pad_idx: int) -> torch.Tensor:
        """Current world positions the pad's particles should track, shape (E, P, 3)."""
        foot_id = self._foot_ids[pad_idx]
        rest_local = self._rest_local[pad_idx]  # (E, P, 3)
        num_envs, num_particles, _ = rest_local.shape
        fp = self._robot.data.body_link_pos_w.torch[:, foot_id, :].unsqueeze(1)  # (E, 1, 3)
        fq = self._robot.data.body_link_quat_w.torch[:, foot_id, :]  # (E, 4)
        fq = fq.unsqueeze(1).expand(-1, num_particles, -1)  # (E, P, 4)
        rotated = quat_apply(fq.reshape(-1, 4), rest_local.reshape(-1, 3)).reshape(num_envs, num_particles, 3)
        target = fp + rotated
        if self.cfg.clamp_to_ground:
            # Floor each pinned target at (ground + rest-height-above-pad-bottom) so the pad cannot be
            # forced below the ground; the foot then compresses into the pad instead of clipping through.
            floor_z = self._ground_z + self._height_above_bottom[pad_idx]  # (E, P)
            target[..., 2] = torch.maximum(target[..., 2], floor_z)
        return target

    def apply_actions(self) -> None:
        # Drive the robot joints first.
        super().apply_actions()

        self._ensure_binding()
        for pad_idx, pad in enumerate(self._pads):
            target_pos = self._pad_world_targets(pad_idx)  # (E, P, 3)
            targets = torch.cat([target_pos, self._free_flag[pad_idx].unsqueeze(-1)], dim=-1)  # (E, P, 4)
            pad.write_nodal_kinematic_target_to_sim_index(targets)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        self._ensure_binding()
        sel_ids = slice(None) if env_ids is None else env_ids
        for pad_idx, pad in enumerate(self._pads):
            target_pos = self._pad_world_targets(pad_idx)  # (E, P, 3)
            sel = target_pos[sel_ids].contiguous()
            ids = None if isinstance(sel_ids, slice) else env_ids
            # Teleport the whole pad back under the foot and zero its velocity.
            pad.write_nodal_pos_to_sim_index(sel, env_ids=ids)
            pad.write_nodal_velocity_to_sim_index(torch.zeros_like(sel), env_ids=ids)
