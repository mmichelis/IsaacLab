# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom command terms for the cable-reach task.

Split off from :mod:`mdp` so the runtime :class:`UniformPoseCommand` import chain —
which eagerly pulls in ``isaaclab.markers.VisualizationMarkers`` and through it the
pxr / omni USD stack — does NOT fire at env-cfg-resolution time. Hydra resolves the
cfg before :class:`AppLauncher` starts Kit, and loading pxr before Kit boots leaves
it half-initialized and crashes the Kit extension startup ("pxr.PhysxSchema has no
attribute 'Tokens'" and friends). The cfg subclass in :mod:`mdp` references this
module through a string ``class_type`` so the runtime import is deferred until the
command manager actually instantiates the term, by which point Kit is up.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.commands import UniformPoseCommand
from isaaclab.utils.math import quat_mul


class GripperAlignedPoseCommand(UniformPoseCommand):
    """UniformPoseCommand with a fixed gripper-aligned reference baked into the sampled quat.

    The Panda hand's body frame has its x-axis "up" (red marker) and z-axis "forward"
    (blue marker) in the natural top-down approach pose. The vanilla
    :class:`UniformPoseCommand` samples around base-frame identity (z-up, x-forward),
    which visualizes the target as "palm pointing straight up" — a pose far from any
    reachable gripper configuration.

    This subclass lets the parent sample a small euler delta in the usual way, then
    LEFT-multiplies the sampled quaternion with a fixed reference quat so the stored
    command is ``R_ref * delta`` (both the reward math and the debug visualizer read
    this value, so they stay in sync). The reference rotation is 180° about the
    base-frame ``(1, 0, 1) / √2`` diagonal axis, which maps:

    * base +Z → base +X (target blue-axis points forward — gripper z)
    * base +X → base +Z (target red-axis points up — gripper x)
    * base +Y → base -Y

    Small euler perturbations on (roll, pitch, yaw) remain small rotations around
    this reference because euler-to-quat stays well-conditioned near identity — we
    avoid the gimbal-lock trap that would hit us if we instead tried to center the
    euler ranges themselves on ``pitch = ±π/2``.
    """

    # (x, y, z, w) — 180° rotation about (1, 0, 1) / √2.
    _REFERENCE_QUAT_B: tuple[float, float, float, float] = (
        0.7071067811865476,
        0.0,
        0.7071067811865476,
        0.0,
    )

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        delta = self.pose_command_b[env_ids, 3:]
        ref = torch.tensor(
            self._REFERENCE_QUAT_B, device=self.device, dtype=delta.dtype
        ).expand_as(delta)
        self.pose_command_b[env_ids, 3:] = quat_mul(ref, delta)
