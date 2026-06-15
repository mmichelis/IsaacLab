# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Config for the soft-pad joint-position action term.

This module is imported at config-resolution time (before the simulation app
launches), so it must stay import-light. The implementation lives in
:mod:`soft_pads_impl` and is referenced via a string ``class_type`` that is
resolved lazily once the app is up -- importing the action implementation here
would pull in :class:`~isaaclab.assets.Articulation` and its Kit C-extensions
before ``SimulationApp`` starts, which corrupts the plugin loader and crashes
app startup.
"""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils.configclass import configclass

# Absolute "module:attr" path to the lazily-imported implementation.
_IMPL = (
    "isaaclab_tasks.manager_based.locomotion.velocity.config.g1_soft_pads.mdp."
    "soft_pads_impl:JointPositionActionWithSoftPads"
)


@configclass
class SoftPadJointPositionActionCfg(JointPositionActionCfg):
    """Joint-position action that also pins a soft pad under each foot.

    See :class:`~...mdp.soft_pads_impl.JointPositionActionWithSoftPads`.
    """

    class_type: type | str = _IMPL

    pad_foot_pairs: list[tuple[str, str]] = MISSING
    """Ordered ``(pad_scene_asset_name, foot_body_name)`` pairs to glue together."""

    pad_z_offset: float = -0.03
    """Rest offset [m] of the pad center below the foot-body origin (negative = down)."""

    pin_fraction: float = 0.25
    """Top fraction of the pad thickness pinned (kinematically) to the foot; the rest deforms freely."""

    clamp_to_ground: bool = True
    """Floor the pinned kinematic targets at the ground so the pad can never be dragged below it."""
