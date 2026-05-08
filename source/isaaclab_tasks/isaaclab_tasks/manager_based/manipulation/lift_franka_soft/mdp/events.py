# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms for the Franka deformable lifting environment."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import warp as wp

logger = logging.getLogger(__name__)

__all__ = ["disable_table_ground_rigid_collision"]


def disable_table_ground_rigid_collision(
    env: object,
    env_ids: Sequence[int] | None,
    label_substrings: tuple[str, ...] = ("ground", "defaultgroundplane", "table"),
) -> None:
    """Disable rigid robot collision against table and ground Newton shapes.

    The Newton soft-body Franka examples disable these rigid collisions to avoid table contact spikes from the
    PD-controlled robot while keeping particle contacts with the soft body active.

    Args:
        env: The manager-based environment.
        env_ids: Environment ids, unused for this global Newton model edit.
        label_substrings: Case-insensitive shape-label substrings to disable.
    """
    del env_ids

    if getattr(env, "_franka_soft_collision_disable_registered", False):
        return

    from isaaclab_newton.physics import NewtonManager

    from isaaclab.physics import PhysicsEvent

    normalized_substrings = tuple(label.lower() for label in label_substrings)

    def _disable(payload=None):
        del payload
        model = getattr(NewtonManager, "_model", None)
        if model is None:
            return

        labels = [str(label) for label in model.shape_label]
        groups = model.shape_collision_group.numpy()
        disabled: list[str] = []
        for index, label in enumerate(labels):
            if any(substring in label.lower() for substring in normalized_substrings):
                groups[index] = 0
                disabled.append(label)

        if disabled:
            model.shape_collision_group.assign(wp.array(groups, dtype=int, device=model.shape_collision_group.device))
            logger.info("Disabled rigid collision for Newton shapes: %s", disabled)

    env._franka_soft_collision_disable_registered = True
    env._franka_soft_collision_disable_handle = NewtonManager.register_callback(_disable, PhysicsEvent.PHYSICS_READY)
    _disable()
