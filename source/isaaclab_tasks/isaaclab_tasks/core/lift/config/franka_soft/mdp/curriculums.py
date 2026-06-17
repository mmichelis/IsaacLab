# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum terms specific to the cable-plug task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.envs import mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def step_widen_pose_range(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    data: dict[str, tuple[float, float]],
    initial_range: dict[str, tuple[float, float]],
    final_range: dict[str, tuple[float, float]],
    num_steps: int,
    start_step: int = 0,
) -> dict[str, tuple[float, float]] | object:
    """Linearly widen a spherical reset-pose range from initial to final over training steps.

    Ranges map a key (``r`` [m], ``theta``/``phi``/``yaw``/``pitch`` [rad]) to its ``(lo, hi)`` bound.
    Each key's tuple is interpolated from :paramref:`initial_range` to
    :paramref:`final_range` as :attr:`~isaaclab.envs.ManagerBasedRLEnv.common_step_counter` ramps
    from ``start_step`` to ``start_step + num_steps``. Keys absent from :paramref:`final_range`
    keep their initial value. Designed for use with :class:`~isaaclab.envs.mdp.modify_term_cfg`
    targeting an event term's ``pose_range`` parameter.

    Args:
        data: Current value of the targeted range (unused; structure comes from the ranges).
        initial_range: Tight range applied before ``start_step``.
        final_range: Widened range reached at ``start_step + num_steps``.
        num_steps: Number of steps over which to ramp.
        start_step: Step at which widening begins.

    Returns:
        The interpolated range dict, or :attr:`modify_env_param.NO_CHANGE` before ``start_step``.
    """
    frac = (env.common_step_counter - start_step) / max(num_steps, 1)
    if frac <= 0.0:
        return mdp.modify_env_param.NO_CHANGE
    frac = min(frac, 1.0)
    widened = dict(initial_range)
    for key, (f_lo, f_hi) in final_range.items():
        i_lo, i_hi = initial_range[key]
        widened[key] = (i_lo + frac * (f_lo - i_lo), i_hi + frac * (f_hi - i_hi))
    return widened
