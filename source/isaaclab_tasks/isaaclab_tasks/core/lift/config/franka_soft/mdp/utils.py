# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sanitized deformable state accessors shared by the deformable lift MDP terms.

The VBD solver can diverge and push individual particles to non-finite positions or velocities.
Since :attr:`~isaaclab.assets.DeformableObject.data.root_pos_w` is the mean of the nodal positions,
a single bad particle makes the whole center of mass non-finite, and every reward or observation
reading it returns ``NaN``. RL libraries check the returned rewards and observations, so a single
diverged environment aborts the whole run.

Terminating on the divergence is not enough on its own. In
:meth:`~isaaclab.envs.ManagerBasedRLEnv.step` the reward manager runs after the termination manager
but *before* the environments are reset, so rewards for the terminating step are still computed from
the diverged state. Observations are computed after the reset and are normally clean, but the
pre-reset paths (an active recorder term, or ``compute_final_obs``) also read the diverged state.

Reward and observation terms therefore read state through the helpers below, which replace
non-finite entries with ``0.0``. This places a diverged body at the world origin, yielding a finite
but meaningless value for exactly one step. That is intentional and acceptable, because
:func:`~isaaclab_tasks.core.lift.config.franka_soft.mdp.deformable_state_invalid` flags the same
step from the raw state and the environment is reset immediately.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.assets import DeformableObject


def _com_w(asset: DeformableObject) -> torch.Tensor:
    """Sanitized world-frame center of mass of a deformable object [m].

    Args:
        asset: The deformable object entity.

    Returns:
        Tensor of shape ``(num_envs, 3)`` with non-finite entries replaced by ``0.0``.
    """
    return torch.nan_to_num(asset.data.root_pos_w.torch, nan=0.0, posinf=0.0, neginf=0.0)


def _nodal_pos_w(asset: DeformableObject) -> torch.Tensor:
    """Sanitized world-frame nodal positions of a deformable object [m].

    Args:
        asset: The deformable object entity.

    Returns:
        Tensor of shape ``(num_envs, num_nodes, 3)`` with non-finite entries replaced by ``0.0``.
    """
    return torch.nan_to_num(asset.data.nodal_pos_w.torch, nan=0.0, posinf=0.0, neginf=0.0)
