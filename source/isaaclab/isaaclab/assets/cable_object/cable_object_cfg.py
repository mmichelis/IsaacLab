# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the cable asset class."""

from __future__ import annotations

from isaaclab.assets.asset_base_cfg import AssetBaseCfg
from isaaclab.utils.configclass import configclass


@configclass
class CableObjectCfg(AssetBaseCfg):
    """Configuration for a cable / 1D-rod asset (Newton backend).

    The cable composes Newton's :class:`~newton.selection.ArticulationView` as a
    backend primitive — like :class:`~isaaclab.assets.RigidObject` does for its
    single-body articulations — but :class:`~isaaclab_contrib.cable.CableObject`
    is a peer of :class:`~isaaclab.assets.RigidObject` /
    :class:`~isaaclab.assets.Articulation` /
    :class:`~isaaclab.assets.DeformableObject` under
    :class:`~isaaclab.assets.AssetBase`, not an Articulation subclass.
    """

    @configclass
    class InitialStateCfg(AssetBaseCfg.InitialStateCfg):
        """Initial pose/velocity of the cable.

        Pose-only (matches :class:`~isaaclab.assets.RigidObjectCfg.InitialStateCfg`):
        the cable has no user-defined joints, so there is no joint-state field.
        :attr:`pos` and :attr:`rot` are inherited from
        :class:`AssetBaseCfg.InitialStateCfg`.
        """

        lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Initial linear velocity of the cable's root body [m/s]. Unused by VBD;
        kept for API symmetry with :class:`~isaaclab.assets.RigidObjectCfg`."""

        ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Initial angular velocity of the cable's root body [rad/s]. Unused by VBD;
        kept for API symmetry with :class:`~isaaclab.assets.RigidObjectCfg`."""

    class_type: type | str = "isaaclab_contrib.cable.cable_object:CableObject"

    init_state: InitialStateCfg = InitialStateCfg()
    """Initial pose/velocity for the cable. See :class:`InitialStateCfg`."""
