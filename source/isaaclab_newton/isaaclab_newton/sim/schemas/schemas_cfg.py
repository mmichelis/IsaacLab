# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.sim.schemas.schemas_cfg import DeformableBodyPropertiesBaseCfg
from isaaclab.utils import configclass


@configclass
class NewtonDeformableBodyPropertiesCfg(DeformableBodyPropertiesBaseCfg):
    """Newton-specific properties to apply to a deformable body.

    Currently empty. Backend-specific fields can be added here when Newton exposes
    a registered deformable body property schema.
    """
