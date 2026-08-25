# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import TYPE_CHECKING

from isaaclab.managers import ActionTermCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .actions import CableForceAction


@configclass
class CableForceActionCfg(ActionTermCfg):
    """Configuration for independent cable segment forces."""

    class_type: type["CableForceAction"] | str = "{DIR}.actions:CableForceAction"

    scale: float = 0.1
    """Maximum force per action component [N]."""
