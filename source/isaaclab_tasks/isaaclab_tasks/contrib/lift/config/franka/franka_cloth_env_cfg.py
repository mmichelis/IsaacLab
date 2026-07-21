# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contributed Franka cloth lifting configuration."""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.lift.config.franka_soft.franka_cloth_env_cfg import (
    FrankaClothEnvCfg as _CoreFrankaClothEnvCfg,
)
from isaaclab_tasks.core.lift.config.franka_soft.franka_cloth_env_cfg import (
    FrankaClothSceneCfg as FrankaClothSceneCfg,
)
from isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg import ActionsCfg as _ActionsCfg


@configclass
class FrankaClothEnvCfg(_CoreFrankaClothEnvCfg):
    """Franka cloth lifting environment with absolute IK."""

    actions: _ActionsCfg = _ActionsCfg()
