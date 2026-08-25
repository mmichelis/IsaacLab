# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.lift.config.franka_soft.agents.rsl_rl_ppo_cfg import FrankaCableShapePPORunnerCfg


@configclass
class CableShapeForcePPORunnerCfg(FrankaCableShapePPORunnerCfg):
    """PPO configuration for direct cable force control."""

    experiment_name = "shape_cable_direct_force"
    max_iterations = 500
