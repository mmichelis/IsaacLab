# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from isaaclab_tasks.core.peg_in_hole.config.franka import agents

##
# Register Gym environments.
##

##
# Franka
##

gym.register(
    id="Isaac-Peg-In-Hole-Franka",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_env_cfg:FrankaPegInHoleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PegInHolePPORunnerCfg",
    },
)
