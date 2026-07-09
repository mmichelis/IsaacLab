# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Sysid-Droid-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.droid_sysid_env_cfg:DroidSysIdEnvCfg",
    },
)

gym.register(
    id="Isaac-Sysid-Droid-Corrected-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.droid_sysid_env_cfg:DroidCorrectedSysIdEnvCfg",
    },
)

gym.register(
    id="Isaac-Sysid-Droid-FR3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.droid_sysid_env_cfg:DroidFr3SysIdEnvCfg",
    },
)
