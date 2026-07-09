# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Sysid-Franka-FR3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fr3_sysid_env_cfg:FR3SysIdEnvCfg",
    },
)

# Same FR3 sysid, but the visible arm carries the static Robotiq 2F-85 gripper the
# real data was collected with (see franka_robotiq_sysid_env_cfg.py).
gym.register(
    id="Isaac-Sysid-Franka-Robotiq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_robotiq_sysid_env_cfg:FR3RobotiqSysIdEnvCfg",
    },
)
