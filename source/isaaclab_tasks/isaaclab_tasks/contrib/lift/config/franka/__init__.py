# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym

from isaaclab_tasks.core.lift.config.franka import agents

##
# Register Gym environments.
##

##
# Inverse Kinematics - Absolute Pose Control
##

gym.register(
    id="IsaacContrib-Lift-Cube-Franka-IK-Abs",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:FrankaCubeLiftEnvCfg",
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="IsaacContrib-Lift-Cube-Franka-IK-Rel",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_env_cfg:FrankaCubeLiftEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc.json",
    },
    disable_env_checker=True,
)

##
# Deformable Objects - Absolute Pose Control
##

gym.register(
    id="IsaacContrib-Lift-Soft-Franka",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_soft_env_cfg:FrankaSoftEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="IsaacContrib-Lift-Cloth-Franka",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_cloth_env_cfg:FrankaClothEnvCfg",
    },
    disable_env_checker=True,
)


##
# Camera environments
##

gym.register(
    id="IsaacContrib-Lift-Soft-Franka-Camera",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_soft_env_cfg:FrankaSoftCameraEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="IsaacContrib-Lift-Cloth-Franka-Camera",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_cloth_env_cfg:FrankaClothCameraEnvCfg",
    },
    disable_env_checker=True,
)
