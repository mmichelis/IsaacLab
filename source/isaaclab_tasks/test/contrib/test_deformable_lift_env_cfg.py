# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for contributed deformable lift task registrations."""

import importlib

import gymnasium as gym
import pytest

from isaaclab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
    JointPositionActionCfg,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_TASK_IDS = [
    "IsaacContrib-Lift-Soft-Franka-IK-Abs",
    "IsaacContrib-Lift-Cloth-Franka-IK-Abs",
]

_TASK_ALIASES = [
    ("Isaac-Lift-Soft-Franka", "IsaacContrib-Lift-Soft-Franka-IK-Abs", DifferentialInverseKinematicsActionCfg),
    ("Isaac-Lift-Cloth-Franka", "IsaacContrib-Lift-Cloth-Franka-IK-Abs", JointPositionActionCfg),
]

_RSL_RL_CFG_ENTRY_POINT = (
    "isaaclab_tasks.core.lift.config.franka_soft.agents.rsl_rl_ppo_cfg:FrankaDeformablePPORunnerCfg"
)
_ENV_CFG_ENTRY_POINTS = {
    "IsaacContrib-Lift-Soft-Franka-IK-Abs": (
        "isaaclab_tasks.contrib.lift.config.franka.franka_soft_env_cfg:FrankaSoftEnvCfg"
    ),
    "IsaacContrib-Lift-Cloth-Franka-IK-Abs": (
        "isaaclab_tasks.contrib.lift.config.franka.franka_cloth_env_cfg:FrankaClothEnvCfg"
    ),
    "Isaac-Lift-Soft-Franka": "isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg:FrankaSoftEnvCfg",
    "Isaac-Lift-Cloth-Franka": "isaaclab_tasks.core.lift.config.franka_soft.franka_cloth_env_cfg:FrankaClothEnvCfg",
}


def _agent_entry_points(task_id: str) -> set[str]:
    """Return registered agent configuration keys."""
    return {
        key for key in gym.spec(task_id).kwargs if key.endswith("_cfg_entry_point") and key != "env_cfg_entry_point"
    }


@pytest.mark.parametrize("task_id", _TASK_IDS)
def test_deformable_lift_tasks_use_absolute_ik_without_agent_config(task_id: str):
    """Deformable lift tasks should expose only absolute IK control."""
    env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
    assert gym.spec(task_id).kwargs["env_cfg_entry_point"] == _ENV_CFG_ENTRY_POINTS[task_id]

    assert isinstance(env_cfg.actions.arm_action, DifferentialInverseKinematicsActionCfg)
    assert env_cfg.actions.arm_action.controller.use_relative_mode is False
    assert env_cfg.actions.arm_action.controller.ik_method == "dls"
    assert _agent_entry_points(task_id) == set()


@pytest.mark.parametrize("alias_task_id,task_id,action_type", _TASK_ALIASES)
def test_deformable_lift_aliases_preserve_released_contract(alias_task_id: str, task_id: str, action_type: type):
    """Legacy IDs should preserve their released actions and agent config."""
    alias_spec = gym.spec(alias_task_id)
    assert alias_spec.kwargs["env_cfg_entry_point"] == _ENV_CFG_ENTRY_POINTS[alias_task_id]
    with pytest.warns(FutureWarning, match=f"Task '{alias_task_id}' is deprecated"):
        env_cfg = load_cfg_from_registry(alias_task_id, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(alias_task_id, "rsl_rl_cfg_entry_point")

    assert isinstance(env_cfg.actions.arm_action, action_type)
    assert alias_spec.kwargs["deprecated"] == {"alias": f"--task {task_id}"}
    assert _agent_entry_points(alias_task_id) == {"rsl_rl_cfg_entry_point"}
    assert alias_spec.kwargs["rsl_rl_cfg_entry_point"] == _RSL_RL_CFG_ENTRY_POINT
    assert agent_cfg.__class__.__name__ == "FrankaDeformablePPORunnerCfg"


def test_deformable_lift_import_paths_preserve_dependency_direction():
    """Contrib configs should extend the released core implementations."""
    core_soft = importlib.import_module("isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg")
    core_cloth = importlib.import_module("isaaclab_tasks.core.lift.config.franka_soft.franka_cloth_env_cfg")
    core_agent = importlib.import_module("isaaclab_tasks.core.lift.config.franka_soft.agents.rsl_rl_ppo_cfg")
    contrib_soft = importlib.import_module("isaaclab_tasks.contrib.lift.config.franka.franka_soft_env_cfg")
    contrib_cloth = importlib.import_module("isaaclab_tasks.contrib.lift.config.franka.franka_cloth_env_cfg")

    assert core_soft.FrankaSoftEnvCfg.__module__ == core_soft.__name__
    assert core_cloth.FrankaClothEnvCfg.__module__ == core_cloth.__name__
    assert contrib_soft.FrankaSoftEnvCfg is core_soft.FrankaSoftEnvCfg
    assert issubclass(contrib_cloth.FrankaClothEnvCfg, core_cloth.FrankaClothEnvCfg)
    assert core_agent.FrankaDeformablePPORunnerCfg.__name__ == "FrankaDeformablePPORunnerCfg"
