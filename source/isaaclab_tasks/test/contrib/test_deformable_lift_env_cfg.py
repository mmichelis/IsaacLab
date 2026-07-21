# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for contributed deformable lift task registrations."""

import gymnasium as gym
import pytest

from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_TASKS = {
    "IsaacContrib-Lift-Soft-Franka": ("isaaclab_tasks.contrib.lift.config.franka.franka_soft_env_cfg:FrankaSoftEnvCfg"),
    "IsaacContrib-Lift-Cloth-Franka": (
        "isaaclab_tasks.contrib.lift.config.franka.franka_cloth_env_cfg:FrankaClothEnvCfg"
    ),
}
_REMOVED_TASK_IDS = [
    "Isaac-Lift-Soft-Franka",
    "Isaac-Lift-Cloth-Franka",
    "IsaacContrib-Lift-Soft-Franka-IK-Abs",
    "IsaacContrib-Lift-Cloth-Franka-IK-Abs",
]


def _agent_entry_points(task_id: str) -> set[str]:
    """Return registered agent configuration keys."""
    return {
        key for key in gym.spec(task_id).kwargs if key.endswith("_cfg_entry_point") and key != "env_cfg_entry_point"
    }


@pytest.mark.parametrize("task_id,entry_point", _TASKS.items())
def test_deformable_lift_tasks_use_absolute_ik_without_agent_config(task_id: str, entry_point: str):
    """Deformable lift tasks should expose only absolute IK control."""
    env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")

    assert gym.spec(task_id).kwargs["env_cfg_entry_point"] == entry_point
    assert isinstance(env_cfg.actions.arm_action, DifferentialInverseKinematicsActionCfg)
    assert env_cfg.actions.arm_action.controller.use_relative_mode is False
    assert env_cfg.actions.arm_action.controller.ik_method == "dls"
    assert _agent_entry_points(task_id) == set()


@pytest.mark.parametrize("task_id", _REMOVED_TASK_IDS)
def test_removed_deformable_lift_task_ids_are_not_registered(task_id: str):
    """Removed deformable lift task IDs should not be registered."""
    assert task_id not in gym.registry
