# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for contributed deformable lift task registrations."""

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


def _agent_entry_points(task_id: str) -> set[str]:
    """Return registered agent configuration keys."""
    return {
        key for key in gym.spec(task_id).kwargs if key.endswith("_cfg_entry_point") and key != "env_cfg_entry_point"
    }


@pytest.mark.parametrize("task_id", _TASK_IDS)
def test_deformable_lift_tasks_use_absolute_ik_without_agent_config(task_id: str):
    """Deformable lift tasks should expose only absolute IK control."""
    env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")

    assert isinstance(env_cfg.actions.arm_action, DifferentialInverseKinematicsActionCfg)
    assert env_cfg.actions.arm_action.controller.use_relative_mode is False
    assert _agent_entry_points(task_id) == set()


@pytest.mark.parametrize("alias_task_id,task_id,action_type", _TASK_ALIASES)
def test_deformable_lift_aliases_are_deprecated_without_agent_config(
    alias_task_id: str, task_id: str, action_type: type
):
    """Legacy IDs should remain agent-free while preserving their action type."""
    alias_spec = gym.spec(alias_task_id)
    env_cfg = load_cfg_from_registry(alias_task_id, "env_cfg_entry_point")

    assert isinstance(env_cfg.actions.arm_action, action_type)
    assert alias_spec.kwargs["deprecated"] == {"alias": f"--task {task_id}"}
    assert _agent_entry_points(alias_task_id) == set()
