# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reward_weight_linear(
    env: ManagerBasedRLEnv,
    _env_ids: Sequence[int],
    term_name: str,
    start_weight: float,
    end_weight: float,
    start_step: int,
    end_step: int,
) -> float:
    """Linearly ramp a reward weight between two environment steps."""
    if end_step <= start_step:
        raise ValueError("end_step must be greater than start_step.")

    alpha = (env.common_step_counter - start_step) / (end_step - start_step)
    alpha = min(max(alpha, 0.0), 1.0)
    weight = start_weight + alpha * (end_weight - start_weight)
    reward_cfg = env.reward_manager.get_term_cfg(term_name)
    reward_cfg.weight = weight
    env.reward_manager.set_term_cfg(term_name, reward_cfg)
    return weight


def linear_interpolate(
    env: ManagerBasedRLEnv,
    _env_ids: Sequence[int],
    _value: float,
    start_value: float,
    end_value: float,
    start_step: int,
    end_step: int,
) -> float:
    """Linearly interpolate a scalar value over environment steps."""
    if end_step <= start_step:
        raise ValueError("end_step must be greater than start_step.")

    alpha = (env.common_step_counter - start_step) / (end_step - start_step)
    alpha = min(max(alpha, 0.0), 1.0)
    return start_value + alpha * (end_value - start_value)
