# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

__all__ = [
    # observations
    "object_position_in_robot_root_frame",
    "deformable_com_in_robot_root_frame",
    "DeformableSampledPointsInRobotRootFrame",
    # rewards
    "object_ee_distance",
    "object_goal_distance",
    "object_goal_distance_delta",
    "object_is_lifted",
    "deformable_lifted",
    "deformable_ee_distance",
    "deformable_com_goal_distance",
    "gripper_close_action",
    # terminations
    "object_reached_goal",
    "object_outside_table_bounds",
    "deformable_com_below_minimum",
    "deformable_outside_table_bounds",
    "ee_below_minimum",
    # curriculums
    "modify_gravity_linear",
]

from .curriculums import modify_gravity_linear

from .observations import (
    DeformableSampledPointsInRobotRootFrame,
    deformable_com_in_robot_root_frame,
    object_position_in_robot_root_frame,
)
from .rewards import (
    deformable_com_goal_distance,
    deformable_ee_distance,
    deformable_lifted,
    gripper_close_action,
    object_ee_distance,
    object_goal_distance,
    object_goal_distance_delta,
    object_is_lifted,
)
from .terminations import (
    deformable_com_below_minimum,
    deformable_outside_table_bounds,
    ee_below_minimum,
    object_outside_table_bounds,
    object_reached_goal,
)
from isaaclab.envs.mdp import *
