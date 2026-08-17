# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.lift.mdp.events_cfg import GraspTravelDistanceCfg


@configclass
class GraspTravelOpeningCfg(GraspTravelDistanceCfg):
    """Spread reset states over grasp distance, travel distance, and gripper opening.

    Distance features follow :attr:`log_scale`; gripper opening remains linear.
    """

    func: str = "{DIR}.events:grasp_travel_opening"

    gripper_joint_names: str | list[str] = MISSING
    """Gripper joints whose absolute positions form the opening [m or rad, depending on joint type]."""
