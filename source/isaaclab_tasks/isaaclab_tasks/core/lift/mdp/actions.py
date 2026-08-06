# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs.mdp.actions import JointPositionToLimitsAction


class LeaderOnlyJointPositionToLimitsAction(JointPositionToLimitsAction):
    """Apply only the leader command while retaining the follower action slot."""

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target_index(target=self.processed_actions[:, :1], joint_ids=self._joint_ids[:1])
