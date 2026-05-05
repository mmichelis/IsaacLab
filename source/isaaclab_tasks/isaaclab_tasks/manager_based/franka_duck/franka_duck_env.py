# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based RL env for the Franka deformable-duck task.

This thin subclass keeps a task-specific entry point for deformable Newton
setup. It is imported by ``gym.make("Isaac-Franka-Duck-v0")``, which happens
after :class:`SimulationApp` has started, so deferred Newton imports do not
conflict with USD schema initialisation.
"""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnv


class FrankaDuckEnv(ManagerBasedRLEnv):
    """:class:`ManagerBasedRLEnv` with deformable Newton hooks pre-registered."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
