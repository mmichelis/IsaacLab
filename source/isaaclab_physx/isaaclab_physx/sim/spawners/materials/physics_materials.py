# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility wrappers for deformable physics material spawning.

The deformable material writer now lives in :mod:`isaaclab.sim.spawners.materials`.
"""

from isaaclab.sim.spawners.materials.physics_materials import spawn_deformable_body_material

__all__ = ["spawn_deformable_body_material"]
