# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone rigid "handle" box for the cable-reach task.

Originally this module generated a URDF cable articulation (handle + chain of
capsule links with revolute joints). That added a lot of physical complexity:
URDF→USD conversion quirks, solver divergence on the 20-DOF chain, and gripper
forces that had to compete with chain inertia during lift. For the learning task
we actually care about — grasp-and-move-to-target — the chain was never the
point; it was scene dressing. This simplified version spawns just the handle as
a :class:`~isaaclab.assets.RigidObjectCfg` so the policy has a clean rigid
object to manipulate. The cable can be re-added once grasp + lift + target
tracking is learned reliably on the bare handle.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg


def build_cable_articulation_cfg(
    prim_path: str = "{ENV_REGEX_NS}/Cable",
    init_pos: tuple[float, float, float] = (0.5, 0.0, 0.05),
    # IsaacLab quaternion convention is (x, y, z, w). (0, 0, 0, 1) is identity.
    init_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    # Handle dimensions. 8 × 6 × 6 cm fits comfortably in the Panda's 8 cm gripper
    # opening (along the 6 cm axis) and is large enough to be easy to target
    # visually during debugging.
    handle_size: tuple[float, float, float] = (0.08, 0.06, 0.06),
    # 120 g: heavy enough to rest stably on the table and resist being bumped
    # away during approach, light enough that the Panda's stock grip force
    # (~40 N with 1 cm gap) lifts it with plenty of margin.
    handle_mass: float = 0.12,
    # High friction on the handle surface helps the Panda fingertips maintain
    # grip during lift accelerations. ``friction_combine_mode="max"`` ensures
    # our high value isn't averaged down against the default ~0.5 of the Panda
    # fingers.
    static_friction: float = 1.2,
    dynamic_friction: float = 1.0,
) -> RigidObjectCfg:
    """Build a :class:`RigidObjectCfg` for a simple rigid box "handle".

    The returned config has the same name as the old URDF-articulation builder
    (``build_cable_articulation_cfg``) to minimise churn in the env cfg, even
    though "articulation" is now a misnomer — it's a single rigid body.
    """
    spawn = sim_utils.CuboidCfg(
        size=handle_size,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            max_linear_velocity=100.0,
            max_angular_velocity=500.0,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=handle_mass),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 0.9)),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
            friction_combine_mode="max",
            restitution=0.0,
        ),
    )

    init_state = RigidObjectCfg.InitialStateCfg(pos=init_pos, rot=init_rot)
    return RigidObjectCfg(prim_path=prim_path, spawn=spawn, init_state=init_state)
