# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Procedural cable articulation for the cable-reach task.

The cable is built as an articulated chain of rigid cylinder links with a slightly
larger cuboid "handle" as the root link for robust two-finger grasping. A ``<freejoint/>``
/ floating-base semantics is provided by the URDF importer with ``fix_base=False``.

Each consecutive pair of links is connected by a single revolute joint with an axis
that alternates between y (pitch) and z (yaw). This is a 1-DOF-per-connection
approximation of a spherical joint that omits twist around the cable axis — which is
physically insignificant for a grasped cable — while preserving bending behavior. A
URDF is generated from the config parameters, written to a temp file, and loaded via
:class:`~isaaclab.sim.spawners.UrdfFileCfg` (which lazily converts URDF to USD on
first use).
"""

from __future__ import annotations

import hashlib
import os
import tempfile

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


def _cuboid_inertia(mass: float, size: tuple[float, float, float]) -> tuple[float, float, float]:
    """Principal-axis inertia for a uniform-density cuboid."""
    lx, ly, lz = size
    ixx = mass * (ly * ly + lz * lz) / 12.0
    iyy = mass * (lx * lx + lz * lz) / 12.0
    izz = mass * (lx * lx + ly * ly) / 12.0
    return ixx, iyy, izz


def _cylinder_inertia(mass: float, radius: float, length: float) -> tuple[float, float, float]:
    """Principal-axis inertia for a cylinder along its local z-axis."""
    # In URDF cylinder convention the cylinder's long axis is z. In the cable we rotate
    # each link so the cylinder lies along x, but inertia tensors are expressed in the
    # (pre-rotation) local frame stored under ``<inertial>``. We report inertia for the
    # canonical orientation (long axis along local x, i.e., post-rotation frame).
    # For a cylinder along x, the moments about y and z are equal.
    ixx = 0.5 * mass * radius * radius  # about x (long axis)
    iyy = mass * (3.0 * radius * radius + length * length) / 12.0
    izz = iyy
    return ixx, iyy, izz


def _cable_urdf(
    num_links: int,
    link_length: float,
    link_radius: float,
    handle_size: tuple[float, float, float],
    handle_mass: float,
    link_mass: float,
    joint_damping: float,
    joint_limit_rad: float,
    joint_effort: float,
    joint_velocity_limit: float,
) -> str:
    """Build URDF XML for the cable articulation."""
    hx, hy, hz = (s / 2.0 for s in handle_size)
    h_ixx, h_iyy, h_izz = _cuboid_inertia(handle_mass, handle_size)
    l_ixx, l_iyy, l_izz = _cylinder_inertia(link_mass, link_radius, link_length)

    # Rotate cylinder (default z-axis) to align with local x.
    cyl_rpy = (0.0, 1.5707963267948966, 0.0)

    first_offset_x = hx + link_length / 2.0
    link_step_x = link_length

    out: list[str] = [
        "<?xml version=\"1.0\"?>",
        "<robot name=\"cable\">",
        "  <link name=\"handle\">",
        "    <inertial>",
        "      <origin xyz=\"0 0 0\"/>",
        f"      <mass value=\"{handle_mass}\"/>",
        f"      <inertia ixx=\"{h_ixx}\" iyy=\"{h_iyy}\" izz=\"{h_izz}\" ixy=\"0\" ixz=\"0\" iyz=\"0\"/>",
        "    </inertial>",
        "    <visual>",
        "      <origin xyz=\"0 0 0\"/>",
        f"      <geometry><box size=\"{2 * hx} {2 * hy} {2 * hz}\"/></geometry>",
        # Material tags intentionally omitted: they trigger the URDF importer to load
        # the MDL extension, which is not ABI-compatible with every Isaac Sim install.
        # Colors can be applied later via ``UrdfFileCfg.visual_material=PreviewSurfaceCfg(...)``.
        "    </visual>",
        "    <collision>",
        "      <origin xyz=\"0 0 0\"/>",
        f"      <geometry><box size=\"{2 * hx} {2 * hy} {2 * hz}\"/></geometry>",
        "    </collision>",
        "  </link>",
    ]

    for i in range(num_links):
        parent = "handle" if i == 0 else f"link_{i - 1}"
        origin_x = first_offset_x if i == 0 else link_step_x
        # Alternate bending axis between y (pitch) and z (yaw).
        axis_xyz = "0 1 0" if i % 2 == 0 else "0 0 1"

        out.extend([
            f"  <joint name=\"joint_{i}\" type=\"revolute\">",
            f"    <parent link=\"{parent}\"/>",
            f"    <child link=\"link_{i}\"/>",
            f"    <origin xyz=\"{origin_x} 0 0\"/>",
            f"    <axis xyz=\"{axis_xyz}\"/>",
            f"    <limit lower=\"{-joint_limit_rad}\" upper=\"{joint_limit_rad}\""
            f" effort=\"{joint_effort}\" velocity=\"{joint_velocity_limit}\"/>",
            f"    <dynamics damping=\"{joint_damping}\" friction=\"0\"/>",
            "  </joint>",
            f"  <link name=\"link_{i}\">",
            "    <inertial>",
            "      <origin xyz=\"0 0 0\"/>",
            f"      <mass value=\"{link_mass}\"/>",
            f"      <inertia ixx=\"{l_ixx}\" iyy=\"{l_iyy}\" izz=\"{l_izz}\""
            " ixy=\"0\" ixz=\"0\" iyz=\"0\"/>",
            "    </inertial>",
            "    <visual>",
            f"      <origin xyz=\"0 0 0\" rpy=\"{cyl_rpy[0]} {cyl_rpy[1]} {cyl_rpy[2]}\"/>",
            f"      <geometry><cylinder radius=\"{link_radius}\" length=\"{link_length}\"/></geometry>",
            # No <material>; see comment on the handle visual above.
            "    </visual>",
            "    <collision>",
            f"      <origin xyz=\"0 0 0\" rpy=\"{cyl_rpy[0]} {cyl_rpy[1]} {cyl_rpy[2]}\"/>",
            f"      <geometry><cylinder radius=\"{link_radius}\" length=\"{link_length}\"/></geometry>",
            "    </collision>",
            "  </link>",
        ])

    out.append("</robot>")
    return "\n".join(out) + "\n"


def _urdf_cache_path(urdf_text: str) -> str:
    digest = hashlib.sha1(urdf_text.encode("utf-8")).hexdigest()[:16]
    cache_dir = os.path.join(tempfile.gettempdir(), "isaaclab_cable_reach")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"cable_{digest}.urdf")


def build_cable_articulation_cfg(
    prim_path: str = "{ENV_REGEX_NS}/Cable",
    init_pos: tuple[float, float, float] = (0.5, 0.0, 0.05),
    # IsaacLab quaternion convention is (x, y, z, w). (0, 0, 0, 1) is identity.
    init_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    num_links: int = 20,
    link_length: float = 0.02,
    link_radius: float = 0.005,
    handle_size: tuple[float, float, float] = (0.03, 0.02, 0.02),
    handle_mass: float = 0.02,
    # Heavier cable links (~5 g each) are more stable under random-policy whipping.
    # Real cables are ~1 g/link but the extra inertia keeps the solver well-behaved.
    total_cable_mass: float = 0.1,
    # Strong joint damping keeps the chain from whipping when the gripper collides
    # with it under random-action exploration early in training.
    joint_damping: float = 0.5,
    joint_limit_deg: float = 60.0,
    joint_effort: float = 100.0,
    # Cap joint velocity so solver blowups don't produce NaN values downstream.
    joint_velocity_limit: float = 5.0,
    self_collision: bool = False,
) -> ArticulationCfg:
    """Build an :class:`ArticulationCfg` for a procedurally generated cable.

    Args:
        prim_path: USD prim path for the cable (supports ``{ENV_REGEX_NS}``).
        init_pos: Initial world position of the handle at spawn.
        init_rot: Initial world orientation of the handle at spawn, as a quaternion in
            ``(x, y, z, w)`` order (IsaacLab convention).
        num_links: Number of cylinder links in the cable body (not counting the handle).
        link_length: Length of each cylinder link [m].
        link_radius: Radius of each cylinder link [m].
        handle_size: (length, width, height) of the rigid handle block [m].
        handle_mass: Mass of the handle [kg].
        total_cable_mass: Total mass distributed equally across the ``num_links`` cable
            links [kg].
        joint_damping: Damping of each cable joint [N·m·s/rad].
        joint_limit_deg: Revolute joint limit [deg] (± per joint).
        joint_effort: Joint effort limit in the URDF [N·m].
        joint_velocity_limit: Joint velocity limit in the URDF [rad/s].
        self_collision: Whether the cable chain self-collides. Disabled by default for
            the first iteration to keep simulation cheap.

    Returns:
        An :class:`ArticulationCfg` whose spawn is a :class:`UrdfFileCfg` referring to a
        cached URDF file generated for these parameters.
    """
    link_mass = total_cable_mass / max(num_links, 1)
    joint_limit_rad = joint_limit_deg * (3.141592653589793 / 180.0)
    urdf_text = _cable_urdf(
        num_links=num_links,
        link_length=link_length,
        link_radius=link_radius,
        handle_size=handle_size,
        handle_mass=handle_mass,
        link_mass=link_mass,
        joint_damping=joint_damping,
        joint_limit_rad=joint_limit_rad,
        joint_effort=joint_effort,
        joint_velocity_limit=joint_velocity_limit,
    )
    path = _urdf_cache_path(urdf_text)
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(urdf_text)

    spawn = sim_utils.UrdfFileCfg(
        asset_path=path,
        fix_base=False,
        # Re-convert only when parameters change — the cache key is the URDF hash.
        # Leaving force conversion off avoids retriggering importer code paths (notably
        # the MDL material path) on every launch once a valid USD exists.
        self_collision=self_collision,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0,
                damping=joint_damping,
            ),
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=self_collision,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
    )

    init_state = ArticulationCfg.InitialStateCfg(
        pos=init_pos,
        rot=init_rot,
        joint_pos={"joint_.*": 0.0},
        joint_vel={"joint_.*": 0.0},
    )

    # Implicit passive actuator with zero stiffness — the cable's joints are not
    # actively driven. Damping matches the URDF-side dynamics so PhysX applies
    # consistent joint damping.
    actuators = {
        "cable_passive": ImplicitActuatorCfg(
            joint_names_expr=["joint_.*"],
            stiffness=0.0,
            damping=joint_damping,
        ),
    }

    return ArticulationCfg(
        prim_path=prim_path,
        spawn=spawn,
        init_state=init_state,
        actuators=actuators,
    )
