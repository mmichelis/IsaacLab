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
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

if TYPE_CHECKING:
    from isaaclab.sim.spawners.from_files.from_files_cfg import UrdfFileCfg


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

        # ``continuous`` joints have no position limits — they can rotate freely. We
        # use them instead of ``revolute`` with ±60° limits because the PhysX
        # articulation solver's penalty-based limit enforcement became unstable on
        # this 20-DOF chain (limits oscillated and the simulation diverged within a
        # few steps of any airborne state). Cable bending is naturally self-limited
        # by the damping + gravity equilibrium, so explicit limits aren't necessary.
        out.extend([
            f"  <joint name=\"joint_{i}\" type=\"continuous\">",
            f"    <parent link=\"{parent}\"/>",
            f"    <child link=\"link_{i}\"/>",
            f"    <origin xyz=\"{origin_x} 0 0\"/>",
            f"    <axis xyz=\"{axis_xyz}\"/>",
            # 'continuous' joints ignore lower/upper, but URDF requires effort and
            # velocity attributes. These are the importer's fallbacks for the drive.
            f"    <limit effort=\"{joint_effort}\" velocity=\"{joint_velocity_limit}\"/>",
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


def _do_spawn_cable_from_urdf(
    prim_path: str,
    cfg: "UrdfFileCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """URDF→USD + strip the auto-generated ``root_joint`` fixed joint.

    The ``urdf_usd_converter`` pipeline ALWAYS emits a ``PhysicsFixedJoint`` named
    ``root_joint`` on the URDF's root link, regardless of ``fix_base``. For a fixed-
    base robot this is correct; for a floating-base articulation (our cable) this
    pins the handle rigidly to the world, which disables gravity on the chain, makes
    teleports detonate the solver, and — crucially for us — prevents the gripper
    from ever lifting the handle off the table.

    The fixed-joint prim is defined in ``payloads/Physics/physics.usda`` and
    ``over``-ridden in the root layer plus the physx/mujoco payloads, so we sweep
    every layer and remove it wherever it appears before the USD is composed onto
    the simulation stage.
    """
    from pxr import Usd

    from isaaclab.sim import converters
    from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file

    urdf_loader = converters.UrdfConverter(cfg)
    usd_path = urdf_loader.usd_path

    if not cfg.fix_base:
        usd_dir = os.path.dirname(usd_path)
        layer_paths = [usd_path]
        payload_dir = os.path.join(usd_dir, "payloads", "Physics")
        if os.path.isdir(payload_dir):
            for fname in os.listdir(payload_dir):
                if fname.endswith((".usda", ".usd", ".usdc")):
                    layer_paths.append(os.path.join(payload_dir, fname))

        for layer_path in layer_paths:
            stage = Usd.Stage.Open(layer_path)
            if stage is None:
                continue
            modified = False
            for prim in list(stage.TraverseAll()):
                if prim.GetName() == "root_joint":
                    stage.RemovePrim(prim.GetPath())
                    modified = True
            if modified:
                stage.GetRootLayer().Save()

    return _spawn_from_usd_file(prim_path, usd_path, cfg, translation, orientation, **kwargs)


# Placeholder for the ``@clone``-decorated version of :func:`_do_spawn_cable_from_urdf`.
# We build it lazily on first invocation to avoid importing ``isaaclab.sim.utils.clone``
# (which transitively imports ``pxr``) at module-load time — that eager pxr import
# during Hydra config resolution (before Isaac Sim Kit has started) clashes with
# Kit's own extension loader and crashes the app on startup.
_cached_clone_wrapped_spawner = None


def _spawn_cable_from_urdf_floating_base(
    prim_path: str,
    cfg: "UrdfFileCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Entry point used by :class:`UrdfFileCfg.func`.

    On first call (once Kit is up), wraps :func:`_do_spawn_cable_from_urdf` with the
    ``@clone`` decorator so ``{ENV_REGEX_NS}/...`` prim-path patterns are resolved
    and the prim is copied across all parallel envs — the stock ``spawn_from_urdf``
    uses the same decorator. On subsequent calls the cached wrapped version is used.
    """
    global _cached_clone_wrapped_spawner
    if _cached_clone_wrapped_spawner is None:
        from isaaclab.sim.utils import clone

        _cached_clone_wrapped_spawner = clone(_do_spawn_cable_from_urdf)
    return _cached_clone_wrapped_spawner(prim_path, cfg, translation, orientation, **kwargs)


def build_cable_articulation_cfg(
    prim_path: str = "{ENV_REGEX_NS}/Cable",
    init_pos: tuple[float, float, float] = (0.5, 0.0, 0.05),
    # IsaacLab quaternion convention is (x, y, z, w). (0, 0, 0, 1) is identity.
    init_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    # DOF + mass tuning: the articulation solver diverges catastrophically when a
    # 20-DOF chain has ~2.5 g links (8:1 mass ratio with the 20 g handle) and ~10⁻⁸
    # kg·m² inertias — FP precision + cumulative constraint error explode. We
    # dramatically simplify: fewer, heavier links with matched mass to the handle.
    num_links: int = 2,
    link_length: float = 0.04,
    link_radius: float = 0.008,
    handle_size: tuple[float, float, float] = (0.03, 0.02, 0.02),
    handle_mass: float = 0.02,
    total_cable_mass: float = 0.04,
    # Light joint damping — previous value (0.5 N·m·s/rad) was ~500× critical for a
    # 2-gram link and effectively turned the cable into a rigid rod that resisted
    # being lifted, overwhelming the gripper clamp. 0.02 leaves enough stability
    # (combined with the ``invalid_cable_state`` failure termination that catches
    # real solver blowups) without defeating physical realism.
    joint_damping: float = 0.02,
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
        # Override the default spawner with one that strips the auto-generated
        # root_joint fixed joint — without this the handle is pinned to the world.
        func=_spawn_cable_from_urdf_floating_base,
        # Re-convert only when parameters change — the cache key is the URDF hash.
        # Leaving force conversion off avoids retriggering importer code paths (notably
        # the MDL material path) on every launch once a valid USD exists.
        self_collision=self_collision,
        # Explicitly enable gravity on every body in this articulation — the URDF
        # importer's default leaves ``disable_gravity`` unset, which for a floating-
        # base articulation can end up with gravity silently disabled. Without this,
        # the cable just floats.
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            max_linear_velocity=100.0,
            max_angular_velocity=500.0,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0,
                damping=joint_damping,
            ),
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=self_collision,
            # Heavy iteration counts for a 20-DOF chain — the default 16/1 diverges
            # within a few steps on any airborne state. 64/16 keeps it stable.
            solver_position_iteration_count=64,
            solver_velocity_iteration_count=16,
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
