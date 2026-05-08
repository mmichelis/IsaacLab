# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import MISSING
from typing import ClassVar

from isaaclab.utils import configclass

# Names that moved out of this submodule into ``isaaclab_physx.sim.spawners.materials.physics_materials_cfg``.
# Resolved lazily so callers using ``from isaaclab.sim.spawners.materials.physics_materials_cfg
# import RigidBodyMaterialCfg`` continue to work without importing ``isaaclab_physx`` at module
# load time.
_PHYSX_FORWARDS = frozenset({"RigidBodyMaterialCfg", "PhysxRigidBodyMaterialCfg"})


def __getattr__(name):
    if name in _PHYSX_FORWARDS:
        try:
            from isaaclab_physx.sim.spawners.materials import physics_materials_cfg as _physx_mat_cfg
        except ImportError as e:
            raise ImportError(
                f"'isaaclab.sim.spawners.materials.physics_materials_cfg.{name}' has moved to"
                " 'isaaclab_physx.sim.spawners.materials.physics_materials_cfg'. Install the"
                " isaaclab_physx extension or update your import. This forwarding shim is scheduled"
                " for removal in 5.0."
            ) from e
        return getattr(_physx_mat_cfg, name)
    raise AttributeError(f"module 'isaaclab.sim.spawners.materials.physics_materials_cfg' has no attribute {name!r}")


@configclass
class PhysicsMaterialCfg:
    """Configuration parameters for creating a physics material.

    Physics material are PhysX schemas that can be applied to a USD material prim to define the
    physical properties related to the material. For example, the friction coefficient, restitution
    coefficient, etc. For more information on physics material, please refer to the
    `PhysX documentation <https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/classPxBaseMaterial.html>`__.
    """

    func: Callable = MISSING
    """Function to use for creating the material."""


@configclass
class RigidBodyMaterialBaseCfg(PhysicsMaterialCfg):
    """Solver-common physics-material parameters for rigid bodies.

    Contains the friction and restitution fields from the `UsdPhysics.MaterialAPI`_ that are common
    across all simulation backends. For PhysX-only material properties (compliant-contact spring,
    combine modes), use :class:`~isaaclab_physx.sim.spawners.materials.PhysxRigidBodyMaterialCfg`.

    See :meth:`spawn_rigid_body_material` for more information.

    .. _UsdPhysics.MaterialAPI: https://openusd.org/dev/api/class_usd_physics_material_a_p_i.html
    """

    # -- Class metadata (not dataclass fields) --
    # ``static_friction`` / ``dynamic_friction`` / ``restitution`` write to ``physics:*``
    # (UsdPhysics standard attributes). The helper's per-declaring-class routing keeps
    # them under the base namespace even when the cfg is a PhysX subclass instance.
    _usd_namespace: ClassVar[str | None] = "physics"
    _usd_applied_schema: ClassVar[str | None] = None
    _usd_field_exceptions: ClassVar[dict] = {}

    func: Callable | str = "{DIR}.physics_materials:spawn_rigid_body_material"

    static_friction: float = 0.5
    """The static friction coefficient. Defaults to 0.5."""

    dynamic_friction: float = 0.5
    """The dynamic friction coefficient. Defaults to 0.5."""

    restitution: float = 0.0
    """The restitution coefficient. Defaults to 0.0."""


@configclass
class OmniPhysicsDeformableMaterialCfg:
    """OmniPhysics material properties for a deformable body.

    These properties are set with the prefix ``omniphysics:<property_name>``. For example, to set the density of the
    deformable body, you would set the property ``omniphysics:density``.

    See the OmniPhysics documentation for more information on the available properties.
    """

    density: float = 1000.0
    """The material density [kg/m^3]. Defaults to 1000.0 kg/m^3, which is the density of water."""

    static_friction: float = 0.25
    """The static friction coefficient. Defaults to 0.25."""

    dynamic_friction: float = 0.25
    """The dynamic friction coefficient. Defaults to 0.25."""

    youngs_modulus: float = 1000000.0
    """The Young's modulus, which defines the body's stiffness [Pa]. Defaults to 1 MPa."""

    poissons_ratio: float = 0.45
    """The Poisson's ratio which defines the body's volume preservation."""


@configclass
class OmniPhysicsSurfaceDeformableMaterialCfg(OmniPhysicsDeformableMaterialCfg):
    """OmniPhysics material properties for a surface deformable body.

    These properties are set with the prefix ``omniphysics:<property_name>``.
    """

    surface_thickness: float = 0.01
    """The thickness of the deformable body's surface [m]. Defaults to 0.01."""

    surface_stretch_stiffness: float = 0.0
    """The stretch stiffness of the deformable body's surface. Defaults to 0.0."""

    surface_shear_stiffness: float = 0.0
    """The shear stiffness of the deformable body's surface. Defaults to 0.0."""

    surface_bend_stiffness: float = 0.0
    """The bend stiffness of the deformable body's surface. Defaults to 0.0."""

    bend_damping: float = 0.0
    """The bend damping for the deformable body's surface. Defaults to 0.0."""


@configclass
class PhysXDeformableMaterialCfg:
    """PhysX-specific material properties for a deformable body.

    These properties are set with the prefix ``physxDeformableBody:<property_name>``.
    """

    elasticity_damping: float = 0.005
    """The elasticity damping for the deformable material. Defaults to 0.005."""


@configclass
class NewtonDeformableMaterialCfg:
    """Newton-specific material properties for a deformable body.

    These properties are set with the prefix ``newton:<property_name>``.
    """

    particle_radius: float = 0.008
    """Particle radius [m] used by the Newton backend."""

    # -- Cloth (triangle surface mesh) parameters

    tri_ke: float = 1e4
    """Triangle area-preserving stiffness [Pa]. Used by Newton backend for cloth meshes."""

    tri_ka: float = 1e4
    """Triangle area stiffness [Pa]. Used by Newton backend for cloth meshes."""

    tri_kd: float = 1.5e-6
    """Triangle area damping. Used by Newton backend for cloth meshes."""

    edge_ke: float = 5.0
    """Bending stiffness. Used by Newton backend for cloth meshes."""

    edge_kd: float = 1e-2
    """Bending damping. Used by Newton backend for cloth meshes."""

    # -- Volumetric (tetrahedral FEM) parameters

    k_damp: float = 0.0
    """Damping stiffness for tetrahedral elements. Defaults to 0.0."""


@configclass
class DeformableBodyMaterialCfg(
    PhysicsMaterialCfg,
    OmniPhysicsDeformableMaterialCfg,
    PhysXDeformableMaterialCfg,
    NewtonDeformableMaterialCfg,
):
    """Physics material parameters for deformable bodies.

    See :meth:`spawn_deformable_body_material` for more information.
    """

    func: Callable | str = "{DIR}.physics_materials:spawn_deformable_body_material"

    _property_prefix: dict[str, list[str]] = {
        "omniphysics": [field.name for field in dataclasses.fields(OmniPhysicsDeformableMaterialCfg)],
        "physxDeformableBody": [field.name for field in dataclasses.fields(PhysXDeformableMaterialCfg)],
        "newton": [field.name for field in dataclasses.fields(NewtonDeformableMaterialCfg)],
    }
    """Mapping between the property prefixes and the properties that fall under each prefix."""


@configclass
class SurfaceDeformableBodyMaterialCfg(DeformableBodyMaterialCfg, OmniPhysicsSurfaceDeformableMaterialCfg):
    """Physics material parameters for surface deformable bodies.

    See :meth:`spawn_deformable_body_material` for more information.
    """

    func: Callable | str = "{DIR}.physics_materials:spawn_deformable_body_material"

    _property_prefix: dict[str, list[str]] = {
        "omniphysics": [field.name for field in dataclasses.fields(OmniPhysicsSurfaceDeformableMaterialCfg)],
        "physxDeformableBody": [field.name for field in dataclasses.fields(PhysXDeformableMaterialCfg)],
        "newton": [field.name for field in dataclasses.fields(NewtonDeformableMaterialCfg)],
    }
    """Extend DeformableBodyMaterialCfg properties under each prefix."""
