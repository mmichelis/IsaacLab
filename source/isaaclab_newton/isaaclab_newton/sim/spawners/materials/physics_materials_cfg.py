# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

from isaaclab.sim.spawners.materials.physics_materials_cfg import (
    DeformableBodyMaterialBaseCfg,
    RigidBodyMaterialFragment,
    SurfaceDeformableBodyMaterialBaseCfg,
)
from isaaclab.utils.configclass import configclass


@configclass
class NewtonDeformableMaterialCfg:
    """Common UsdPhysics material properties for a Newton deformable body."""

    _usd_namespace: ClassVar[str | None] = "physics"
    _usd_applied_schema: ClassVar[str | None] = "PhysicsMaterialAPI"
    _usd_field_exceptions: ClassVar[dict] = {}

    density: float = 1.0
    """The material density [kg/m^3]. Defaults to 1.0 kg/m^3."""


@configclass
class NewtonDeformableBodyMaterialCfg(DeformableBodyMaterialBaseCfg, NewtonDeformableMaterialCfg):
    """UsdPhysics material parameters for Newton volume deformable bodies."""

    _usd_namespace: ClassVar[str | None] = "physics"
    _usd_applied_schema: ClassVar[str | None] = "PhysicsVolumeDeformableMaterialAPI"
    _usd_field_exceptions: ClassVar[dict] = {}

    func: Callable | str = "isaaclab_newton.sim.spawners.materials.physics_materials:spawn_deformable_body_material"

    youngs_modulus: float = 2.5e5
    """Young's modulus [Pa]. Defaults to 2.5e5 Pa."""

    poissons_ratio: float = 0.25
    """Poisson's ratio [dimensionless]. Defaults to 0.25."""

    particle_radius: float | None = None
    """Deprecated particle contact radius [m]. It has no canonical USD equivalent."""

    k_mu: float | None = None
    """Deprecated first Lamé parameter [Pa]. Use :attr:`youngs_modulus` and :attr:`poissons_ratio`."""

    k_lambda: float | None = None
    """Deprecated second Lamé parameter [Pa]. Use :attr:`youngs_modulus` and :attr:`poissons_ratio`."""

    k_damp: float | None = None
    """Deprecated tetrahedral damping [Pa·s]. It has no canonical USD equivalent."""


@configclass
class NewtonSurfaceDeformableBodyMaterialCfg(SurfaceDeformableBodyMaterialBaseCfg, NewtonDeformableMaterialCfg):
    """UsdPhysics material parameters for Newton surface deformable bodies."""

    _usd_namespace: ClassVar[str | None] = "physics"
    _usd_applied_schema: ClassVar[str | None] = "PhysicsSurfaceDeformableMaterialAPI"
    _usd_field_exceptions: ClassVar[dict] = {}

    func: Callable | str = "isaaclab_newton.sim.spawners.materials.physics_materials:spawn_deformable_body_material"

    density: float = 62.5
    """The material density [kg/m^3]. Defaults to 62.5 kg/m^3."""

    thickness: float = 0.016
    """The surface thickness [m]. Defaults to 0.016 m."""

    stretch_stiffness: float = 6.25e5
    """The stretch stiffness [Pa]. Defaults to 6.25e5 Pa."""

    shear_stiffness: float | None = None
    """The shear stiffness [Pa]. Defaults to None."""

    bend_stiffness: float = 1_220_703.125
    """The bend stiffness [Pa]. Defaults to 1,220,703.125 Pa."""

    particle_radius: float | None = None
    """Deprecated particle contact radius [m]. Use :attr:`thickness`."""

    tri_ke: float | None = None
    """Deprecated triangle stretch stiffness. Use :attr:`stretch_stiffness`."""

    tri_ka: float | None = None
    """Deprecated triangle area stiffness. It has no canonical USD equivalent."""

    tri_kd: float | None = None
    """Deprecated triangle damping. It has no canonical USD equivalent."""

    edge_ke: float | None = None
    """Deprecated edge bending stiffness. Use :attr:`bend_stiffness`."""

    edge_kd: float | None = None
    """Deprecated edge damping. It has no canonical USD equivalent."""


@configclass
class NewtonMaterialCfg(RigidBodyMaterialFragment):
    """``newton:*`` rigid-body material attributes read by Newton's USD material schema resolver.

    Single-namespace fragment (see
    :class:`~isaaclab.sim.spawners.materials.RigidBodyMaterialFragment`) for the Newton-only
    friction knobs (torsional and rolling friction) and the per-material contact model (contact
    stiffness/damping, friction gain, adhesion) that replaces the deprecated per-shape
    ``ke``/``kd``/``kf``/``ka`` parameters. The ``NewtonMaterialAPI`` schema is applied (and the
    ``newton:*`` attributes authored) by the generic :func:`~isaaclab.sim.schemas.apply_namespaced`
    writer. ``None`` fields are left unchanged.

    .. note::
        The generated ``NewtonMaterialAPI`` USD schema currently only declares the two friction
        attributes; the four contact attributes are still authored as raw ``newton:*`` USD
        attributes and are read directly by Newton's schema resolver.

    Composes with other rigid-body material fragments (e.g.
    :class:`~isaaclab.sim.spawners.materials.UsdPhysicsRigidBodyMaterialCfg`) in the same fragment
    list passed to
    :func:`~isaaclab.sim.spawners.materials.spawn_rigid_body_material_from_fragments`. For the
    legacy (non-fragment) equivalent, see
    :class:`~isaaclab_newton.sim.schemas.NewtonMaterialPropertiesCfg`.
    """

    _usd_namespace: ClassVar[str | None] = "newton"
    _usd_applied_schema: ClassVar[str | None] = "NewtonMaterialAPI"

    torsional_friction: float | None = None
    """Torsional friction coefficient (resistance to spinning at a contact point) [dimensionless].

    Writes ``newton:torsionalFriction``. Range: [0, inf).
    """

    rolling_friction: float | None = None
    """Rolling friction coefficient (resistance to rolling motion) [dimensionless].

    Writes ``newton:rollingFriction``. Range: [0, inf).
    """

    contact_stiffness: float | None = None
    """Contact normal-force stiffness [N/m].

    Writes ``newton:contactStiffness``. Replaces the deprecated per-shape ``ke`` contact parameter;
    used by the SemiImplicit, Featherstone, MuJoCo, and VBD solvers.
    """

    contact_damping: float | None = None
    """Contact normal-force damping coefficient [N·s/m].

    Writes ``newton:contactDamping``. Replaces the deprecated per-shape ``kd`` contact parameter;
    used by the SemiImplicit, Featherstone, MuJoCo, and VBD solvers.
    """

    contact_friction_gain: float | None = None
    """Friction-force stiffness gain used by the tangential (friction) contact response [N·s/m].

    Writes ``newton:contactFrictionGain``. Replaces the deprecated per-shape ``kf`` contact
    parameter; used by the SemiImplicit and Featherstone solvers.
    """

    contact_adhesion: float | None = None
    """Contact adhesion distance: shapes closer than this threshold experience an attractive
    (adhesive) force [m].

    Writes ``newton:contactAdhesion``. Replaces the deprecated per-shape ``ka`` contact parameter;
    used by the SemiImplicit and Featherstone solvers.
    """
