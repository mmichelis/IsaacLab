# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from typing import ClassVar, Literal

from isaaclab.sim.spawners.materials.physics_materials_cfg import (
    DeformableBodyMaterialCfg,
    NewtonDeformableMaterialCfg,
    OmniPhysicsDeformableMaterialCfg,
    OmniPhysicsSurfaceDeformableMaterialCfg,
    PhysXDeformableMaterialCfg,
    RigidBodyMaterialBaseCfg,
    SurfaceDeformableBodyMaterialCfg,
)
from isaaclab.utils import configclass

__all__ = [
    "DeformableBodyMaterialCfg",
    "NewtonDeformableMaterialCfg",
    "OmniPhysicsDeformableMaterialCfg",
    "OmniPhysicsSurfaceDeformableMaterialCfg",
    "PhysXDeformableMaterialCfg",
    "PhysxRigidBodyMaterialCfg",
    "RigidBodyMaterialCfg",
    "SurfaceDeformableBodyMaterialCfg",
]


@configclass
class PhysxRigidBodyMaterialCfg(RigidBodyMaterialBaseCfg):
    """PhysX-specific physics-material parameters for rigid bodies.

    Extends :class:`~isaaclab.sim.spawners.materials.RigidBodyMaterialBaseCfg` with the
    `PhysxMaterialAPI`_ schema fields: compliant-contact spring (stiffness/damping) and the
    friction/restitution combine-mode tokens. None of these fields have a Newton consumer
    today; they are PhysX-engine-only knobs.

    See :meth:`~isaaclab.sim.spawners.materials.spawn_rigid_body_material` for more information.

    .. _PhysxMaterialAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_material_a_p_i.html
    """

    # -- Class metadata (not dataclass fields) --
    # USD applied schema written when at least one PhysX-namespaced field is set.
    _usd_applied_schema: ClassVar[str | None] = "PhysxMaterialAPI"
    # Prim attribute namespace for PhysX-specific fields.
    _usd_namespace: ClassVar[str | None] = "physxMaterial"

    compliant_contact_stiffness: float | None = None
    """Spring stiffness for a compliant contact model using implicit springs.

    A higher stiffness results in behavior closer to a rigid contact. The compliant contact model
    is only enabled if the stiffness is larger than 0. PhysX-only; not consumed by Newton.
    """

    compliant_contact_damping: float | None = None
    """Damping coefficient for a compliant contact model using implicit springs.

    Irrelevant if compliant contacts are disabled when :attr:`compliant_contact_stiffness` is set
    to zero and rigid contacts are active. PhysX-only; not consumed by Newton.
    """

    friction_combine_mode: Literal["average", "min", "multiply", "max"] | None = None
    """Determines the way friction will be combined during collisions.

    .. attention::

        When two physics materials with different combine modes collide, the combine mode with
        the higher priority will be used. The priority order is provided `here
        <https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/structPxCombineMode.html>`__.
    """

    restitution_combine_mode: Literal["average", "min", "multiply", "max"] | None = None
    """Determines the way restitution coefficient will be combined during collisions.

    .. attention::

        When two physics materials with different combine modes collide, the combine mode with
        the higher priority will be used. The priority order is provided `here
        <https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/structPxCombineMode.html>`__.
    """


@configclass
class RigidBodyMaterialCfg(PhysxRigidBodyMaterialCfg):
    """Deprecated: use :class:`PhysxRigidBodyMaterialCfg` or
    :class:`~isaaclab.sim.spawners.materials.RigidBodyMaterialBaseCfg`.

    .. deprecated:: 4.6.22
        ``RigidBodyMaterialCfg`` has been split into
        :class:`~isaaclab.sim.spawners.materials.RigidBodyMaterialBaseCfg` (solver-common) and
        :class:`PhysxRigidBodyMaterialCfg` (PhysX-specific) and relocated to
        :mod:`isaaclab_physx.sim.spawners.materials`. This alias preserves backwards compatibility
        and is scheduled for removal in 5.0.
    """

    def __post_init__(self):
        warnings.warn(
            "'RigidBodyMaterialCfg' is deprecated and will be removed in 5.0. Use"
            " 'isaaclab_physx.sim.spawners.materials.PhysxRigidBodyMaterialCfg' for PhysX"
            " properties, or 'isaaclab.sim.spawners.materials.RigidBodyMaterialBaseCfg' for"
            " solver-common properties only.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__post_init__()
