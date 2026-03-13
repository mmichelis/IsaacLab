# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses

from isaaclab.utils import configclass


@configclass
class OmniPhysicsPropertiesCfg:
    """OmniPhysics properties for a deformable body.

    These properties are set with the prefix ``omniphysics:<property_name>``. For example, to set the mass of the
    deformable body, you would set the property ``omniphysics:mass``.

    See the OmniPhysics documentation for more information on the available properties.
    """

    deformable_body_enabled: bool | None = None
    """Enables deformable body."""

    kinematic_enabled: bool = False
    """Enables kinematic body. Defaults to False, which means that the body is not kinematic.

    Similar to rigid bodies, this allows setting user-driven motion for the deformable body. For more information,
    please refer to the `documentation <https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/SoftBodies.html#kinematic-soft-bodies>`__.
    """


@configclass
class PhysXDeformableBodyPropertiesCfg:
    """PhysX-specific properties for a deformable body.

    These properties are set with the prefix ``physxDeformableBody:<property_name>``. For example, to set the rest
    offset of the deformable body, you would set the property ``physxDeformableBody:restOffset``.

    See the PhysX documentation for more information on the available properties.
    """

    solver_position_iteration_count: int | None = None
    """Number of the solver positional iterations per step. Range is [1,255]"""

    self_collision: bool | None = None
    """Whether to enable or disable self-collisions for the deformable body based on the rest position distances."""

    self_collision_filter_distance: float | None = None
    """Penetration value that needs to get exceeded before contacts for self collision are generated.

    This parameter must be greater than of equal to twice the :attr:`rest_offset` value.

    This value has an effect only if :attr:`self_collision` is enabled.
    """

    settling_threshold: float | None = None
    """Threshold vertex velocity (in m/s) under which sleep damping is applied in addition to velocity damping."""

    sleep_damping: float | None = None
    """Coefficient for the additional damping term if fertex velocity drops below setting threshold."""

    sleep_threshold: float | None = None
    """The velocity threshold (in m/s) under which the vertex becomes a candidate for sleeping in the next step."""


    vertex_velocity_damping: float | None = None
    """Coefficient for artificial damping on the vertex velocity.

    This parameter can be used to approximate the effect of air drag on the deformable body.
    """

    collision_simplification: bool = True
    """Whether or not to simplify the collision mesh before creating a soft body out of it. Defaults to True.

    Note:
        This flag is ignored if the user provides the simulation mesh points directly. However, we assume that
        most users will not provide the simulation mesh points directly. Hence, this flag is enabled by default.

        If you want to provide the simulation mesh points directly, please set this flag to False.
    """

    collision_simplification_remeshing: bool = True
    """Whether or not the collision mesh should be remeshed before simplification. Defaults to True.

    This parameter is ignored if :attr:`collision_simplification` is False.
    """

    collision_simplification_remeshing_resolution: int = 0
    """The resolution used for remeshing. Defaults to 0, which means that a heuristic is used to determine the
    resolution.

    This parameter is ignored if :attr:`collision_simplification_remeshing` is False.
    """

    collision_simplification_target_triangle_count: int = 0
    """The target triangle count used for the simplification. Defaults to 0, which means that a heuristic based on
    the :attr:`simulation_hexahedral_resolution` is used to determine the target count.

    This parameter is ignored if :attr:`collision_simplification` is False.
    """

    collision_simplification_force_conforming: bool = True
    """Whether or not the simplification should force the output mesh to conform to the input mesh. Defaults to True.

    The flag indicates that the tretrahedralizer used to generate the collision mesh should produce tetrahedra
    that conform to the triangle mesh. If False, the simplifier uses the output from the tretrahedralizer used.

    This parameter is ignored if :attr:`collision_simplification` is False.
    """

    max_depenetration_velocity: float | None = None
    """Maximum depenetration velocity permitted to be introduced by the solver (in m/s)."""


@configclass
class PhysXCollisionPropertiesCfg:
    """PhysX-specific collision properties for a deformable body.

    These properties are set with the prefix ``physxCollision:<property_name>``. For example, to set the rest offset
    of the deformable body's collision mesh, you would set the property ``physxCollision:restOffset``.

    See the PhysX documentation for more information on the available properties.
    """
    
    contact_offset: float | None = None
    """Contact offset for the collision shape (in m).

    The collision detector generates contact points as soon as two shapes get closer than the sum of their
    contact offsets. This quantity should be non-negative which means that contact generation can potentially start
    before the shapes actually penetrate.
    """

    rest_offset: float | None = None
    """Rest offset for the collision shape (in m).

    The rest offset quantifies how close a shape gets to others at rest, At rest, the distance between two
    vertically stacked objects is the sum of their rest offsets. If a pair of shapes have a positive rest
    offset, the shapes will be separated at rest by an air gap.
    """

@configclass
class DeformableBodyPropertiesCfg(OmniPhysicsPropertiesCfg, PhysXDeformableBodyPropertiesCfg, PhysXCollisionPropertiesCfg):
    """Properties to apply to a deformable body.

    A deformable body is a body that can deform under forces. The configuration allows users to specify
    the properties of the deformable body, such as the solver iteration counts, damping, and self-collision.

    An FEM-based deformable body is created by providing a collision mesh and simulation mesh. The collision mesh
    is used for collision detection and the simulation mesh is used for simulation. The collision mesh is usually
    a simplified version of the simulation mesh.

    Based on the above, the PhysX team provides APIs to either set the simulation and collision mesh directly
    (by specifying the points) or to simplify the collision mesh based on the simulation mesh. The simplification
    process involves remeshing the collision mesh and simplifying it based on the target triangle count.

    Since specifying the collision mesh points directly is not a common use case, we only expose the parameters
    to simplify the collision mesh based on the simulation mesh. If you want to provide the collision mesh points,
    please open an issue on the repository and we can add support for it.

    See :meth:`modify_deformable_body_properties` for more information.

    .. note::
        If the values are :obj:`None`, they are not modified. This is useful when you want to set only a subset of
        the properties and leave the rest as-is.
    """

    _property_prefix: dict[str, list[str]] = {
        "omniphysics": [field.name for field in dataclasses.fields(OmniPhysicsPropertiesCfg)],
        "physxDeformableBody": [field.name for field in dataclasses.fields(PhysXDeformableBodyPropertiesCfg)],
        "physxCollision": [field.name for field in dataclasses.fields(PhysXCollisionPropertiesCfg)],
    }
    """Mapping between the property prefixes and the properties that fall under each prefix."""
