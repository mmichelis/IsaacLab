# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: Usd.Stage | None
from __future__ import annotations

import logging

from pxr import Usd, UsdGeom

from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.string import to_camel_case

from isaaclab.sim.utils import (
    apply_nested,
    safe_set_attribute_on_usd_prim,
)

from isaaclab.sim.schemas import schemas_cfg

from isaaclab_physx.sim.schemas.schemas_cfg import DeformableBodyPropertiesCfg

# import logger
logger = logging.getLogger(__name__)


"""
Deformable body properties.
"""

def define_deformable_body_properties(
    prim_path: str, cfg: DeformableBodyPropertiesCfg, stage: Usd.Stage | None = None
):
    """Apply the deformable body schema on the input prim and set its properties.

    See :func:`modify_deformable_body_properties` for more details on how the properties are set.

    .. note::
        If the input prim is not a mesh, this function will traverse the prim and find the first mesh
        under it. If no mesh or multiple meshes are found, an error is raised. This is because the deformable
        body schema can only be applied to a single mesh.

    Args:
        prim_path: The prim path where to apply the deformable body schema.
        cfg: The configuration for the deformable body.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Raises:
        ValueError: When the prim path is not valid.
        ValueError: When the prim has no mesh or multiple meshes.
    """
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # get USD prim
    prim = stage.GetPrimAtPath(prim_path)
    # check if prim path is valid
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")

    # set deformable body properties
    modify_deformable_body_properties(prim_path, cfg, stage)


@apply_nested
def modify_deformable_body_properties(
    prim_path: str, cfg: DeformableBodyPropertiesCfg, stage: Usd.Stage | None = None
):
    """Modify PhysX parameters for a deformable body prim.

    A `deformable body`_ is a single body that can be simulated by PhysX. Unlike rigid bodies, deformable bodies
    support relative motion of the nodes in the mesh. Consequently, they can be used to simulate deformations
    under applied forces.

    PhysX soft body simulation employs Finite Element Analysis (FEA) to simulate the deformations of the mesh.
    It uses two tetrahedral meshes to represent the deformable body:

    1. **Simulation mesh**: This mesh is used for the simulation and is the one that is deformed by the solver.
    2. **Collision mesh**: This mesh only needs to match the surface of the simulation mesh and is used for
       collision detection.

    For most applications, we assume that the above two meshes are computed from the "render mesh" of the deformable
    body. The render mesh is the mesh that is visible in the scene and is used for rendering purposes. It is composed
    of triangles and is the one that is used to compute the above meshes based on PhysX cookings.

    The schema comprises of attributes that belong to the `PhysxDeformableBodyAPI`_. schemas containing the PhysX
    parameters for the deformable body.

    .. caution::
        The deformable body schema is still under development by the Omniverse team. The current implementation
        works with the PhysX schemas shipped with Isaac Sim 4.0.0 onwards. It may change in future releases.

    .. note::
        This function is decorated with :func:`apply_nested` that sets the properties to all the prims
        (that have the schema applied on them) under the input prim path.

    .. _deformable body: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/SoftBodies.html
    .. _PhysxDeformableBodyAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_deformable_a_p_i.html

    Args:
        prim_path: The prim path to the deformable body.
        cfg: The configuration for the deformable body.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Returns:
        True if the properties were successfully set, False otherwise.
    """
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # get deformable-body USD prim
    deformable_body_prim = stage.GetPrimAtPath(prim_path)

    # check if the prim is valid
    if not deformable_body_prim.IsValid():
        return False

    from omni.physx.scripts import deformableUtils
    # set deformable body properties based on the type of the mesh (surface vs volume)
    if deformable_body_prim.IsA(UsdGeom.Mesh):
        success = deformableUtils.set_physics_surface_deformable_body(stage, prim_path)
        # apply physx extension api
        if "PhysxSurfaceDeformableBodyAPI" not in deformable_body_prim.GetAppliedSchemas():
            deformable_body_prim.AddAppliedSchema("PhysxSurfaceDeformableBodyAPI")
    elif deformable_body_prim.IsA(UsdGeom.TetMesh):
        success = deformableUtils.set_physics_volume_deformable_body(stage, prim_path)
        # apply physx extension api
        if "PhysxBaseDeformableBodyAPI" not in deformable_body_prim.GetAppliedSchemas():
            deformable_body_prim.AddAppliedSchema("PhysxBaseDeformableBodyAPI")
    else:
        print(f"Unsupported deformable body prim type: '{deformable_body_prim.GetTypeName()}'. Only Mesh and TetMesh are supported.")
        success = False
    # api failure
    if not success:
        return False

    # ensure PhysX collision API is applied on the collision mesh 
    if "PhysxCollisionAPI" not in deformable_body_prim.GetAppliedSchemas():
        deformable_body_prim.AddAppliedSchema("PhysxCollisionAPI")

    # convert to dict
    cfg = cfg.to_dict()
    # set into PhysX API (collision prim attributes: physxCollision:* for rest/contact offset, physxDeformable:* for rest on deformable prim)
    # prefixes for each attribute
    property_prefixes = cfg["_property_prefix"]
    for prefix, attr_list in property_prefixes.items():
        for attr_name in attr_list:
            safe_set_attribute_on_usd_prim(
                deformable_body_prim, f"{prefix}:{to_camel_case(attr_name, 'cC')}", cfg[attr_name], camel_case=False
            )

    # success
    return True
