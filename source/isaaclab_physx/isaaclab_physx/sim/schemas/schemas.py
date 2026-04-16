# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: Usd.Stage | None
from __future__ import annotations

import logging

import numpy as np

from omni.physx.scripts import deformableUtils
from pxr import Usd, UsdPhysics, UsdGeom, Sdf

from isaaclab.sim.utils import (
    apply_nested,
    create_prim,
    get_all_matching_child_prims,
    safe_set_attribute_on_usd_prim,
)
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.string import to_camel_case

from isaaclab_physx.sim.schemas.schemas_cfg import DeformableBodyPropertiesCfg

# import logger
logger = logging.getLogger(__name__)


"""
Deformable body properties.
"""


def define_deformable_body_properties(
    prim_path: str,
    cfg: DeformableBodyPropertiesCfg,
    stage: Usd.Stage | None = None,
    deformable_type: str = "volume",
    sim_mesh_prim_path: str | None = None,
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
        deformable_type: The type of the deformable body (surface or volume).
            This is used to determine which PhysX API to use for the deformable body. Defaults to "volume".
        sim_mesh_prim_path: Optional override for the simulation mesh prim path.
            If None, it is set to ``{prim_path}/sim_mesh`` for surface deformables
            and ``{prim_path}/sim_tetmesh`` for volume deformables.

    Raises:
        ValueError: When the prim path is not valid.
        ValueError: When the prim has no mesh or multiple meshes.
        RuntimeError: When setting the deformable body properties fails.
    """
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # get USD prim
    root_prim = stage.GetPrimAtPath(prim_path)
    # check if prim path is valid
    if not root_prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")

    # traverse the prim and get the mesh. If none or multiple meshes are found, raise error.
    matching_prims = get_all_matching_child_prims(prim_path, lambda p: p.GetTypeName() == "Mesh")
    # check if the volume deformable mesh is valid
    if len(matching_prims) == 0:
        raise ValueError(f"Could not find any mesh in '{prim_path}'. Please check asset.")
    if len(matching_prims) > 1:
        # get list of all meshes found
        mesh_paths = [p.GetPrimPath() for p in matching_prims]
        raise ValueError(
            f"Found multiple meshes in '{prim_path}': {mesh_paths}."
            " Deformable body schema can only be applied to one mesh for now."
        )
    mesh_prim = matching_prims[0]
    mesh_prim_path = mesh_prim.GetPrimPath()

    # check if the prim is valid
    if not mesh_prim.IsValid():
        raise ValueError(f"Mesh prim path '{mesh_prim_path}' is not valid.")

    # set root prim properties based on the type of the deformable mesh (surface vs volume)
    sim_mesh_prim_path = prim_path + "/sim_mesh" if sim_mesh_prim_path is None else sim_mesh_prim_path
    if deformable_type == "surface":
        success = deformableUtils.create_auto_surface_deformable_hierarchy(
            stage=stage,
            root_prim_path=prim_path,
            simulation_mesh_path=sim_mesh_prim_path,
            cooking_src_mesh_path=mesh_prim_path,
            cooking_src_simplification_enabled=False,
            set_visibility_with_guide_purpose=True,
        )
        if not success:
            raise RuntimeError(f"Failed to set deformable body properties on prim '{mesh_prim_path}'.")
    elif deformable_type == "volume":
        # remove potential previous configuration
        deformableUtils.remove_deformable_body(stage, prim_path)

        # extract surface mesh vertices and faces, create tetrahedral volume mesh
        vertices = np.array(mesh_prim.GetAttribute("points").Get())
        faces = np.array(mesh_prim.GetAttribute("faceVertexIndices").Get()).flatten()
        tet_mesh_points, tet_mesh_indices = deformableUtils.compute_conforming_tetrahedral_mesh(vertices, faces)
        _, surface_face_indices = deformableUtils.extractTriangleSurfaceFromTetra(tet_mesh_points, tet_mesh_indices)
        tet_mesh_indices = np.asarray(tet_mesh_indices).reshape(-1, 4)
        sim_mesh_prim = create_prim(
            sim_mesh_prim_path,
            prim_type="TetMesh",
            attributes={
                "points": tet_mesh_points,
                "tetVertexIndices": tet_mesh_indices,
            },
            stage=stage,
        )

        # apply sim API
        if not sim_mesh_prim.ApplyAPI("OmniPhysicsVolumeDeformableSimAPI"):
            raise RuntimeError(f"Failed to set volume deformable body API on prim '{sim_mesh_prim_path}'.")

        # apply collision API
        if not sim_mesh_prim.ApplyAPI(UsdPhysics.CollisionAPI):
            raise RuntimeError(f"Failed to set volume deformable collision API on prim '{sim_mesh_prim_path}'.")
        
        # set surface faces and rest-shape attributes required by OmniPhysicsVolumeDeformableSimAPI
        surface_face_indices = UsdGeom.TetMesh.ComputeSurfaceFaces(UsdGeom.TetMesh(sim_mesh_prim), Usd.TimeCode.Default())
        UsdGeom.TetMesh(sim_mesh_prim).GetSurfaceFaceVertexIndicesAttr().Set(surface_face_indices)
        sim_mesh_prim.GetAttribute("omniphysics:restShapePoints").Set(tet_mesh_points)
        sim_mesh_prim.GetAttribute("omniphysics:restTetVtxIndices").Set(tet_mesh_indices)

        # apply bind pose deformable pose API
        purposes = ["bindPose"]
        mesh_prim.ApplyAPI("OmniPhysicsDeformablePoseAPI", "default")
        mesh_prim.CreateAttribute("deformablePose:default:omniphysics:purposes", Sdf.ValueTypeNames.TokenArray).Set(purposes)
        point_based = UsdGeom.PointBased(mesh_prim)
        points = point_based.GetPointsAttr().Get()
        mesh_prim.CreateAttribute("deformablePose:default:omniphysics:points", Sdf.ValueTypeNames.Point3fArray).Set(points)

        sim_mesh_prim.GetPrim().ApplyAPI("OmniPhysicsDeformablePoseAPI", "default")
        sim_mesh_prim.GetPrim().CreateAttribute("deformablePose:default:omniphysics:purposes", Sdf.ValueTypeNames.TokenArray).Set(purposes)

        # apply deformable body api
        if not root_prim.ApplyAPI("OmniPhysicsDeformableBodyAPI"):
            raise RuntimeError(f"Failed to set volume deformable body API on prim '{prim_path}'.")

    else:
        raise ValueError(
            f"""Unsupported deformable type: '{deformable_type}'.
            Only surface and volume deformables are supported."""
        )

    # set deformable body properties
    modify_deformable_body_properties(prim_path, cfg, stage)


@apply_nested
def modify_deformable_body_properties(prim_path: str, cfg: DeformableBodyPropertiesCfg, stage: Usd.Stage | None = None):
    """Modify PhysX parameters for a deformable body prim.

    A `deformable body`_ is a single body (either surface or volume deformable) that can be simulated by PhysX.
    Unlike rigid bodies, deformable bodies support relative motion of the nodes in the mesh.
    Consequently, they can be used to simulate deformations under applied forces.

    PhysX deformable body simulation employs Finite Element Analysis (FEA) to simulate the deformations of the mesh.
    It uses two meshes to represent the deformable body:

    1. **Simulation mesh**: This mesh is used for the simulation and is the one that is deformed by the solver.
    2. **Collision mesh**: This mesh only needs to match the surface of the simulation mesh and is used for
       collision detection.

    For most applications, we assume that the above two meshes are computed from the "render mesh" of the deformable
    body. The render mesh is the mesh that is visible in the scene and is used for rendering purposes. It is composed
    of triangles, while the simulation mesh is composed of tetrahedrons for volume deformables,
    and triangles for surface deformables.

    .. caution::
        The deformable body schema is still under development by the Omniverse team. The current implementation
        works with the PhysX schemas shipped with Isaac Sim 6.0.0 onwards. It may change in future releases.

    .. note::
        This function is decorated with :func:`apply_nested` that sets the properties to all the prims
        (that have the schema applied on them) under the input prim path.

    .. _deformable body: https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/DeformableVolume.html
    .. _PhysxDeformableBodyAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/physxschema/annotated.html

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
    # check if deformable body API is applied
    if "OmniPhysicsDeformableBodyAPI" not in deformable_body_prim.GetAppliedSchemas():
        return False

    # apply customization to deformable API
    if "PhysxBaseDeformableBodyAPI" not in deformable_body_prim.GetAppliedSchemas():
        deformable_body_prim.AddAppliedSchema("PhysxBaseDeformableBodyAPI")

    # ensure PhysX collision API is applied on the collision mesh
    if "PhysxCollisionAPI" not in deformable_body_prim.GetAppliedSchemas():
        deformable_body_prim.AddAppliedSchema("PhysxCollisionAPI")

    # convert to dict
    cfg = cfg.to_dict()
    # set into PhysX API
    if cfg["kinematic_enabled"]:
        logger.warning(
            "Kinematic deformable bodies are not fully supported in the current version of Omni Physics. "
            "Setting kinematic_enabled to True may lead to unexpected behavior."
        )
    # prefixes for each attribute (collision attributes: physxCollision:*, and physxDeformable:* for rest)
    property_prefixes = cfg["_property_prefix"]
    for prefix, attr_list in property_prefixes.items():
        for attr_name in attr_list:
            safe_set_attribute_on_usd_prim(
                deformable_body_prim, f"{prefix}:{to_camel_case(attr_name, 'cC')}", cfg[attr_name], camel_case=False
            )
    # success
    return True
