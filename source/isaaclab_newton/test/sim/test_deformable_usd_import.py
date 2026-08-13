# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitless tests for Newton's canonical USD deformable import."""

import isaaclab_newton.physics.newton_manager as newton_manager_module
import newton
import numpy as np
import pytest
import warp as wp
from isaaclab_newton.physics import NewtonManager
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import (
    NewtonDeformableBodyMaterialCfg,
    NewtonSurfaceDeformableBodyMaterialCfg,
)
from isaaclab_newton.sim.spawners.materials.physics_materials import spawn_deformable_body_material
from newton.selection import DeformableView
from newton.usd import SchemaResolverNewton, SchemaResolverPhysx

from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils

_CLOTH_PATH = "/World/Cloth/Sim"
_SOFT_PATH = "/World/Soft/Sim"


def _bind_deformable_material(
    stage: Usd.Stage,
    prim: Usd.Prim,
    path: str,
    api_schema: str,
    attributes: dict[str, float],
):
    material = UsdShade.Material.Define(stage, path)
    material.GetPrim().AddAppliedSchema("PhysicsMaterialAPI")
    material.GetPrim().AddAppliedSchema(api_schema)
    for name, value in attributes.items():
        material.GetPrim().CreateAttribute(f"physics:{name}", Sdf.ValueTypeNames.Float).Set(value)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")


def _make_deformable_stage() -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.Scene.Define(stage, "/PhysicsScene")

    cloth_body = UsdGeom.Xform.Define(stage, "/World/Cloth").GetPrim()
    cloth_body.AddAppliedSchema("PhysicsDeformableBodyAPI")
    cloth = UsdGeom.Mesh.Define(stage, _CLOTH_PATH)
    cloth.CreatePointsAttr([(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)])
    cloth.CreateFaceVertexCountsAttr([3, 3])
    cloth.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
    cloth.GetPrim().AddAppliedSchema("PhysicsSurfaceDeformableSimAPI")
    cloth.GetPrim().AddAppliedSchema("PhysicsCollisionAPI")
    _bind_deformable_material(
        stage,
        cloth.GetPrim(),
        "/World/Cloth/Material",
        "PhysicsSurfaceDeformableMaterialAPI",
        {"density": 62.5, "thickness": 0.01, "stretchStiffness": 2000.0, "bendStiffness": 10.0},
    )

    soft_body = UsdGeom.Xform.Define(stage, "/World/Soft").GetPrim()
    soft_body.AddAppliedSchema("PhysicsDeformableBodyAPI")
    soft = UsdGeom.TetMesh.Define(stage, _SOFT_PATH)
    soft.CreatePointsAttr([(0.0, 0.0, 2.0), (1.0, 0.0, 2.0), (0.0, 1.0, 2.0), (0.0, 0.0, 3.0)])
    soft.CreateTetVertexIndicesAttr([(0, 1, 2, 3)])
    soft.GetPrim().AddAppliedSchema("PhysicsVolumeDeformableSimAPI")
    soft.GetPrim().AddAppliedSchema("PhysicsCollisionAPI")
    _bind_deformable_material(
        stage,
        soft.GetPrim(),
        "/World/Soft/Material",
        "PhysicsVolumeDeformableMaterialAPI",
        {"density": 1200.0, "youngsModulus": 9000.0, "poissonsRatio": 0.25},
    )
    return stage


def test_newton_body_cfg_authors_canonical_usd():
    """Newton body cfg must author canonical schemas and import as cloth."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage, "/World/Cloth")
    visual = UsdGeom.Mesh.Define(stage, "/World/Cloth/Visual")
    visual.CreatePointsAttr([(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)])
    visual.CreateFaceVertexCountsAttr([3, 3])
    visual.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
    cfg = NewtonDeformableBodyPropertiesCfg(
        body_enabled=True,
        kinematic_enabled=False,
        mass=2.0,
        density=1000.0,
    )

    with sim_utils.use_stage(stage):
        sim_utils.define_deformable_body_properties(
            "/World/Cloth",
            cfg,
            stage=stage,
            deformable_type="surface",
        )

    body = stage.GetPrimAtPath("/World/Cloth")
    schemas = set(body.GetPrimTypeInfo().GetAppliedAPISchemas())
    assert "PhysicsDeformableBodyAPI" in schemas
    assert body.GetAttribute("physics:bodyEnabled").Get() is True
    assert body.GetAttribute("physics:kinematicEnabled").Get() is False
    assert body.GetAttribute("physics:mass").Get() == pytest.approx(2.0)
    assert body.GetAttribute("physics:density").Get() == pytest.approx(1000.0)
    assert not any(attribute.GetName().startswith("newton:") for attribute in body.GetAuthoredAttributes())

    sim_prim = stage.GetPrimAtPath("/World/Cloth/sim_mesh")
    sim_schemas = set(sim_prim.GetPrimTypeInfo().GetAppliedAPISchemas())
    assert "PhysicsSurfaceDeformableSimAPI" in sim_schemas
    assert "PhysicsCollisionAPI" in sim_schemas
    assert not any(schema.startswith("OmniPhysics") for schema in sim_schemas)

    _bind_deformable_material(
        stage,
        sim_prim,
        "/World/Cloth/Material",
        "PhysicsSurfaceDeformableMaterialAPI",
        {"thickness": 0.002, "stretchStiffness": 1.0, "bendStiffness": 1.0},
    )

    builder = newton.ModelBuilder()
    builder.add_usd(stage, schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()])
    model = builder.finalize(device="cpu")
    cloth = DeformableView(model, "/World/Cloth/sim_mesh", family="surface", verbose=False)
    assert cloth.ranges("particle") == [(0, 4)]


@pytest.mark.parametrize(
    ("cfg", "family_api", "expected"),
    [
        (
            NewtonDeformableBodyMaterialCfg(),
            "PhysicsVolumeDeformableMaterialAPI",
            {"density": 1.0, "youngsModulus": 2.5e5, "poissonsRatio": 0.25},
        ),
        (
            NewtonSurfaceDeformableBodyMaterialCfg(),
            "PhysicsSurfaceDeformableMaterialAPI",
            {"density": 62.5, "thickness": 0.016, "stretchStiffness": 6.25e5, "bendStiffness": 1_220_703.125},
        ),
    ],
)
def test_newton_material_cfgs_author_canonical_usd(cfg, family_api, expected):
    """Newton material cfg inheritance must author the base and family schemas."""
    stage = Usd.Stage.CreateInMemory()
    with sim_utils.use_stage(stage):
        prim = spawn_deformable_body_material("/Material", cfg)

    authored_schemas = prim.GetPrimStack()[0].GetInfo("apiSchemas").prependedItems
    assert set(authored_schemas) == {"PhysicsMaterialAPI", family_api}
    for name, value in expected.items():
        assert prim.GetAttribute(f"physics:{name}").Get() == pytest.approx(value)
    assert not any(attribute.GetName().startswith("newton:") for attribute in prim.GetAuthoredAttributes())


def test_add_usd_imports_canonical_surface_and_volume_deformables():
    """Canonical surface and volume schemas must create addressable Newton deformables."""
    builder = newton.ModelBuilder()
    builder.add_usd(_make_deformable_stage(), schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()])
    model = builder.finalize(device="cpu")

    cloth = DeformableView(model, _CLOTH_PATH, family="surface", verbose=False)
    soft = DeformableView(model, _SOFT_PATH, family="volume", verbose=False)
    assert cloth.labels == [_CLOTH_PATH]
    assert cloth.worlds == [-1]
    assert cloth.ranges("particle") == [(0, 4)]
    assert soft.labels == [_SOFT_PATH]
    assert soft.worlds == [-1]
    assert soft.ranges("particle") == [(4, 8)]


def test_cloth_particle_radius_derives_from_thickness():
    """Cloth particle radius is importer-set to half the authored thickness (0.01 -> 0.005)."""
    builder = newton.ModelBuilder()
    builder.add_usd(_make_deformable_stage(), schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()])

    assert builder.particle_radius[:4] == pytest.approx([0.005] * 4)


def test_volume_particle_radius_uses_builder_default():
    """Volume deformables use Newton's builder default because USD has no radius field."""
    builder = newton.ModelBuilder()
    builder.default_particle_radius = 0.012
    builder.add_usd(_make_deformable_stage(), schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()])

    assert builder.particle_radius[4:] == pytest.approx([0.012] * 4)


def test_deformable_views_replicate_with_world_offsets():
    """Replicated deformable views must retain labels and particle ranges per world."""
    source = newton.ModelBuilder()
    source.add_usd(_make_deformable_stage(), schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()])
    scene = newton.ModelBuilder()
    scene.replicate(source, 2)
    model = scene.finalize(device="cpu")

    cloth = DeformableView(model, _CLOTH_PATH, family="surface", verbose=False)
    soft = DeformableView(model, _SOFT_PATH, family="volume", verbose=False)
    assert cloth.labels == [_CLOTH_PATH, _CLOTH_PATH]
    assert cloth.worlds == [0, 1]
    assert cloth.ranges("particle") == [(0, 4), (8, 12)]
    assert soft.labels == [_SOFT_PATH, _SOFT_PATH]
    assert soft.worlds == [0, 1]
    assert soft.ranges("particle") == [(4, 8), (12, 16)]


def test_manager_env_count_resets_before_flat_import(monkeypatch: pytest.MonkeyPatch):
    """A flat import must not inherit the previous replicated environment count."""
    stage = _make_deformable_stage()
    monkeypatch.setattr(newton_manager_module, "get_current_stage", lambda fabric=False: stage)

    NewtonManager._num_envs = 2
    try:
        NewtonManager.clear()
        NewtonManager.instantiate_builder_from_stage()
        assert NewtonManager.get_num_envs() == 1
        model = NewtonManager._builder.finalize(device="cpu")
        cloth = DeformableView(model, _CLOTH_PATH, family="surface", verbose=False)
        soft = DeformableView(model, _SOFT_PATH, family="volume", verbose=False)
        assert cloth.world_count == soft.world_count == 1
        assert cloth.worlds == soft.worlds == [-1]
    finally:
        NewtonManager.clear()


def test_newton_owns_and_evaluates_imported_deformable_visual_meshes():
    """Newton must own and evaluate authored deformable graphics meshes."""
    stage = _make_deformable_stage()
    cloth_visual = UsdGeom.Mesh.Define(stage, "/World/Cloth/Visual")
    cloth_points = [(-0.75, 0.25, 1.0), (-0.25, 0.25, 1.0), (-0.5, 0.75, 1.0)]
    cloth_visual.CreatePointsAttr(cloth_points)
    cloth_visual.CreateFaceVertexCountsAttr([3])
    cloth_visual.CreateFaceVertexIndicesAttr([0, 1, 2])
    cloth_visual.AddTranslateOp().Set((1.0, 0.0, 0.0))
    soft_visual = UsdGeom.Mesh.Define(stage, "/World/Soft/Visual")
    soft_points = [(0.1, 0.1, 2.1), (0.7, 0.1, 2.1), (0.1, 0.7, 2.1)]
    soft_visual.CreatePointsAttr(soft_points)
    soft_visual.CreateFaceVertexCountsAttr([3])
    soft_visual.CreateFaceVertexIndicesAttr([0, 1, 2])

    builder = newton.ModelBuilder()
    builder.add_usd(stage, schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()])
    model = builder.finalize(device="cpu")
    meshes = {mesh.graphics_path: mesh for mesh in model.deformable_visual_meshes}
    assert set(meshes) == {"/World/Cloth/Visual", "/World/Soft/Visual"}
    cloth_mesh = meshes["/World/Cloth/Visual"]
    soft_mesh = meshes["/World/Soft/Visual"]
    assert cloth_mesh.kind == newton.DeformableVisualMesh.Kind.TRIANGLE
    assert cloth_mesh.body_path == "/World/Cloth"
    assert cloth_mesh.sim_path == _CLOTH_PATH
    assert soft_mesh.kind == newton.DeformableVisualMesh.Kind.TET
    assert soft_mesh.body_path == "/World/Soft"
    assert soft_mesh.sim_path == _SOFT_PATH
    assert all(mesh.world == -1 for mesh in meshes.values())

    visuals = model.deformable_visuals()
    state = model.state()
    model.update_deformable_visuals(state, visuals)
    cloth_world_points = np.asarray(cloth_points) + np.asarray([1.0, 0.0, 0.0])
    np.testing.assert_allclose(
        visuals.get_points(cloth_mesh).numpy(),
        cloth_world_points,
        atol=1.0e-5,
    )
    np.testing.assert_allclose(
        visuals.get_points(soft_mesh).numpy(),
        np.asarray(soft_points),
        atol=1.0e-5,
    )

    shift = np.asarray([0.5, -0.25, 0.75], dtype=np.float32)
    state.particle_q = wp.array(state.particle_q.numpy() + shift, dtype=wp.vec3, device=state.particle_q.device)
    model.update_deformable_visuals(state, visuals)
    np.testing.assert_allclose(visuals.get_points(cloth_mesh).numpy(), cloth_world_points + shift, atol=1.0e-5)
    np.testing.assert_allclose(visuals.get_points(soft_mesh).numpy(), np.asarray(soft_points) + shift, atol=1.0e-5)


def test_legacy_volume_material_fields_translate_to_canonical_usd():
    """Deprecated Lamé parameters remain accepted for one release."""
    stage = Usd.Stage.CreateInMemory()
    cfg = NewtonDeformableBodyMaterialCfg(k_mu=2.0, k_lambda=3.0, particle_radius=0.01, k_damp=0.5)

    with pytest.deprecated_call(match="deprecated"):
        with sim_utils.use_stage(stage):
            prim = spawn_deformable_body_material("/Material", cfg)

    assert prim.GetAttribute("physics:youngsModulus").Get() == pytest.approx(5.2)
    assert prim.GetAttribute("physics:poissonsRatio").Get() == pytest.approx(0.3)
    assert not any(
        name in {attribute.GetName() for attribute in prim.GetAuthoredAttributes()}
        for name in ("physics:kMu", "physics:kLambda", "physics:kDamp", "physics:particleRadius")
    )


def test_legacy_surface_material_fields_translate_to_canonical_usd():
    """Deprecated cloth fields remain accepted for one release."""
    stage = Usd.Stage.CreateInMemory()
    cfg = NewtonSurfaceDeformableBodyMaterialCfg(
        density=1.0,
        particle_radius=0.01,
        tri_ke=4.0,
        tri_ka=2.0,
        tri_kd=0.5,
        edge_ke=8.0e-6,
        edge_kd=0.25,
    )

    with pytest.deprecated_call(match="deprecated"):
        with sim_utils.use_stage(stage):
            prim = spawn_deformable_body_material("/Material", cfg)

    assert prim.GetAttribute("physics:density").Get() == pytest.approx(50.0)
    assert prim.GetAttribute("physics:thickness").Get() == pytest.approx(0.02)
    assert prim.GetAttribute("physics:stretchStiffness").Get() == pytest.approx(200.0)
    assert prim.GetAttribute("physics:bendStiffness").Get() == pytest.approx(1.0)
    legacy_names = {
        "physics:particleRadius",
        "physics:triKe",
        "physics:triKa",
        "physics:triKd",
        "physics:edgeKe",
        "physics:edgeKd",
    }
    assert not legacy_names.intersection(attribute.GetName() for attribute in prim.GetAuthoredAttributes())
