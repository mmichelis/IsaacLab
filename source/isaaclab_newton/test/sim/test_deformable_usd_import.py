# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitless tests for Newton's canonical USD deformable import."""

import isaaclab_newton.physics.newton_manager as newton_manager_module
import newton
import pytest
from isaaclab_newton.physics import NewtonManager, deformable_groups
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import (
    NewtonDeformableBodyMaterialCfg,
    NewtonSurfaceDeformableBodyMaterialCfg,
)
from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx

from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.materials.physics_materials import spawn_deformable_body_material

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
        {"density": 1000.0, "thickness": 0.01, "stretchStiffness": 2000.0, "bendStiffness": 10.0},
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

    builder = newton.ModelBuilder()
    builder.add_usd(stage, schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()])
    assert builder._cloth_label == ["/World/Cloth/sim_mesh"]


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
            {"density": 1000.0, "thickness": 0.016, "stretchStiffness": 6.25e5, "bendStiffness": 1_220_703.125},
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
    """Canonical surface and volume schemas must populate Newton and its group metadata."""
    builder = newton.ModelBuilder()
    builder.add_usd(_make_deformable_stage(), schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()])

    assert (
        builder._cloth_label,
        builder._cloth_world,
        builder._cloth_particle_start,
        builder._cloth_particle_end,
    ) == ([_CLOTH_PATH], [-1], [0], [4])
    assert (
        builder._soft_label,
        builder._soft_world,
        builder._soft_particle_start,
        builder._soft_particle_end,
    ) == ([_SOFT_PATH], [-1], [4], [8])


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


def test_deformable_groups_replicate_with_world_offsets():
    """Replicated deformable groups must retain labels and offset every element range."""
    source = newton.ModelBuilder()
    source.add_usd(_make_deformable_stage(), schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()])
    scene = newton.ModelBuilder()
    scene.replicate(source, 2)

    assert (
        scene._cloth_label,
        scene._cloth_world,
        scene._cloth_particle_start,
        scene._cloth_particle_end,
    ) == ([_CLOTH_PATH, _CLOTH_PATH], [0, 1], [0, 8], [4, 12])
    assert (
        scene._soft_label,
        scene._soft_world,
        scene._soft_particle_start,
        scene._soft_particle_end,
    ) == ([_SOFT_PATH, _SOFT_PATH], [0, 1], [4, 12], [8, 16])


@pytest.mark.parametrize(
    ("attribute", "value", "error"),
    [
        ("_soft_particle_end", 9, "invalid particle range"),
        ("_soft_particle_start", 3, "particle ranges overlap"),
    ],
)
def test_manager_rejects_invalid_deformable_particle_ranges(
    monkeypatch: pytest.MonkeyPatch, attribute: str, value: int, error: str
):
    """All imported ranges must be valid even when filtering groups."""
    builder = newton.ModelBuilder()
    builder.add_usd(_make_deformable_stage(), schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()])
    getattr(builder, attribute)[0] = value
    monkeypatch.setattr(NewtonManager, "_builder", builder)

    with pytest.raises(RuntimeError, match=error):
        deformable_groups.get_deformable_particle_groups(builder, "/World/Cloth")


def test_manager_env_count_resets_before_flat_import(monkeypatch: pytest.MonkeyPatch):
    """A flat import must not inherit the previous replicated environment count."""
    stage = _make_deformable_stage()
    monkeypatch.setattr(newton_manager_module, "get_current_stage", lambda fabric=False: stage)

    NewtonManager._num_envs = 2
    try:
        NewtonManager.clear()
        assert NewtonManager.get_num_envs() is None

        NewtonManager.instantiate_builder_from_stage()
        assert NewtonManager.get_num_envs() == 1
        groups = deformable_groups.get_deformable_particle_groups(NewtonManager.get_builder())
        assert groups and {group.world for group in groups} == {0}
    finally:
        NewtonManager.clear()


def test_manager_skips_unsupported_visual_meshes(monkeypatch: pytest.MonkeyPatch, caplog):
    """Unsupported visual topology must not block canonical deformable physics."""
    stage = _make_deformable_stage()
    builder = newton.ModelBuilder()
    builder.add_usd(stage, schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()])

    monkeypatch.setattr(newton_manager_module, "get_current_stage", lambda fabric=False: stage)
    monkeypatch.setattr(NewtonManager, "_builder", builder)

    bindings = deformable_groups.get_deformable_visual_bindings(builder, stage)
    assert [binding.visual_prim_path for binding in bindings] == [_CLOTH_PATH]
    assert "expected one visual mesh" in caplog.text

    soft_visual = UsdGeom.Mesh.Define(stage, "/World/Soft/Visual")
    soft_visual.CreatePointsAttr([(0.0, 0.0, 2.0)] * 3)
    caplog.clear()
    bindings = deformable_groups.get_deformable_visual_bindings(builder, stage)
    assert [binding.visual_prim_path for binding in bindings] == [_CLOTH_PATH]
    assert "visual mesh has 3 points, imported 4" in caplog.text

    soft_visual.GetPointsAttr().Set(
        [
            (1.0, 0.0, 2.0),
            (0.0, 0.0, 2.0),
            (0.0, 1.0, 2.0),
            (0.0, 0.0, 3.0),
        ]
    )
    caplog.clear()
    bindings = deformable_groups.get_deformable_visual_bindings(builder, stage)
    assert [binding.visual_prim_path for binding in bindings] == [_CLOTH_PATH]
    assert "visual points do not match imported particles" in caplog.text


def test_manager_resolves_imported_groups_and_visual_meshes(monkeypatch: pytest.MonkeyPatch):
    """Newton manager bindings derive from importer groups and USD visual meshes."""
    stage = _make_deformable_stage()
    builder = newton.ModelBuilder()
    builder.add_usd(stage, schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()])

    cloth_visual = UsdGeom.Mesh.Define(stage, "/World/Cloth/Visual")
    cloth_visual.CreatePointsAttr(
        [
            (-1.0, 0.0, 1.0),
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
        ]
    )
    cloth_visual.AddTranslateOp().Set((1.0, 0.0, 0.0))
    soft_visual = UsdGeom.Mesh.Define(stage, "/World/Soft/Visual")
    soft_visual.CreatePointsAttr(
        [
            (0.0, 0.0, 2.0),
            (1.0, 0.0, 2.0),
            (0.0, 1.0, 2.0),
            (0.0, 0.0, 3.0),
        ]
    )

    monkeypatch.setattr(newton_manager_module, "get_current_stage", lambda fabric=False: stage)
    monkeypatch.setattr(NewtonManager, "_builder", builder)

    groups = deformable_groups.get_deformable_particle_groups(builder, "/World/Cloth")
    assert [(group.family, group.world, group.particle_start, group.particle_end) for group in groups] == [
        ("cloth", 0, 0, 4)
    ]

    bindings = deformable_groups.get_deformable_visual_bindings(builder, stage)
    assert [
        (binding.visual_prim_path, binding.world, binding.particle_start, binding.particle_count)
        for binding in bindings
    ] == [
        ("/World/Cloth/Visual", 0, 0, 4),
        ("/World/Soft/Visual", 0, 4, 4),
    ]
