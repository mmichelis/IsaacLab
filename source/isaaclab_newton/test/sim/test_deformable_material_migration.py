# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native importer contracts for legacy and canonical material inputs."""

import newton
import numpy as np
import pytest
from isaaclab_newton.sim.spawners.materials import (
    NewtonSurfaceDeformableMaterialCfg,
    NewtonVolumeDeformableMaterialCfg,
)

from pxr import Usd, UsdGeom, UsdShade

from isaaclab.sim.spawners.materials import spawn_physics_material_from_fragments


@pytest.mark.parametrize("canonical", [False, True])
def test_surface_material_native_mass_and_stiffness(canonical):
    """Preserve areal density or explicit volumetric density without guessing numeric sentinels."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    body = UsdGeom.Xform.Define(stage, "/Body").GetPrim()
    body.AddAppliedSchema("PhysicsDeformableBodyAPI")
    mesh = UsdGeom.Mesh.Define(stage, "/Body/sim")
    mesh.GetPrim().AddAppliedSchema("PhysicsSurfaceDeformableSimAPI")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    cfg = NewtonSurfaceDeformableMaterialCfg(
        density=62.5,
        thickness=0.02 if canonical else None,
        particle_radius=None if canonical else 0.01,
        tri_ke=123.0,
        edge_ke=5.0,
    )
    prim = spawn_physics_material_from_fragments("/Material", cfg, stage)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(UsdShade.Material(prim), materialPurpose="physics")
    builder = newton.ModelBuilder()
    builder.add_usd(stage)
    assert sum(builder.particle_mass) == pytest.approx(0.625 if canonical else 31.25)
    assert np.asarray(builder.tri_materials)[0, 0] == pytest.approx(123.0)
    assert cfg.density == 62.5
    if canonical:
        spawn_physics_material_from_fragments(
            "/Material", NewtonSurfaceDeformableMaterialCfg(stretch_stiffness=100.0), stage
        )
        updated = newton.ModelBuilder()
        updated.add_usd(stage)
        assert sum(updated.particle_mass) == pytest.approx(0.625)
        assert np.asarray(updated.tri_materials)[0, 0] == pytest.approx(2.0)
        assert prim.GetAttribute("physics:surfaceThickness").Get() == pytest.approx(0.02)


def test_volume_material_conversion_and_unsupported_warning():
    """Convert a legacy Lamé pair and expose unsupported damping rather than dropping it silently."""
    stage = Usd.Stage.CreateInMemory()
    cfg = NewtonVolumeDeformableMaterialCfg(k_mu=100000.0, k_lambda=100000.0, k_damp=2.0)
    with pytest.warns(UserWarning, match="k_damp.*2.0.*not applied"):
        prim = spawn_physics_material_from_fragments("/Material", cfg, stage)
    assert prim.GetAttribute("physics:youngsModulus").Get() == pytest.approx(250000.0)
    assert prim.GetAttribute("physics:poissonsRatio").Get() == pytest.approx(0.25)
    assert not prim.GetAttribute("physics:kDamp")
    assert cfg.k_damp == 2.0


def test_surface_material_rejects_zero_thickness():
    """Reject invalid conversion units before emitting material attributes."""
    stage = Usd.Stage.CreateInMemory()
    with pytest.raises(ValueError, match="thickness"):
        spawn_physics_material_from_fragments(
            "/Material", NewtonSurfaceDeformableMaterialCfg(particle_radius=0.0), stage
        )
