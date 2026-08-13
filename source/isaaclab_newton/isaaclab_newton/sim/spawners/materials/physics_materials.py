# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-compatible deformable physics material spawning exports."""

import warnings

from isaaclab.sim.spawners.materials.physics_materials import spawn_deformable_body_material as _spawn_canonical

from .physics_materials_cfg import NewtonDeformableBodyMaterialCfg, NewtonSurfaceDeformableBodyMaterialCfg


def spawn_deformable_body_material(
    prim_path: str, cfg: NewtonDeformableBodyMaterialCfg | NewtonSurfaceDeformableBodyMaterialCfg
):
    """Spawn a canonical deformable material while translating deprecated Newton fields."""
    material = cfg.copy()
    if isinstance(material, NewtonSurfaceDeformableBodyMaterialCfg):
        aliases = ("particle_radius", "tri_ke", "tri_ka", "tri_kd", "edge_ke", "edge_kd")
        used = [name for name in aliases if getattr(material, name) is not None]
        if used:
            warnings.warn(
                f"Newton surface material fields {used} are deprecated; use canonical USD material fields.",
                DeprecationWarning,
                stacklevel=2,
            )
            thickness = 2.0 * material.particle_radius if material.particle_radius is not None else material.thickness
            old_density = 1.0 if material.density == 62.5 else material.density
            material.thickness = thickness
            material.density = old_density / thickness
            material.stretch_stiffness = (material.tri_ke if material.tri_ke is not None else 1.0e4) / thickness
            material.bend_stiffness = (material.edge_ke if material.edge_ke is not None else 5.0) / thickness**3
    elif isinstance(material, NewtonDeformableBodyMaterialCfg):
        aliases = ("particle_radius", "k_mu", "k_lambda", "k_damp")
        used = [name for name in aliases if getattr(material, name) is not None]
        if used:
            warnings.warn(
                f"Newton volume material fields {used} are deprecated; use canonical USD material fields.",
                DeprecationWarning,
                stacklevel=2,
            )
            if material.k_mu is not None or material.k_lambda is not None:
                mu = material.k_mu if material.k_mu is not None else 1.0e5
                lam = material.k_lambda if material.k_lambda is not None else 1.0e5
                denominator = lam + mu
                material.youngs_modulus = mu * (3.0 * lam + 2.0 * mu) / denominator
                material.poissons_ratio = lam / (2.0 * denominator)
    else:
        aliases = ()

    for name in aliases:
        setattr(material, name, None)
    return _spawn_canonical(prim_path, material)


__all__ = ["spawn_deformable_body_material"]
