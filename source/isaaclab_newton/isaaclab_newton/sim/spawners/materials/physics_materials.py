# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-compatible deformable physics material spawning exports."""

import math
import warnings

from pxr import Usd, UsdShade

from isaaclab.sim.utils import clone, get_current_stage, safe_set_attribute_on_usd_prim

from .physics_materials_cfg import (
    NewtonDeformableBodyMaterialCfg,
    NewtonSurfaceDeformableBodyMaterialCfg,
    NewtonSurfaceDeformableMaterialCfg,
    NewtonVolumeDeformableMaterialCfg,
)


@clone
def spawn_deformable_body_material(
    prim_path: str, cfg: NewtonDeformableBodyMaterialCfg | NewtonSurfaceDeformableBodyMaterialCfg
) -> Usd.Prim:
    """Spawn canonical fields with legacy conversion and a deprecated material-thickness bridge."""
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        prim = UsdShade.Material.Define(stage, prim_path).GetPrim()
    if not prim.IsA(UsdShade.Material):
        raise ValueError(f"A non-material prim already exists at '{prim_path}'.")
    apply_deformable_material(cfg, prim_path, stage)
    return prim


def apply_deformable_material(
    cfg: NewtonDeformableBodyMaterialCfg
    | NewtonSurfaceDeformableBodyMaterialCfg
    | NewtonVolumeDeformableMaterialCfg
    | NewtonSurfaceDeformableMaterialCfg,
    prim_path: str,
    stage: Usd.Stage,
) -> bool:
    """Author canonical fields using the deprecated material-thickness bridge for surfaces."""
    prim = stage.GetPrimAtPath(prim_path)
    surface = isinstance(cfg, (NewtonSurfaceDeformableBodyMaterialCfg, NewtonSurfaceDeformableMaterialCfg))
    values = {"density": cfg.density}
    if surface:
        thickness = cfg.thickness
        if thickness is None and cfg.particle_radius is not None:
            thickness = 2 * cfg.particle_radius
        if thickness is None:
            for name in ("physics:surfaceThickness", "physics:thickness"):
                attribute = prim.GetAttribute(name)
                if attribute.HasAuthoredValueOpinion():
                    thickness = attribute.Get()
                    break
        if thickness is None:
            thickness = 0.016
        if not math.isfinite(thickness) or thickness <= 0:
            raise ValueError("Deformable material thickness must be finite and positive.")
        if cfg.thickness is None and cfg.density is not None:
            values["density"] = cfg.density / thickness
        values.update(
            surfaceThickness=thickness,
            surfaceStretchStiffness=(
                cfg.stretch_stiffness * thickness if cfg.stretch_stiffness is not None else cfg.tri_ke
            ),
            surfaceShearStiffness=cfg.shear_stiffness * thickness if cfg.shear_stiffness is not None else None,
            surfaceBendStiffness=cfg.bend_stiffness * thickness**3 if cfg.bend_stiffness is not None else cfg.edge_ke,
        )
        if all(
            value is None
            for value in (
                cfg.thickness,
                cfg.particle_radius,
                cfg.density,
                cfg.stretch_stiffness,
                cfg.shear_stiffness,
                cfg.bend_stiffness,
                cfg.tri_ke,
                cfg.edge_ke,
            )
        ):
            values.pop("surfaceThickness")
        legacy = ("particle_radius", "tri_ke", "edge_ke")
        unsupported = ("tri_ka", "tri_kd", "edge_kd")
    else:
        youngs, poissons = cfg.youngs_modulus, cfg.poissons_ratio
        if youngs is None or poissons is None:
            if cfg.k_mu is not None or cfg.k_lambda is not None:
                mu = cfg.k_mu if cfg.k_mu is not None else 1e5
                lam = cfg.k_lambda if cfg.k_lambda is not None else 1e5
                if mu <= 0 or lam + mu <= 0:
                    raise ValueError("Legacy Lamé parameters must define a valid elastic material.")
                youngs = mu * (3 * lam + 2 * mu) / (lam + mu) if youngs is None else youngs
                poissons = lam / (2 * (lam + mu)) if poissons is None else poissons
        values.update(youngsModulus=youngs, poissonsRatio=poissons)
        legacy = ("k_mu", "k_lambda")
        unsupported = ("k_damp", "particle_radius")
    converted = [name for name in legacy if getattr(cfg, name) is not None]
    if converted:
        warnings.warn(
            f"Legacy Newton material fields {converted} are deprecated; use canonical material fields.",
            DeprecationWarning,
            stacklevel=2,
        )
    ignored = {name: getattr(cfg, name) for name in unsupported if getattr(cfg, name) not in (None, 0.0)}
    if ignored:
        warnings.warn(
            f"Newton USD import does not support material fields {ignored}; these values are not applied."
            + (" Volume collision radius must be configured on the Newton builder." if not surface else ""),
            UserWarning,
            stacklevel=2,
        )
    prim.AddAppliedSchema(f"Physics{'Surface' if surface else 'Volume'}DeformableMaterialAPI")
    for name, value in values.items():
        if value is not None:
            safe_set_attribute_on_usd_prim(prim, f"physics:{name}", value, camel_case=False)
    return True


__all__ = ["spawn_deformable_body_material"]
