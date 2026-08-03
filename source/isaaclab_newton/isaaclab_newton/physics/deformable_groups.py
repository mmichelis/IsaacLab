# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Discovery of Newton-imported deformable particle groups and their USD visual meshes."""

from __future__ import annotations

import logging
import re
from typing import NamedTuple

from pxr import Gf, Usd, UsdGeom

logger = logging.getLogger(__name__)


class DeformableParticleGroup(NamedTuple):
    """Newton particle range imported from one deformable simulation prim."""

    family: str
    sim_prim_path: str
    world: int
    particle_start: int
    particle_end: int


class DeformableVisualBinding(NamedTuple):
    """Visual mesh paired with one Newton deformable particle range."""

    visual_prim_path: str
    world: int
    particle_start: int
    particle_count: int


def get_deformable_particle_groups(builder, prim_path_expr: str | None = None) -> list[DeformableParticleGroup]:
    """Return Newton-imported deformable particle groups.

    Args:
        builder: The Newton model builder holding imported deformable metadata.
        prim_path_expr: Optional prim-path regular expression to filter groups by
            their simulation prim path. Defaults to ``None`` (no filtering).

    Returns:
        Deformable particle groups sorted by particle start index. Empty when
        ``builder`` is ``None``.
    """
    if builder is None:
        return []

    pattern = re.compile(rf"^(?:{prim_path_expr})(?:/.*)?$") if prim_path_expr is not None else None
    particle_count = int(builder.particle_count)
    groups: list[DeformableParticleGroup] = []
    for family in ("cloth", "soft"):
        labels = tuple(getattr(builder, f"_{family}_label", ()))
        worlds = tuple(getattr(builder, f"_{family}_world", ()))
        starts = tuple(getattr(builder, f"_{family}_particle_start", ()))
        ends = tuple(getattr(builder, f"_{family}_particle_end", ()))
        if not (len(labels) == len(worlds) == len(starts) == len(ends)):
            raise RuntimeError(f"Newton {family} group metadata has inconsistent lengths.")

        for label, world, start, end in zip(labels, worlds, starts, ends, strict=True):
            if not isinstance(label, str):
                raise RuntimeError(f"Newton {family} group metadata is missing a label.")
            start = int(start)
            end = int(end)
            if start < 0 or end <= start or end > particle_count:
                raise RuntimeError(f"Newton {family} group '{label}' has invalid particle range ({start}, {end}).")
            groups.append(
                DeformableParticleGroup(
                    family=family,
                    sim_prim_path=label,
                    world=0 if world is None or int(world) < 0 else int(world),
                    particle_start=start,
                    particle_end=end,
                )
            )

    groups.sort(key=lambda group: group.particle_start)
    for previous, current in zip(groups, groups[1:]):
        if current.particle_start < previous.particle_end:
            raise RuntimeError(
                f"Newton deformable particle ranges overlap for '{previous.sim_prim_path}' and"
                f" '{current.sim_prim_path}'."
            )
    return groups if pattern is None else [group for group in groups if pattern.fullmatch(group.sim_prim_path)]


def get_deformable_visual_bindings(builder, stage) -> list[DeformableVisualBinding]:
    """Pair Newton deformable groups with their USD visual meshes.

    Args:
        builder: The Newton model builder holding imported deformable metadata.
        stage: The USD stage to resolve visual meshes against.

    Returns:
        Visual bindings for deformable groups whose visual mesh matches the
        imported particle range. Groups with ambiguous or mismatched meshes are
        skipped with a warning.
    """
    xform_cache = UsdGeom.XformCache()

    def _has_api(prim, api_name: str) -> bool:
        return any(schema.split(":")[0] == api_name for schema in prim.GetPrimTypeInfo().GetAppliedAPISchemas())

    def _has_sim_api(prim) -> bool:
        return _has_api(prim, "PhysicsSurfaceDeformableSimAPI") or _has_api(prim, "PhysicsVolumeDeformableSimAPI")

    bindings: list[DeformableVisualBinding] = []
    for group in get_deformable_particle_groups(builder):
        sim_prim = stage.GetPrimAtPath(group.sim_prim_path)
        if not sim_prim.IsValid():
            raise RuntimeError(f"Newton deformable simulation prim not found at '{group.sim_prim_path}'.")

        body_prim = sim_prim
        while body_prim.IsValid() and not _has_api(body_prim, "PhysicsDeformableBodyAPI"):
            parent = body_prim.GetParent()
            if not parent.IsValid() or parent == body_prim:
                break
            body_prim = parent
        if not _has_api(body_prim, "PhysicsDeformableBodyAPI"):
            body_prim = sim_prim

        visual_prims = [prim for prim in Usd.PrimRange(body_prim) if prim.IsA(UsdGeom.Mesh) and not _has_sim_api(prim)]
        if not visual_prims and sim_prim.IsA(UsdGeom.Mesh):
            visual_prims = [sim_prim]
        if len(visual_prims) != 1:
            paths = [prim.GetPath().pathString for prim in visual_prims]
            logger.warning(
                "Skipping visual sync for Newton deformable '%s': expected one visual mesh, found %s.",
                group.sim_prim_path,
                paths,
            )
            continue

        visual_prim = visual_prims[0]
        points = UsdGeom.PointBased(visual_prim).GetPointsAttr().Get()
        particle_count = group.particle_end - group.particle_start
        if points is None or len(points) != particle_count:
            point_count = 0 if points is None else len(points)
            logger.warning(
                "Skipping visual sync for Newton deformable '%s': visual mesh has %d points, imported %d.",
                group.sim_prim_path,
                point_count,
                particle_count,
            )
            continue
        world_transform = xform_cache.GetLocalToWorldTransform(visual_prim)
        particle_points = builder.particle_q[group.particle_start : group.particle_end]
        points_match = all(
            Gf.IsClose(
                world_transform.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2]))),
                Gf.Vec3d(float(particle[0]), float(particle[1]), float(particle[2])),
                1.0e-5,
            )
            for point, particle in zip(points, particle_points, strict=True)
        )
        if not points_match:
            logger.warning(
                "Skipping visual sync for Newton deformable '%s': visual points do not match imported particles.",
                group.sim_prim_path,
            )
            continue
        bindings.append(
            DeformableVisualBinding(
                visual_prim_path=visual_prim.GetPath().pathString,
                world=group.world,
                particle_start=group.particle_start,
                particle_count=particle_count,
            )
        )
    return bindings
