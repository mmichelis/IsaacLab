# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared partition and selector helpers for proxy-coupled Newton managers.

These helpers split a Newton :class:`~newton.Model` between two solver entries
(``entry_a`` and ``entry_b``) and resolve proxy-body selectors. They are used
by both :class:`~isaaclab_contrib.deformable.proxy_coupled_mjwarp_vbd_manager.NewtonProxyCoupledMJWarpVBDManager`
and :class:`~isaaclab_contrib.deformable.proxy_coupled_mjwarp_mpm_manager.NewtonProxyCoupledMJWarpMPMManager`.

Body selectors accept either :class:`~isaaclab.managers.SceneEntityCfg`
(scoped by the asset's ``prim_path``, optionally narrowed by ``body_names``
full-matched against body short names) or raw prim-path regex strings
(e.g. ``"/World/envs/env_.*/MyCube"``) matched against ``model.body_label``
via ``^<string>(/|$)``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from newton import Model, ShapeFlags

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveSceneCfg


def resolve_entity_to_body_ids(
    model: Model,
    spec: SceneEntityCfg | str,
    scene_cfg: InteractiveSceneCfg | None,
    *,
    cfg_label: str,
    field: str,
) -> list[int]:
    """Resolve one selector to ``model.body_label`` indices.

    Args:
        model: Finalized Newton model whose ``body_label`` array is matched.
        spec: A :class:`SceneEntityCfg` (resolved through ``scene_cfg``) or a
            raw prim-path regex string matched via ``^<spec>(/|$)``.
        scene_cfg: Scene config used to look up assets when ``spec`` is a
            :class:`SceneEntityCfg`; ignored for string specs.
        cfg_label: Name of the calling config class (e.g.
            ``"ProxyCoupledMJWarpMPMSolverCfg"``), used in error messages.
        field: Name of the calling config field (e.g. ``"mjwarp_bodies"``),
            used in error messages.

    Returns:
        Sorted list of matched body indices in ``model``.

    Raises:
        ValueError: Asset missing on ``scene_cfg``; ``body_names`` pattern
            with zero matches; or a string spec with zero matches.
    """
    if isinstance(spec, str):
        prim_path, patterns, spec_repr = spec, None, f"prim-path regex {spec!r}"
    else:
        asset_cfg = getattr(scene_cfg, spec.name, None) if scene_cfg is not None else None
        if asset_cfg is None or not hasattr(asset_cfg, "prim_path"):
            raise ValueError(
                f"{cfg_label}.{field}: scene entity {spec.name!r} "
                f"is not on the attached scene cfg (or lacks `prim_path`)."
            )
        prim_path = asset_cfg.prim_path
        patterns = [spec.body_names] if isinstance(spec.body_names, str) else spec.body_names
        spec_repr = f"asset {spec.name!r}"

    asset_re = re.compile(rf"^{prim_path}(/|$)")
    # Treat patterns=None as ".*" so the loop is uniform across both branches.
    compiled = [re.compile(p) for p in (patterns if patterns is not None else [r".*"])]
    matched = [False] * len(compiled)
    body_ids: list[int] = []
    for b in range(int(model.body_count)):
        lbl = model.body_label[b]
        if not asset_re.match(lbl):
            continue
        short = lbl.rsplit("/", 1)[-1]
        hit = next((i for i, rx in enumerate(compiled) if rx.fullmatch(short)), None)
        if hit is None:
            continue
        matched[hit] = True
        body_ids.append(b)

    if patterns is not None:
        unmatched = [p for p, ok in zip(patterns, matched) if not ok]
        if unmatched:
            raise ValueError(
                f"{cfg_label}.{field}: {spec_repr} has no bodies matching {unmatched}."
            )
    elif isinstance(spec, str) and not body_ids:
        raise ValueError(
            f"{cfg_label}.{field}: {spec_repr} matched no bodies in "
            f"`model.body_label` (labels are full post-clone prim paths)."
        )
    return body_ids


def partition_model_by_entities(
    model: Model,
    entry_a_bodies: list[SceneEntityCfg | str],
    entry_b_bodies: list[SceneEntityCfg | str],
    scene_cfg: InteractiveSceneCfg | None,
    *,
    cfg_label: str,
    entry_a_field: str,
    entry_b_field: str,
) -> tuple[list[int], list[int], list[int], list[int], list[int], list[int]]:
    """Split bodies / joints / shapes between two solver entries.

    Joints and shapes inherit their (child) body's owner. Static shapes
    (``body == -1``) always go to entry B so its proxy collision pipeline
    can test rigid proxies against the world.

    Args:
        model: Finalized Newton model.
        entry_a_bodies: Selectors routed to entry A (e.g. MJWarp).
        entry_b_bodies: Selectors routed to entry B (e.g. VBD or MPM).
        scene_cfg: Scene config for resolving :class:`SceneEntityCfg` specs.
        cfg_label: Name of the calling config class for error messages.
        entry_a_field: Field name on the config for entry A (e.g.
            ``"mjwarp_bodies"``), used in error messages.
        entry_b_field: Field name on the config for entry B (e.g.
            ``"vbd_bodies"`` or ``"mpm_bodies"``), used in error messages.

    Returns:
        Tuple ``(a_bodies, b_bodies, a_joints, b_joints, a_shapes, b_shapes)``
        of sorted/insertion-ordered index lists.

    Raises:
        ValueError: A body matches both partitions or neither.
    """
    a_owned: set[int] = set()
    for spec in entry_a_bodies:
        a_owned.update(
            resolve_entity_to_body_ids(model, spec, scene_cfg, cfg_label=cfg_label, field=entry_a_field)
        )
    b_owned: set[int] = set()
    for spec in entry_b_bodies:
        b_owned.update(
            resolve_entity_to_body_ids(model, spec, scene_cfg, cfg_label=cfg_label, field=entry_b_field)
        )

    def _preview(ids: list[int]) -> str:
        return ", ".join(f"{b}:{model.body_label[b]!r}" for b in ids[:5])

    if overlap := sorted(a_owned & b_owned):
        raise ValueError(
            f"{cfg_label}: {len(overlap)} bodies match both "
            f"`{entry_a_field}` and `{entry_b_field}` (first few: {_preview(overlap)})."
        )
    unclaimed = [b for b in range(int(model.body_count)) if b not in a_owned and b not in b_owned]
    if unclaimed:
        raise ValueError(
            f"{cfg_label}: {len(unclaimed)} bodies unclaimed by "
            f"`{entry_a_field}`/`{entry_b_field}` (first few: {_preview(unclaimed)})."
        )

    a_joints: list[int] = []
    b_joints: list[int] = []
    if int(model.joint_count):
        for j, c in enumerate(model.joint_child.numpy()):
            child = int(c)
            if child in a_owned:
                a_joints.append(j)
            elif child in b_owned:
                b_joints.append(j)

    a_shapes: list[int] = []
    b_shapes: list[int] = []
    if int(model.shape_count):
        for s, b in enumerate(model.shape_body.numpy()):
            body = int(b)
            if body < 0 or body in b_owned:
                b_shapes.append(s)
            elif body in a_owned:
                a_shapes.append(s)

    return sorted(a_owned), sorted(b_owned), a_joints, b_joints, a_shapes, b_shapes


def select_proxy_bodies(
    model: Model,
    proxy_bodies: list[SceneEntityCfg | str],
    scene_cfg: InteractiveSceneCfg | None,
    *,
    cfg_label: str,
) -> list[int]:
    """Resolve proxy bodies, filtered to those owning a ``COLLIDE_SHAPES`` shape.

    Args:
        model: Finalized Newton model.
        proxy_bodies: Selectors naming bodies to expose as proxies.
        scene_cfg: Scene config for resolving :class:`SceneEntityCfg` specs.
        cfg_label: Name of the calling config class for error messages.

    Returns:
        Deduplicated, insertion-ordered list of body indices that match a
        selector **and** own at least one :class:`newton.ShapeFlags.COLLIDE_SHAPES`
        shape.

    Raises:
        ValueError: A :class:`SceneEntityCfg` entry has ``body_names=None``
            (proxies must be a subset, not the whole asset).
    """
    if not proxy_bodies:
        return []

    shape_count = int(model.shape_count)
    collide_flag = int(ShapeFlags.COLLIDE_SHAPES)
    collide_bodies: set[int] = set()
    if shape_count:
        shape_body_np = model.shape_body.numpy()
        shape_flags_np = model.shape_flags.numpy()
        collide_bodies = {
            int(shape_body_np[s])
            for s in range(shape_count)
            if int(shape_body_np[s]) >= 0 and int(shape_flags_np[s]) & collide_flag
        }

    proxy_ids: list[int] = []
    seen: set[int] = set()
    for spec in proxy_bodies:
        if isinstance(spec, SceneEntityCfg) and spec.body_names is None:
            raise ValueError(
                f"{cfg_label}.proxy_bodies entry {spec.name!r} requires "
                f"`body_names` (proxies must be a subset of the asset)."
            )
        for body_id in resolve_entity_to_body_ids(
            model, spec, scene_cfg, cfg_label=cfg_label, field="proxy_bodies"
        ):
            if body_id in collide_bodies and body_id not in seen:
                seen.add(body_id)
                proxy_ids.append(body_id)

    return proxy_ids
