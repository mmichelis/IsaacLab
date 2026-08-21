# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration shared by cable insertion environments."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import CableObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.peg_in_hole.peg_in_hole_env_cfg import EventCfg as PegInHoleEventCfg
from isaaclab_tasks.core.peg_in_hole.peg_in_hole_env_cfg import ObjectTableSceneCfg

if TYPE_CHECKING:
    from pxr import Usd

_CABLE_LENGTH = 0.30
_CABLE_SEGMENT_COUNT = 15
_PEG_HALF_LENGTH = 0.035


def _cable_rest_positions() -> list[tuple[float, float, float]]:
    """Create a smooth cable rest shape that exits along the peg axis."""
    segment_length = _CABLE_LENGTH / _CABLE_SEGMENT_COUNT
    positions = [(0.0, 0.0, 0.0)]
    x = 0.0
    z = 0.0
    for index in range(_CABLE_SEGMENT_COUNT):
        angle = 0.5 * math.pi * index / (_CABLE_SEGMENT_COUNT - 1)
        x += segment_length * math.sin(angle)
        z += segment_length * math.cos(angle)
        positions.append((x, 0.0, z))
    return positions


def spawn_attached_cable(
    prim_path: str,
    cfg: AttachedCableCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a cable with its first point attached to a sibling rigid body."""
    from pxr import Sdf

    cable_prim = sim_utils.spawn_cable(prim_path, cfg, translation, orientation, **kwargs)
    stage = cable_prim.GetStage()
    for cable_path in sim_utils.find_matching_prim_paths(prim_path, stage):
        target_path = f"{cable_path.rsplit('/', 1)[0]}/{cfg.target_prim_name}"
        attachment = stage.DefinePrim(f"{cable_path}/PegAttachment", "PhysicsAttachment")
        attachment.CreateRelationship("physics:src0").SetTargets([f"{cable_path}/geometry/mesh"])
        attachment.CreateRelationship("physics:src1").SetTargets([target_path])
        attachment.CreateAttribute("physics:type0", Sdf.ValueTypeNames.Token).Set("point")
        attachment.CreateAttribute("physics:type1", Sdf.ValueTypeNames.Token).Set("xform")
        attachment.CreateAttribute("physics:indices0", Sdf.ValueTypeNames.IntArray).Set([0])
        attachment.CreateAttribute("physics:coords1", Sdf.ValueTypeNames.Vector3fArray).Set([cfg.target_local_pos])
        attachment.CreateAttribute("physics:attachmentEnabled", Sdf.ValueTypeNames.Bool).Set(True)
    return cable_prim


@configclass
class AttachedCableCfg(sim_utils.CableCfg):
    """Cable spawner with one endpoint pinned to a sibling prim."""

    func: Callable[..., Usd.Prim] = spawn_attached_cable
    target_prim_name: str = "Object"
    target_local_pos: tuple[float, float, float] = (0.0, 0.0, _PEG_HALF_LENGTH)


@configclass
class CableInsertionSceneCfg(ObjectTableSceneCfg):
    """Peg-in-hole scene with a cable extending from the peg's long axis."""

    cable: CableObjectCfg = CableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cable",
        spawn=AttachedCableCfg(
            positions=_cable_rest_positions(),
            physics_material=sim_utils.CableMaterialCfg(),
            collision_props=[sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)],
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.85)),
        ),
        init_state=CableObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.04 + _PEG_HALF_LENGTH)),
    )


@configclass
class CableInsertionEventCfg(PegInHoleEventCfg):
    """Peg-in-hole events with a cable reset aligned to the peg."""

    reset_cable = EventTerm(
        func="isaaclab_tasks.core.cable_insertion.mdp.events:reset_cable_from_object",
        mode="reset",
        params={
            "object_cfg": SceneEntityCfg("object"),
            "cable_cfg": SceneEntityCfg("cable"),
        },
    )
