# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files import spawn_from_usd

if TYPE_CHECKING:
    from pxr import Usd


def spawn_four_bar_robot(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn Newton's four-bar USD without its packaged static plane.

    Args:
        prim_path: Prim path where the linkage should be spawned.
        cfg: USD spawn configuration.
        translation: Root translation [m].
        orientation: Root orientation quaternion ``(x, y, z, w)``.
        **kwargs: Additional keyword arguments passed through to the USD spawner.

    Returns:
        The source prim spawned by the USD spawner.
    """
    prim = spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    static_geometry_prim = sim_utils.get_current_stage().GetPrimAtPath(f"{prim.GetPath()}/StaticGeometry")
    if static_geometry_prim.IsValid():
        static_geometry_prim.SetActive(False)
    return prim
