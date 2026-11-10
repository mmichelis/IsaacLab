# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBaseCfg, SceneEntityCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab_tasks.contrib.nist.utils.collision_analyzer import CollisionAnalyzer


@configclass
class CollisionAnalyzerCfg(ManagerTermBaseCfg):
    """Configuration for :class:`CollisionAnalyzer`.

    Rigid assets use sampled collision-surface points; deformable assets use all
    current world-space nodes, ignoring :attr:`num_points` and body selections.
    Every point must clear the configured obstacle meshes by :attr:`min_dist`.
    This sampled geometric predicate can miss intersections between points; it
    does not check native contacts, self-collision, or unlisted obstacles.
    Obstacles may be rigid, articulated, or static scene assets (one root per
    environment), but not deformable. Obstacle mesh geometry and scale are fixed
    at construction; root and articulated body poses are refreshed on each call.
    """

    func: type[CollisionAnalyzer] | str = "{DIR}.collision_analyzer:CollisionAnalyzer"
    """The callable class to instantiate from this configuration."""

    num_points: int = 32
    """Number of surface points sampled per rigid body of the asset."""

    max_dist: float = 0.5
    """Upper clamp for signed-distance queries [m]. Points farther than this
    from any obstacle surface are clamped to this value."""

    min_dist: float = 0.0
    """Minimum clearance threshold [m]. An environment is collision-free only
    when every sampled point has a signed distance >= this value."""

    asset_cfg: SceneEntityCfg = MISSING
    """Scene entity whose bodies are sampled for collision checking."""

    obstacle_cfgs: list[SceneEntityCfg] = MISSING
    """Scene entities treated as obstacles to check against."""
