# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from math import pi
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .cable_shape_commands import CableShapeCommand


@configclass
class CableShapeCommandCfg(CommandTermCfg):
    """Configuration for a planar whole-cable position command."""

    class_type: type["CableShapeCommand"] | str = "{DIR}.cable_shape_commands:CableShapeCommand"

    asset_name: str | None = MISSING
    """Name of the asset whose root frame defines the command. None uses the environment frame."""

    object_name: str = MISSING
    """Name of the cable asset."""

    @configclass
    class Ranges:
        """Sampling ranges for the first target and initial heading."""

        pos_x: tuple[float, float] = MISSING
        """First target x-position range [m]."""

        pos_y: tuple[float, float] = MISSING
        """First target y-position range [m]."""

        heading: tuple[float, float] = (-pi, pi)
        """Initial heading range [rad]."""

    ranges: Ranges = MISSING
    """Sampling ranges for the command."""

    segment_length: float = MISSING
    """Distance between consecutive cable segment targets [m]."""

    target_z: float = MISSING
    """Target height in the command frame [m]."""

    max_turn_angle: float = pi / 4
    """Maximum signed heading change between consecutive segments [rad]."""

    target_xy_bounds: tuple[tuple[float, float], tuple[float, float]] = ((0.0, 1.0), (-0.5, 0.5))
    """Allowed target x- and y-coordinate bounds in the command frame [m]."""

    max_sampling_attempts: int = 512
    """Maximum rejection-sampling attempts per command."""

    success_vis_asset_name: str = MISSING
    """Name of the asset used to visualize task success."""

    success_threshold: float = MISSING
    """Maximum segment position error for success [m]."""

    success_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/SuccessMarkers", markers={}
    )
    """Success visualization configuration."""

    target_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/cable_shape_target",
        markers={
            "target": sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            )
        },
    )
    """Target point visualization configuration."""

    current_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/cable_shape_current",
        markers={
            "current": sim_utils.SphereCfg(
                radius=0.008,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            )
        },
    )
    """Current cable point visualization configuration."""
