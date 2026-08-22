# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka cable shaping environment."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils.configclass import configclass

from ... import mdp
from .franka_cable_env_cfg import FrankaCableEnvCfg
from .franka_soft_env_cfg import TABLE_SPAWN_CFG
from .franka_soft_env_cfg import CurriculumCfg as FrankaSoftCurriculumCfg

_CABLE_SHAPE_SUCCESS_THRESHOLD = 0.03


@configclass
class CommandsCfg:
    """Planar target shape for all cable segments."""

    cable_shape = mdp.CableShapeCommandCfg(
        asset_name="robot",
        object_name="cable",
        ranges=mdp.CableShapeCommandCfg.Ranges(
            pos_x=(0.35, 0.65),
            pos_y=(-0.1, 0.1),
            heading=(-math.pi, math.pi),
        ),
        segment_length=0.03,
        target_z=0.011,
        max_turn_angle=math.pi / 4,
        target_xy_bounds=((0.0, 1.0), (-0.5, 0.5)),
        max_sampling_attempts=512,
        success_vis_asset_name="table",
        success_threshold=_CABLE_SHAPE_SUCCESS_THRESHOLD,
        success_visualizer_cfg=VisualizationMarkersCfg(
            prim_path="/Visuals/SuccessMarkers",
            markers={
                "failure": TABLE_SPAWN_CFG.replace(
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.5, 0.5)), visible=True
                ),
                "success": TABLE_SPAWN_CFG.replace(
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.8, 0.5)), visible=True
                ),
            },
        ),
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
    )


@configclass
class ObservationsCfg:
    """Policy observations for cable shaping."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        cable_segment_positions = ObsTerm(
            func=mdp.cable_segment_positions_in_robot_root_frame,
            params={"asset_cfg": SceneEntityCfg("cable")},
        )
        target_shape = ObsTerm(func=mdp.generated_commands, params={"command_name": "cable_shape"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Cable shaping rewards."""

    reaching_cable = RewTerm(
        func=mdp.cable_ee_distance,
        params={"std": 0.3, "asset_cfg": SceneEntityCfg("cable")},
        weight=1.0,
    )

    cable_shape_tracking = RewTerm(
        func=mdp.CableShapeGoalDistance,
        params={
            "std": 0.1,
            "command_name": "cable_shape",
            "success_threshold": _CABLE_SHAPE_SUCCESS_THRESHOLD,
            "asset_cfg": SceneEntityCfg("cable"),
        },
        weight=5.0,
    )

    success_bonus = RewTerm(
        func=mdp.cable_shape_goal_reached,
        params={
            "command_name": "cable_shape",
            "success_threshold": _CABLE_SHAPE_SUCCESS_THRESHOLD,
            "asset_cfg": SceneEntityCfg("cable"),
        },
        weight=20.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-3)


@configclass
class CurriculumCfg(FrankaSoftCurriculumCfg):
    """Action-rate curriculum with fixed normal gravity."""

    gravity: None = None

    def __post_init__(self) -> None:
        self.action_rate.params["num_steps"] = 60_000


@configclass
class FrankaCableShapeEnvCfg(FrankaCableEnvCfg):
    """Manager-based RL environment for shaping a 12-segment cable."""

    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
