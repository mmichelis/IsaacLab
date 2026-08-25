# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot-free cable shaping with direct segment forces."""

from __future__ import annotations

import math

from isaaclab_newton.physics import NewtonCfg, NewtonShapeCfg, VBDSolverCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, CableObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.core.lift import mdp as lift_mdp

from . import mdp

_CABLE_SEGMENT_COUNT = 12
_CABLE_SHAPE_SUCCESS_THRESHOLD = 0.03

_TABLE_SPAWN_CFG = sim_utils.CuboidCfg(
    size=(1.3, 0.9, 1.05),
    collision_props=sim_utils.CollisionPropertiesCfg(),
    visible=False,
)


@configclass
class CableShapeForceSceneCfg(InteractiveSceneCfg):
    """Cable, table, ground, and lighting."""

    cable: CableObjectCfg = CableObjectCfg(
        class_type=mdp.ForceControlledCableObject,
        prim_path="{ENV_REGEX_NS}/Cable",
        spawn=sim_utils.CableCfg(
            positions=[(0.03 * index, 0.0, 0.0) for index in range(_CABLE_SEGMENT_COUNT + 1)],
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.85)),
            physics_material=sim_utils.CableMaterialCfg(
                thickness=0.01,
                density=1000.0,
                stretch_stiffness=1.0e6,
                bend_stiffness=1.0e5,
            ),
            collision_props=[sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)],
        ),
        init_state=CableObjectCfg.InitialStateCfg(pos=(0.32, 0.0, 0.011)),
    )

    table: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, -0.525]),
        spawn=_TABLE_SPAWN_CFG.replace(
            physics_material=RigidBodyMaterialBaseCfg(static_friction=0.01, dynamic_friction=0.01),
        ),
    )

    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    sky_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class ActionsCfg:
    """World-frame force actions for every cable segment."""

    cable_force = mdp.CableForceActionCfg(asset_name="cable")


@configclass
class CommandsCfg:
    """Planar target shape in the environment frame."""

    cable_shape = lift_mdp.CableShapeCommandCfg(
        asset_name=None,
        object_name="cable",
        ranges=lift_mdp.CableShapeCommandCfg.Ranges(
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
                "failure": _TABLE_SPAWN_CFG.replace(
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.5, 0.5)), visible=True
                ),
                "success": _TABLE_SPAWN_CFG.replace(
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.8, 0.5)), visible=True
                ),
            },
        ),
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
    )


@configclass
class ObservationsCfg:
    """Cable state, target shape, and previous forces."""

    @configclass
    class PolicyCfg(ObsGroup):
        cable_segment_positions = ObsTerm(
            func=mdp.cable_segment_positions_in_env_frame,
            params={"asset_cfg": SceneEntityCfg("cable")},
        )
        cable_segment_velocities = ObsTerm(
            func=mdp.cable_segment_velocities,
            params={"asset_cfg": SceneEntityCfg("cable")},
        )
        target_shape = ObsTerm(func=lift_mdp.generated_commands, params={"command_name": "cable_shape"})
        actions = ObsTerm(func=lift_mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Cable shape tracking rewards."""

    cable_shape_tracking = RewTerm(
        func=lift_mdp.CableShapeGoalDistance,
        params={
            "std": 0.1,
            "command_name": "cable_shape",
            "success_threshold": _CABLE_SHAPE_SUCCESS_THRESHOLD,
            "robot_cfg": None,
            "asset_cfg": SceneEntityCfg("cable"),
        },
        weight=5.0,
    )

    success_bonus = RewTerm(
        func=lift_mdp.cable_shape_goal_reached,
        params={
            "command_name": "cable_shape",
            "success_threshold": _CABLE_SHAPE_SUCCESS_THRESHOLD,
            "robot_cfg": None,
            "asset_cfg": SceneEntityCfg("cable"),
        },
        weight=20.0,
    )

    action_rate = RewTerm(func=lift_mdp.action_rate_l2, weight=-1e-3)


@configclass
class EventCfg:
    """Cable reset randomization."""

    reset_cable = EventTerm(
        func=lift_mdp.reset_cable_state_uniform,
        mode="reset",
        params={
            "position_range": {"x": (-0.15, 0.1), "y": (-0.2, 0.2), "z": (0.0, 0.0)},
            "asset_cfg": SceneEntityCfg("cable"),
        },
    )


@configclass
class TerminationsCfg:
    """Time limit and cable workspace bounds."""

    time_out = DoneTerm(func=lift_mdp.time_out, time_out=True)

    cable_out_of_bounds = DoneTerm(
        func=lift_mdp.cable_outside_bounds,
        params={
            "x_bounds": (0.0, 1.0),
            "y_bounds": (-0.5, 0.5),
            "z_bounds": (-0.02, 1.0),
            "asset_cfg": SceneEntityCfg("cable"),
        },
    )


@configclass
class CableShapeForceEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based cable shaping with direct per-segment forces."""

    scene: CableShapeForceSceneCfg = CableShapeForceSceneCfg(
        num_envs=8192, env_spacing=2.0, replicate_physics=True
    )
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 5.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physics = NewtonCfg(
            solver_cfg=VBDSolverCfg(iterations=10),
            default_shape_cfg=NewtonShapeCfg(ke=2.5e3, kd=100.0, mu=10.0),
            num_substeps=4,
        )
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(0.75, 0.25, 0.65), lookat=(0.5, 0.0, 0.0))
