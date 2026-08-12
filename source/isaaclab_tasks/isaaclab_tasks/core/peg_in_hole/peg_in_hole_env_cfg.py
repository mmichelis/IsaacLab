# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_visualizers.newton import NewtonVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg
from isaaclab_contrib.deformable.newton_manager_cfg import VBDSolverCfg

from isaaclab_tasks.core.peg_in_hole import mdp
from isaaclab_tasks.utils import PresetCfg

TABLE_SPAWN_CFG = sim_utils.CuboidCfg(
    size=(1.3, 0.9, 1.05),
    collision_props=sim_utils.CollisionPropertiesCfg(),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    # trick: we let visualizer's color to show the table with success coloring
    visible=False,
)

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for a peg-in-hole scene with a robot and an object."""

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # target object
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[0, 0, 0, 1]),
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                    scale=(scale, scale, scale),
                )
                for scale in (0.64, 0.72, 0.80, 0.88, 0.96)
            ],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    # static table collider with its top surface at z = 0. Kept invisible: the success
    # visualizer renders the visible table, colored by whether the goal is reached
    # (see CommandsCfg).
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, -0.525]),
        spawn=TABLE_SPAWN_CFG,
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.ObjectUniformPoseCommandCfg(
        asset_name="robot",
        object_name="object",
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.ObjectUniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
        success_vis_asset_name="table",
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
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: (
        mdp.JointPositionActionCfg | mdp.RelativeJointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg
    ) = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg | mdp.JointPositionToLimitsActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class ProprioObsCfg(ObsGroup):
        """Observations for proprioception group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PerceptionObsCfg(ObsGroup):
        """Observations for perception group."""

        object_point_cloud = ObsTerm(
            func=mdp.object_point_cloud_b,
            # clip=(-2.0, 2.0),  # clamp between -2 m to 2 m
            params={"num_points": 32, "flatten": True},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_dim = 0
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    proprio: ProprioObsCfg = ProprioObsCfg()
    perception: PerceptionObsCfg = PerceptionObsCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    variable_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="reset",
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            "operation": "abs",
        },
    )

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": [0.5, 1.0],
            "dynamic_friction_range": [0.5, 1.0],
            "restitution_range": [0.0, 0.0],
            "num_buckets": 250,
        },
    )

    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "static_friction_range": [0.5, 1.0],
            "dynamic_friction_range": [0.5, 1.0],
            "restitution_range": [0.0, 0.0],
            "num_buckets": 250,
        },
    )

    object_physics_inertia = EventTerm(
        func=mdp.randomize_rigid_body_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "inertia_distribution_params": [0.9, 1.1],
            "operation": "scale",
            "diagonal_only": True,
        },
    )

    joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": [0.8, 1.2],
            "damping_distribution_params": [0.8, 1.2],
            "operation": "scale",
        },
    )

    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": [0.5, 2.0],
            "operation": "scale",
        },
    )

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.2, "object_cfg": SceneEntityCfg("object", body_names="Object")},
        weight=1.0,
    )

    # Dense lift-off shaping: bridges the flat region between reaching and goal tracking.
    lifting_object = RewTerm(
        func=mdp.object_lifting,
        params={
            "std": 0.05,
            "minimal_height": 0.055,
            "object_cfg": SceneEntityCfg("object", body_names="Object"),
        },
        weight=5.0,
    )

    object_goal_tracking_delta = RewTerm(
        func=mdp.object_goal_distance_delta,
        params={
            "minimal_height": 0.0,
            "command_name": "object_pose",
            "success_threshold": 0.05,
            "object_cfg": SceneEntityCfg("object", body_names="Object"),
        },
        weight=500.0,
    )

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.3,
            "minimal_height": 0.0,
            "command_name": "object_pose",
            "success_threshold": 0.05,
            "object_cfg": SceneEntityCfg("object", body_names="Object"),
        },
        weight=3.0,
    )

    success_bonus = RewTerm(
        func=mdp.object_goal_reached,
        params={
            "minimal_height": 0.0,
            "command_name": "object_pose",
            "success_threshold": 0.05,
            "object_cfg": SceneEntityCfg("object", body_names="Object"),
        },
        weight=50.0,
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_out_of_bounds = DoneTerm(
        func=mdp.object_outside_bounds,
        params={
            "x_bounds": (0.0, 1.0),
            "y_bounds": (-0.5, 0.5),
            "z_bounds": (-0.05, 1.0),
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    joint_vel_out_of_limit = DoneTerm(
        func=mdp.joint_vel_out_of_sim_limit,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-2, "num_steps": 50000}
    )

    # Since we use 24 steps per env, 10000 steps correspond to 10000/24 = 416.67 learning iterations
    gravity = CurrTerm(
        func=mdp.gravity_range_linear,
        params={
            "event_name": "variable_gravity",
            "start_gravity_z": -0.0001,
            "end_gravity_z": -9.81,
            "start_step": 0,
            "end_step": 20000,
        },
    )


##
# Environment configuration
##


@configclass
class PegInHolePhysicsCfg(PresetCfg):
    isaacsim_physx: PhysxCfg = PhysxCfg(
        bounce_threshold_velocity=0.01,
        gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
        gpu_total_aggregate_pairs_capacity=16 * 1024,
        friction_correlation_distance=0.00625,
    )

    physx: PhysxAutoCfg = PhysxAutoCfg(isaacsim_physx=isaacsim_physx)

    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            cone="pyramidal",
            integrator="implicitfast",
            use_mujoco_contacts=False,
        ),
        num_substeps=2,
        collision_decimation=1,
    )

    newton_mjwarp_vbd_proxy: NewtonCfg = NewtonCfg(
        solver_cfg=CouplerProxyCfg(
            entries=[
                CouplerEntryCfg(
                    name="rigid",
                    solver_cfg=MJWarpSolverCfg(cone="elliptic", ls_iterations=20, integrator="implicitfast"),
                    bodies=[r"/World/envs/env_.*/Robot", r"/World/envs/env_.*/Table"],
                ),
                CouplerEntryCfg(
                    name="object",
                    solver_cfg=VBDSolverCfg(iterations=10),
                    bodies=[r"/World/envs/env_.*/Object"],
                    include_static_shapes=True,
                ),
            ],
            proxies=[
                CouplerProxyMappingCfg(
                    source="rigid",
                    destination="object",
                    bodies=[
                        r"/World/envs/env_.*/Robot/panda_hand",
                        r"/World/envs/env_.*/Robot/panda_(left|right)finger",
                        r"/World/envs/env_.*/Table",
                    ],
                )
            ],
            iterations=1,
        ),
        num_substeps=2,
    )

    default = newton_mjwarp


@configclass
class PegInHoleEnvCfg(ManagerBasedRLEnvCfg):
    """Initial peg-in-hole configuration copied from the lift task."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=8192, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 1.0 / 120
        self.sim.render_interval = self.decimation

        self.sim.physics = PegInHolePhysicsCfg()

        # Camera pose (lowered eye framing the tabletop workspace).
        eye, lookat = (2.75, -2.0, 1.0), (0.5, 0.0, -0.5)
        self.viewer.eye = eye
        self.viewer.lookat = lookat
        self.sim.visualizer_cfgs = [NewtonVisualizerCfg(eye=eye, lookat=lookat, window_width=1920, window_height=1080)]
