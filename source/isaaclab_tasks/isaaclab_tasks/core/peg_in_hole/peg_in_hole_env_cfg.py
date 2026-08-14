# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.sim.spawners.materials import NewtonMaterialCfg
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
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg
from isaaclab_contrib.deformable.newton_manager_cfg import VBDSolverCfg

from isaaclab_tasks.core.peg_in_hole import mdp
from isaaclab_tasks.utils import PresetCfg

TABLE_SPAWN_CFG = sim_utils.CuboidCfg(
    size=(1.3, 0.9, 1.05),
    collision_props=sim_utils.CollisionPropertiesCfg(),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
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
    # task object
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.0, 0.03]),
        spawn=sim_utils.CuboidCfg(
            size=(0.02, 0.02, 0.05),
            physics_material=[
                UsdPhysicsRigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
                NewtonMaterialCfg(contact_stiffness=1.0e4, contact_damping=100.0),
            ],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        ),
    )

    # visual target
    target: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.0, 0.0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.02, 0.02, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), opacity=0.35),
        ),
    )

    # static table collider with its top surface at z = 0
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
class ActionsCfg:
    pass


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        target_object_pose = ObsTerm(
            func=mdp.asset_pose_b,
            params={"asset_cfg": SceneEntityCfg("target"), "reference_cfg": SceneEntityCfg("robot")},
        )

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

    conditional_reset = EventTerm(
        func="isaaclab_tasks.core.lift.mdp.events:conditional_reset",
        mode="reset",
        params={
            "terms": {
                "reset_target_position": EventTerm(
                    func=mdp.reset_root_state_uniform,
                    mode="reset",
                    params={
                        "pose_range": {
                            "x": (-0.1, 0.1),
                            "y": (-0.25, 0.25),
                            "z": (0.08, 0.5),
                            "yaw": (-0.5, 0.5),
                        },
                        "velocity_range": {},
                        "asset_cfg": SceneEntityCfg("target"),
                    },
                ),
                "reset_robot_arm_joints": EventTerm(
                    func=mdp.reset_joints_by_offset,
                    mode="reset",
                    params={
                        "position_range": (-0.2, 0.2),
                        "velocity_range": (0.0, 0.0),
                        "asset_cfg": SceneEntityCfg("robot", joint_names="panda_joint.*"),
                    },
                ),
                "reset_robot_gripper_joints": EventTerm(
                    func=mdp.reset_joints_shared_offset,
                    mode="reset",
                    params={
                        "position_range": (-0.01, 0.0),
                        "asset_cfg": SceneEntityCfg("robot", joint_names="panda_finger_joint.*"),
                    },
                ),
                "reset_object_position": EventTerm(
                    func=mdp.reset_root_state_uniform,
                    mode="reset",
                    params={
                        "pose_range": {"x": (-0.25, 0.25), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
                        "velocity_range": {},
                        "asset_cfg": SceneEntityCfg("object", body_names="Object"),
                    },
                ),
            },
            "buffer_size_per_group": 32768,
            "oversample_factor": 2.0,
            "diversity_feature": mdp.GraspTravelDistanceCfg(
                asset_name="robot",
                body_names="panda_hand",
                object_name="object",
                target_name="target",
                log_scale=True,
            ),
            "valid_criteria": {
                "object_robot_clearance": mdp.MeshClearanceCfg(
                    asset_name="robot",
                    body_names=".*",
                    object_name="object",
                    num_object_points=32,
                    min_clearance=0.01,
                ),
                "robot_table_clearance": mdp.SlabClearanceCfg(
                    asset_name="robot",
                    body_names=["panda_link[1-7]", "panda_hand", ".*finger"],
                    object_name="object",
                    obstacle_slabs=[((-0.15, 1.15), (-0.45, 0.45), 0.0)],
                    num_object_points=32,
                    min_clearance=0.001,
                ),
            },
            "success_monitor": mdp.SuccessMonitorCfg(target_success_rate=0.5),
        },
    )

    variable_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="reset",
        params={
            "gravity_distribution_params": ([0.0, 0.0, -9.81], [0.0, 0.0, -9.81]),
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

    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": [0.5, 2.0],
            "operation": "scale",
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.2, "object_cfg": SceneEntityCfg("object", body_names="Object")},
        weight=5.0,
    )

    lifting_object = RewTerm(
        func=mdp.object_lifting,
        params={
            "std": 0.02,
            "minimal_height": 0.025,
            "object_cfg": SceneEntityCfg("object", body_names="Object"),
        },
        weight=10.0,
    )

    goal_distance = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.3,
            "minimal_height": 0.0,
            "object_cfg": SceneEntityCfg("object", body_names="Object"),
            "target_cfg": SceneEntityCfg("target"),
        },
        weight=2.0,
    )

    success = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.05,
            "minimal_height": 0.0,
            "success_threshold": 0.05,
            "object_cfg": SceneEntityCfg("object", body_names="Object"),
            "target_cfg": SceneEntityCfg("target"),
        },
        weight=10.0,
    )

    success_bonus = RewTerm(
        func=mdp.object_target_point_cloud_reached,
        params={
            "minimal_height": 0.0,
            "success_threshold": 0.05,
            "object_cfg": SceneEntityCfg("object", body_names="Object"),
            "target_cfg": SceneEntityCfg("target"),
            "success_vis_asset_name": "table",
            "success_visualizer_cfg": VisualizationMarkersCfg(
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
        },
        weight=10.0,
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-3)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_out_of_bounds = DoneTerm(
        func=mdp.object_outside_bounds,
        params={
            "x_bounds": (0.0, 1.0),
            "y_bounds": (-0.5, 0.5),
            "z_bounds": (-0.02, 1.0),
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
        func=mdp.reward_weight_linear,
        params={
            "term_name": "action_rate",
            "start_weight": -1e-3,
            "end_weight": -1e-1,
            "start_step": 30000,
            "end_step": 40000,
        },
    )

    adr = CurrTerm(
        func=mdp.DifficultyScheduler, params={"init_difficulty": 0, "min_difficulty": 0, "max_difficulty": 10}
    )

    gravity_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "events.variable_gravity.params.gravity_distribution_params",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {
                "initial_value": ((0.0, 0.0, -0.98), (0.0, 0.0, -0.98)),
                "final_value": ((0.0, 0.0, -9.81), (0.0, 0.0, -9.81)),
                "difficulty_term_str": "adr",
            },
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
                    bodies=[r"/World/envs/env_.*/Robot", r"/World/envs/env_.*/Target", r"/World/envs/env_.*/Table"],
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
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=8192, env_spacing=2.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 8.0
        # simulation settings
        self.sim.dt = 1.0 / 120
        self.sim.render_interval = self.decimation

        self.sim.physics = PegInHolePhysicsCfg()

        # Camera pose (lowered eye framing the tabletop workspace).
        eye, lookat = (2.75, -2.0, 1.0), (0.5, 0.0, -0.5)
        self.viewer.eye = eye
        self.viewer.lookat = lookat
        self.sim.visualizer_cfgs = [NewtonVisualizerCfg(eye=eye, lookat=lookat, window_width=1920, window_height=1080)]

    def play_mode(self):
        super().play_mode()
        reset_params = self.events.conditional_reset.params
        reset_params["buffer_size_per_group"] = 64
        reset_params["oversample_factor"] = 1.0
        reset_params["diversity_feature"] = None
        if self.curriculum is not None:
            self.curriculum.adr.params["init_difficulty"] = self.curriculum.adr.params["max_difficulty"]
            self.curriculum.adr.params["promotion_only"] = True
