# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka deformable (soft beam) lifting environment."""

from __future__ import annotations

from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
    NewtonShapeSDFCfg,
)
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import NewtonDeformableBodyMaterialCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxDeformableBodyPropertiesCfg
from isaaclab_physx.sim.spawners.materials import PhysxDeformableBodyMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.assets.deformable_object import DeformableObjectCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_contrib.coupling import (
    CouplerEntryCfg,
    CouplerProxyCfg,
    CouplerProxyMappingCfg,
)
from isaaclab_contrib.deformable.newton_manager_cfg import (
    CoupledMJWarpVBDSolverCfg,
    NewtonModelCfg,
    VBDSolverCfg,
)

from isaaclab_tasks.utils import PresetCfg

from . import mdp

##
# Pre-defined configs
##

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort:skip


##
# Helpers
##


# Shared volume material parameters. The Newton config below uses the equivalent Lame parameters.
YOUNGS_MODULUS = 5e5
POISSONS_RATIO = 0.4

# Table collider whose top surface sits at z = 0. Spawned invisible: the command term's success
# visualizer draws it instead, tinted by whether the goal is reached.
TABLE_SPAWN_CFG = sim_utils.CuboidCfg(
    size=(1.3, 0.9, 1.05),
    collision_props=sim_utils.CollisionPropertiesCfg(),
    visible=False,
)


@configclass
class DeformableCfg(PresetCfg):
    """Preset config for the deformable object, matching the Newton example."""

    newton_mjwarp_vbd: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Deformable",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.05)),
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.3, 0.04, 0.04),
            deformable_props=NewtonDeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.85)),
            physics_material=NewtonDeformableBodyMaterialCfg(
                density=1000.0,
                k_mu=YOUNGS_MODULUS / (2.0 * (1.0 + POISSONS_RATIO)),
                k_lambda=(YOUNGS_MODULUS * POISSONS_RATIO / ((1.0 + POISSONS_RATIO) * (1.0 - 2.0 * POISSONS_RATIO))),
                particle_radius=0.0025,
            ),
        ),
    )

    isaacsim_physx: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Deformable",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.05)),
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.3, 0.04, 0.04),
            deformable_props=PhysxDeformableBodyPropertiesCfg(
                rest_offset=0.0005, contact_offset=0.005, solver_position_iteration_count=32
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.85)),
            physics_material=PhysxDeformableBodyMaterialCfg(
                density=1000.0,
                youngs_modulus=YOUNGS_MODULUS,
                poissons_ratio=POISSONS_RATIO,
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        ),
    )

    newton_mjwarp_vbd_proxy = newton_mjwarp_vbd

    default = newton_mjwarp_vbd_proxy


@configclass
class PhysicsCfg(PresetCfg):
    newton_mjwarp_vbd: NewtonCfg = NewtonCfg(
        solver_cfg=CoupledMJWarpVBDSolverCfg(
            rigid_solver_cfg=MJWarpSolverCfg(
                njmax=40,
                nconmax=20,
                ls_iterations=20,
                cone="pyramidal",
                impratio=1,
                integrator="implicitfast",
                ccd_iterations=100,
            ),
            soft_solver_cfg=VBDSolverCfg(
                iterations=10,
                integrate_with_external_rigid_solver=True,
                particle_enable_self_contact=False,
                particle_collision_detection_interval=-1,
            ),
            coupling_mode="two_way",
            model_cfg=NewtonModelCfg(
                soft_contact_ke=1e4,
                soft_contact_kd=1e-5,
                soft_contact_mu=5.0,
            ),
        ),
        default_shape_cfg=NewtonShapeCfg(ke=4e4, kd=1e-5, mu=5.0),
        num_substeps=10,
    )

    newton_mjwarp_vbd_proxy: NewtonCfg = NewtonCfg(
        solver_cfg=CouplerProxyCfg(
            entries=[
                CouplerEntryCfg(
                    name="rigid",
                    solver_cfg=MJWarpSolverCfg(
                        cone="elliptic",
                        ls_iterations=20,
                        integrator="implicitfast",
                    ),
                    bodies=[r"/World/envs/env_.*/Robot"],
                ),
                CouplerEntryCfg(
                    name="soft",
                    solver_cfg=VBDSolverCfg(iterations=10, rigid_body_particle_contact_buffer_size=256),
                    all_particles=True,
                    include_static_shapes=True,
                ),
            ],
            proxies=[
                CouplerProxyMappingCfg(
                    source="rigid",
                    destination="soft",
                    bodies=[
                        r"/World/envs/env_.*/Robot/panda_hand",
                        r"/World/envs/env_.*/Robot/panda_(left|right)finger",
                    ],
                    # detect finger/beam contact every substep so the gripper stops at the surface
                    collide_interval=1,
                    # Watertight rigid-soft contact: generate soft contacts over the full beam
                    # surface (edges + triangle interiors) against the gripper SDFs, not just at
                    # beam vertices. Requires the gripper SDFs provisioned via sdf_shape_cfgs below.
                    # NOTE: functional harvesting of these full-surface contacts through the proxy
                    # coupler depends on the parallel Newton-core proxy-harvest generalization draft;
                    # not runtime-verified here.
                    # The auto-estimate is shape_count * particle_count, which on the refined mesh asks
                    # for ~294M contacts (3.5 GB for a single array). Measured peak is ~1.3k records
                    # per env, so 6k per env leaves ample headroom. Newton warns on overflow.
                    collision_pipeline=NewtonCollisionPipelineCfg(
                        enable_rigid_soft_full_surface_contact=True,
                        # Sized for the 2048-env runs: measured peak is ~1.3k records per env, so 6k
                        # leaves ample headroom and Newton warns on overflow. The auto-estimate
                        # (shape_count * particle_count) asks for ~294M on the refined mesh, which
                        # is 3.5 GB for a single array. Raise this proportionally for more envs.
                        soft_contact_max=6_000 * 2048,
                    ),
                )
            ],
            iterations=1,
            model_cfg=NewtonModelCfg(
                soft_contact_ke=1.0e3,
                soft_contact_kd=1.0e-2,
                soft_contact_mu=1.0,
            ),
        ),
        sdf_shape_cfgs=[
            NewtonShapeSDFCfg(
                shape_label_patterns=[
                    r"/World/envs/env_.*/Robot/panda_hand/collisions/collisions",
                    r"/World/envs/env_.*/Robot/panda_(left|right)finger/collisions/collisions",
                ],
                # ~2.3 mm voxels on the fingers; finer isn't needed for contact resolution.
                max_resolution=8,
            )
        ],
        num_substeps=2,
    )

    isaacsim_physx: PhysxCfg = PhysxCfg()

    default = newton_mjwarp_vbd_proxy


##
# Scene definition
##


@configclass
class _FrankaSoftSceneCfg(InteractiveSceneCfg):
    """Scene for the Franka deformable environment."""

    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # end-effector frame for reward shaping
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
            ),
        ],
    )

    deformable: DeformableCfg = DeformableCfg()

    # static table collider with its top surface at z = 0. Kept invisible: the success
    # visualizer renders the visible table, colored by whether the goal is reached
    # (see CommandsCfg).
    table: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, -0.525]),
        spawn=TABLE_SPAWN_CFG,
    )

    # ground plane
    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    def __post_init__(self) -> None:
        # Re-tuned Franka actuators: stiff arm gains with realistic armature so the low-inertia
        # default gains do not let the fingers tunnel through the soft body, and a slower, weaker
        # gripper so it settles on the beam surface instead of crushing it. Velocity limits are
        # required by the joint_vel_out_of_sim_limit termination. Scoped here rather than in
        # FRANKA_PANDA_CFG so the other Franka tasks keep the stock asset.
        shoulder = self.robot.actuators["panda_shoulder"]
        shoulder.velocity_limit_sim = 2.175
        shoulder.stiffness = 600.0
        shoulder.damping = 50.0
        shoulder.armature = {"panda_joint[1-2]": 0.6057, "panda_joint[3-4]": 0.4625}

        forearm = self.robot.actuators["panda_forearm"]
        forearm.velocity_limit_sim = 2.61
        forearm.stiffness = {"panda_joint5": 250.0, "panda_joint6": 150.0, "panda_joint7": 50.0}
        forearm.damping = {"panda_joint5": 30.0, "panda_joint6": 25.0, "panda_joint7": 15.0}
        forearm.armature = 0.2055

        hand = self.robot.actuators["panda_hand"]
        hand.effort_limit_sim = 70.0
        hand.velocity_limit_sim = 0.2
        hand.stiffness = 750.0
        hand.damping = 175.0
        hand.armature = 0.1


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Commands for the deformable goal pose (xyz + identity quat in robot root frame)."""

    deformable_pose = mdp.DeformableUniformPoseCommandCfg(
        asset_name="robot",
        object_name="deformable",
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.DeformableUniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6),
            pos_y=(-0.25, 0.25),
            pos_z=(0.25, 0.5),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
        # the invisible table is drawn by these markers, tinted green once the goal is reached
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
class _JointActionsCfg:
    """7-dim relative joint-position arm targets + 1-dim limit-rescaled gripper."""

    arm_action = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=["panda_joint.*"], scale=0.03)

    gripper_action = mdp.JointPositionToLimitsActionCfg(
        asset_name="robot", joint_names=["panda_finger.*"], rescale_to_limits=True
    )


@configclass
class _IkActionsCfg:
    """7-dim absolute end-effector pose (xyz + quaternion) via differential IK + 1-dim binary gripper."""

    arm_action = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.6},
        ),
        body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
    )

    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.05},
        close_command_expr={"panda_finger_.*": 0.015},
    )


@configclass
class ActionsCfg(PresetCfg):
    """Action-space presets: joint-space for RL, task-space IK for scripted end-effector control."""

    joint: _JointActionsCfg = _JointActionsCfg()

    ik: _IkActionsCfg = _IkActionsCfg()

    default = joint


@configclass
class ObservationsCfg:
    """Policy observations: joint state, deformable COM in robot frame, target, last action."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        deformable_sampled_points = ObsTerm(
            func=mdp.DeformableSampledPointsInRobotRootFrame,
            params={"asset_cfg": SceneEntityCfg("deformable"), "num_points": 20},
        )
        target_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "deformable_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset events: robot to default joint config, deformable with small position randomization."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.9, 1.1), "velocity_range": (0.0, 0.0)},
    )

    reset_deformable = EventTerm(
        func=mdp.reset_nodal_state_uniform,
        mode="reset",
        params={
            "position_range": {"x": (-0.15, 0.1), "y": (-0.2, 0.2), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("deformable"),
        },
    )


@configclass
class RewardsCfg:
    """Deformable analogue of the winning rigid-cube lift recipe.

    The dense lift reward is ungated (``minimal_height`` 0.0) so it is graded across the FULL COM
    height range: pressing the soft beam into the table (which drops its COM below rest) is
    penalized and raising it is rewarded, giving a continuous escape gradient from the
    press-into-table local optimum. Goal-tracking gates sit just above rest (0.06 m) so those
    terms engage only after a genuine lift. Success = COM within 5 cm of the goal position.
    """

    # The hand targets the COM so the grasp lands mid-beam (targeting the nearest node lets the
    # gripper grab an end, which turns the beam into an unstable pole). The fingers target the
    # nearest surface node instead: targeting the COM with both fingers is maximized only when
    # both fingertips reach the object's center, i.e. by closing and indenting the beam.
    reaching_deformable = RewTerm(
        func=mdp.deformable_com_ee_distance,
        params={"std": 0.1, "asset_cfg": SceneEntityCfg("deformable")},
        weight=5.0,
    )

    # grasping_deformable = RewTerm(
    #     func=mdp.deformable_fingertip_distance,
    #     params={
    #         "std": 0.1,
    #         "asset_cfg": SceneEntityCfg("deformable"),
    #         "robot_cfg": SceneEntityCfg("robot", body_names=["panda_leftfinger", "panda_rightfinger"]),
    #         "target_com": False,
    #     },
    #     weight=2.0,
    # )

    lifting_deformable = RewTerm(
        func=mdp.deformable_lifting,
        params={"std": 0.1, "minimal_height": 0.02, "asset_cfg": SceneEntityCfg("deformable")},
        weight=5.0,
    )

    deformable_goal_tracking_delta = RewTerm(
        func=mdp.deformable_com_goal_distance_delta,
        params={
            "minimal_height": 0.0,
            "command_name": "deformable_pose",
            "asset_cfg": SceneEntityCfg("deformable"),
        },
        weight=500.0,
    )

    deformable_goal_tracking = RewTerm(
        func=mdp.deformable_com_goal_distance,
        params={
            "std": 0.3,
            "minimal_height": 0.0,
            "command_name": "deformable_pose",
            "success_threshold": 0.05,
            "asset_cfg": SceneEntityCfg("deformable"),
        },
        weight=2.0,
    )

    success_bonus = RewTerm(
        func=mdp.deformable_com_goal_reached,
        params={
            "minimal_height": 0.0,
            "command_name": "deformable_pose",
            "success_threshold": 0.05,
            "asset_cfg": SceneEntityCfg("deformable"),
        },
        weight=10.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)


@configclass
class CurriculumCfg:
    """Ramp the action-rate penalty once the policy has learned to lift (matches rigid recipe)."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-2, "num_steps": 40000}
    )

    # Since we use 24 steps per env, 10000 steps correspond to 10000/24 = 416.67 learning iterations
    gravity = CurrTerm(
        func=mdp.modify_gravity_linear,
        params={"start_gravity_z": -0.0001, "end_gravity_z": -9.81, "start_step": 0, "end_step": 10000},
    )


@configclass
class SoftEventCfg(EventCfg):
    """Parent reset events plus per-env randomization of the beam's Young's modulus and density."""

    randomize_deformable_material = EventTerm(
        func=mdp.randomize_deformable_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("deformable"),
            "youngs_modulus_range": (5e5, 1e6),
            "density_range": (900.0, 1000.0),
            "poissons_ratio": 0.3,
        },
    )


@configclass
class TerminationsCfg:
    """Time out + table bounds/drop termination."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    deformable_outside_table = DoneTerm(
        func=mdp.deformable_outside_table_bounds,
        params={
            "x_bounds": (0.0, 1.0),
            "y_bounds": (-0.5, 0.5),
            "asset_cfg": SceneEntityCfg("deformable"),
        },
    )

    deformable_dropped = DoneTerm(
        func=mdp.deformable_com_below_minimum,
        params={"minimum_height": -0.1, "asset_cfg": SceneEntityCfg("deformable")},
    )

    ee_below_table = DoneTerm(
        func=mdp.ee_below_minimum,
        params={"minimum_height": 0.0, "ee_frame_cfg": SceneEntityCfg("ee_frame")},
    )

    joint_vel_out_of_limit = DoneTerm(
        func=mdp.joint_vel_out_of_sim_limit,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # deformable_vel_out_of_limit = DoneTerm(
    #     func=mdp.deformable_nodal_vel_above_maximum,
    #     params={"maximum_velocity": 1.0, "asset_cfg": SceneEntityCfg("deformable")},
    # )

    # real failure, not a time out: a diverged solve must bootstrap as a termination
    deformable_invalid = DoneTerm(
        func=mdp.deformable_state_invalid,
        params={"asset_cfg": SceneEntityCfg("deformable")},
    )

    # The measured divergence poisons the beam too, so deformable_invalid resets that case. This
    # covers a robot-only divergence: every reward term is deformable-driven and sanitized, and the
    # other robot terminations fail open on NaN, so nothing else would ever reset the environment.
    robot_invalid = DoneTerm(
        func=mdp.robot_state_invalid,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


##
# Environment configuration
##


@configclass
class FrankaSoftSceneCfg(PresetCfg):
    newton_mjwarp_vbd: _FrankaSoftSceneCfg = _FrankaSoftSceneCfg(num_envs=2048, env_spacing=2.0, replicate_physics=True)

    # PhysX does not support replicating physics for deformable objects
    isaacsim_physx: _FrankaSoftSceneCfg = _FrankaSoftSceneCfg(num_envs=2048, env_spacing=2.0, replicate_physics=False)

    newton_mjwarp_vbd_proxy = newton_mjwarp_vbd

    default = newton_mjwarp_vbd_proxy


@configclass
class FrankaSoftEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based RL environment: Franka Panda lifting a soft beam to a target pose."""

    # Scene settings
    scene: FrankaSoftSceneCfg = FrankaSoftSceneCfg()
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # Parent reset events + per-env material domain randomization.
    events: EventCfg = EventCfg()
    # Ramp the action-rate penalty once the policy has learned to lift.
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        # general settings
        self.decimation = 4
        self.episode_length_s = 5.0

        # simulation settings
        self.sim.dt = 1.0 / 120
        self.sim.render_interval = self.decimation
        self.sim.gravity = (0.0, 0.0, -9.81)
        self.sim.physics = PhysicsCfg()

        # Camera for --video / viewer: lower and closer, framed on the table/lift zone.
        # The ViewerCfg default (7.5, 7.5, 7.5) -> (0, 0, 0) sits far too high above the action;
        # the video recorder copies these into cfg.video_recorder (manager_based_rl_env).
        self.viewer.eye = (0.5, 0.5, 0.6)
        self.viewer.lookat = (0.0, 1.0, 0.35)
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=self.viewer.eye, lookat=self.viewer.lookat)
