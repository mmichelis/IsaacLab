# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka deformable-duck lifting environment.

The scene mirrors ``newton/examples/softbody/example_softbody_franka.py``:
a Franka Panda manipulator on a tabletop with a tetrahedral rubber duck simulated
by VBD. The RL task is to lift the duck's centre of mass to a randomised target
position sampled in the robot's root frame.
"""

from __future__ import annotations

import os

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.assets.deformable_object import DeformableObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_contrib.deformable.newton_manager_cfg import CoupledMJWarpVBDSolverCfg, NewtonModelCfg, VBDSolverCfg

from . import mdp

##
# Pre-defined configs
##

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort:skip


##
# Helpers
##

# Local copy of the Newton rubber-duck asset, flattened into a self-contained
# USDA file with ``defaultPrim = "root"`` and a ``UsdGeom.Mesh`` at
# ``/root/Model/SurfaceMesh`` that IsaacLab's deformable spawner can find when
# applying the deformable-body schema. See ``data/duck.LICENSE`` for the asset
# license. Sourced from
# ``newton.utils.download_asset("manipulation_objects/rubber_duck")``.
DUCK_USD_PATH = os.path.join(os.path.dirname(__file__), "data", "duck.usda")


@configclass
class DeformableNewtonCfg(NewtonCfg):
    """NewtonCfg extended with model-level contact parameters for deformable objects.

    Mirrors the pattern in ``isaaclab_tasks.direct.pick_vbd_cube``: a distinct class
    so that ``_is_kitless_physics`` does not match it, ensuring Kit is launched for
    USD deformable spawning.
    """

    model_cfg: NewtonModelCfg | None = None
    """Global Newton model parameters applied after builder finalization."""


##
# Scene definition
##


@configclass
class FrankaDuckSceneCfg(InteractiveSceneCfg):
    """Scene for the Franka deformable-duck environment."""

    # robot: Franka Panda with stiffer PD does not run stable with Featherstone
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit_sim=87.0,
                stiffness=400.0,
                damping=80.0,
                armature=1e-3,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit_sim=12.0,
                stiffness=400.0,
                damping=80.0,
                armature=1e-3,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit_sim=200.0,
                stiffness=2000.0,
                damping=100.0,
            ),
        },
    )

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

    # deformable rubber duck (tetrahedral mesh from Newton's asset cache)
    duck: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="/World/envs/env_.*/Duck",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.05)),
        spawn=UsdFileCfg(
            usd_path=DUCK_USD_PATH,
            scale=(1.0, 1.0, 1.0),
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.85, 0.1)),
            # Lamé parameters from the Newton example (k_mu = k_lambda = 1e6, density = 100, k_damp ≈ 0):
            # μ = E / (2(1+ν)) = 2.5e6 / 2.5 = 1e6
            # λ = Eν / ((1+ν)(1−2ν)) = 2.5e6·0.25 / (1.25·0.5) = 1e6
            physics_material=sim_utils.DeformableBodyMaterialCfg(
                density=500.0,
                youngs_modulus=2.5e6,
                poissons_ratio=0.25,
            ),
        ),
    )

    # static table matching the Newton example: half-extents (0.4, 0.4, 0.1) → top at z = 0.2
    # NOTE: SeattleLabTable USD has its origin on the top surface, so the deformable duck
    # sits directly on it when placed at z = 0.05.
    table: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, 0.0], rot=[0.0, 0.0, 0.707, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # ground plane
    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    dome_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Commands for the duck goal pose (xyz + identity quat in robot root frame)."""

    duck_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="panda_hand",
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6),
            pos_y=(-0.25, 0.25),
            pos_z=(0.25, 0.5),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
        # Render the goal as a transparent colored sphere (a point) instead of a coordinate frame.
        goal_pose_visualizer_cfg=VisualizationMarkersCfg(
            prim_path="/Visuals/Command/goal_pose",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.03,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.1, 0.9, 0.2),
                        opacity=0.4,
                    ),
                ),
            },
        ),
    )


@configclass
class ActionsCfg:
    """7-dim arm joint position + 1-dim binary gripper."""

    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.05},
        close_command_expr={"panda_finger_.*": 0.02},
    )


@configclass
class ObservationsCfg:
    """Policy observations: joint state, duck COM in robot frame, target, last action."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        duck_com = ObsTerm(
            func=mdp.deformable_com_in_robot_root_frame,
            params={"asset_cfg": SceneEntityCfg("duck")},
        )
        target_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "duck_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset events: robot to default joint config, duck with small position randomization."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )

    reset_duck = EventTerm(
        func=mdp.reset_nodal_state_uniform,
        mode="reset",
        params={
            "position_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("duck"),
        },
    )


@configclass
class RewardsCfg:
    """Lift-to-target reward, mirroring the franka_lift structure but on the duck COM."""

    reaching_duck = RewTerm(
        func=mdp.deformable_com_ee_distance,
        params={"std": 0.1, "asset_cfg": SceneEntityCfg("duck")},
        weight=1.0,
    )
    lifting_duck = RewTerm(
        func=mdp.deformable_com_lifted,
        params={"minimal_height": 0.04, "asset_cfg": SceneEntityCfg("duck")},
        weight=15.0,
    )
    duck_goal_tracking = RewTerm(
        func=mdp.deformable_com_goal_distance,
        params={
            "std": 0.3,
            "minimal_height": 0.04,
            "command_name": "duck_pose",
            "asset_cfg": SceneEntityCfg("duck"),
        },
        weight=16.0,
    )
    duck_goal_tracking_fine_grained = RewTerm(
        func=mdp.deformable_com_goal_distance,
        params={
            "std": 0.05,
            "minimal_height": 0.04,
            "command_name": "duck_pose",
            "asset_cfg": SceneEntityCfg("duck"),
        },
        weight=5.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Time out + drop termination."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    duck_dropped = DoneTerm(
        func=mdp.deformable_com_below_minimum,
        params={"minimum_height": -0.1, "asset_cfg": SceneEntityCfg("duck")},
    )


##
# Environment configuration
##


@configclass
class FrankaDuckEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based RL environment: Franka Panda lifting a deformable rubber duck."""

    # Scene settings
    scene: FrankaDuckSceneCfg = FrankaDuckSceneCfg(num_envs=128, env_spacing=2.5, replicate_physics=True)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0

        # simulation settings
        self.sim.dt = 1 / 60.0
        self.sim.render_interval = self.decimation

        # Newton physics: Featherstone rigid + VBD soft, two-way coupled
        # (matches newton/examples/softbody/example_softbody_franka.py)
        self.sim.physics = DeformableNewtonCfg(
            solver_cfg=CoupledMJWarpVBDSolverCfg(
                # rigid_solver_cfg=FeatherstoneSolverCfg(update_mass_matrix_interval=10),
                rigid_solver_cfg=MJWarpSolverCfg(
                    njmax=40,
                    nconmax=20,
                    ls_iterations=20,
                    cone="pyramidal",
                    impratio=1,
                    ls_parallel=False,
                    integrator="implicitfast",
                    ccd_iterations=100,
                ),
                soft_solver_cfg=VBDSolverCfg(
                    iterations=5,
                    integrate_with_external_rigid_solver=True,
                    particle_enable_self_contact=False,
                    particle_collision_detection_interval=-1,
                ),
                coupling_mode="two_way",
            ),
            model_cfg=NewtonModelCfg(
                soft_contact_ke=2e6,
                soft_contact_kd=1e-7,
                soft_contact_mu=0.5,
            ),
            num_substeps=20,
            use_cuda_graph=True,
        )

        # Soften the gripper actuator so it does not crush the soft duck.
        # Same values used by the franka_lift teddy-bear deformable example.
        self.scene.robot.actuators["panda_hand"].effort_limit_sim = 50.0
        self.scene.robot.actuators["panda_hand"].stiffness = 40.0
        self.scene.robot.actuators["panda_hand"].damping = 10.0
