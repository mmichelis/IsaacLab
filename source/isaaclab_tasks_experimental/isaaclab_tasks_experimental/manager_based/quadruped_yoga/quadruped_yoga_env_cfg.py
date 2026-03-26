# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sensors import ContactSensorCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import isaaclab.envs.mdp as mdp

from isaaclab_physx.assets import DeformableObjectCfg
from isaaclab_physx.sim import DeformableBodyPropertiesCfg, SurfaceDeformableBodyMaterialCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.anymal import ANYMAL_D_CFG  # isort:skip
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Custom MDP terms
##


def ball_pos_env(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("ball")) -> torch.Tensor:
    """Ball center of mass position in the environment frame."""
    asset = env.scene[asset_cfg.name]
    return wp.to_torch(asset.data.root_pos_w) - env.scene.env_origins


def robot_pos_rel_ball(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Robot base position relative to ball center in the environment frame."""
    robot = env.scene[robot_cfg.name]
    ball = env.scene[ball_cfg.name]
    robot_pos = wp.to_torch(robot.data.root_pos_w)
    ball_pos = wp.to_torch(ball.data.root_pos_w)
    return robot_pos - ball_pos


def ball_vel_x(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("ball")) -> torch.Tensor:
    """Reward for ball velocity in the positive x direction."""
    asset = env.scene[asset_cfg.name]
    ball_vel = wp.to_torch(asset.data.root_vel_w)
    return ball_vel[:, 0]


def ball_vel_y_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("ball")) -> torch.Tensor:
    """Penalize ball lateral (y) velocity to encourage straight-line rolling."""
    asset = env.scene[asset_cfg.name]
    ball_vel = wp.to_torch(asset.data.root_vel_w)
    return ball_vel[:, 1].abs()


def ball_vel_env(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("ball")) -> torch.Tensor:
    """Ball velocity in world frame."""
    asset = env.scene[asset_cfg.name]
    return wp.to_torch(asset.data.root_vel_w)


def robot_forward_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Robot base forward (x) velocity in world frame."""
    asset = env.scene[asset_cfg.name]
    return wp.to_torch(asset.data.root_vel_w)[:, 0]


def robot_ball_xy_distance_sq(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Squared horizontal distance between robot and ball."""
    rel = robot_pos_rel_ball(env, robot_cfg, ball_cfg)
    return (rel[:, :2] ** 2).sum(dim=-1)


def reset_ball(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
):
    """Reset the deformable ball to its default shape.

    Must be called after the robot reset event so the robot's new position is available.
    """
    robot = env.scene[robot_cfg.name]
    ball = env.scene[ball_cfg.name]

    # Get default ball nodal state and shift xy to match robot, keep default z (ball_height)
    nodal_state = wp.to_torch(ball.data.default_nodal_state_w)[env_ids].clone()
    # Zero velocities
    nodal_state[..., 3:] = 0.0

    ball.write_nodal_state_to_sim_index(nodal_state, env_ids=env_ids)


##
# Scene definition
##


@configclass
class QuadrupedYogaSceneCfg(InteractiveSceneCfg):
    """Configuration for a quadruped yoga scene."""

    # ground plane with low friction for ball rolling
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(100.0, 100.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.2,
                dynamic_friction=0.2,
                restitution=0.0,
            ),
        ),
    )

    # lights
    # dome_light = AssetBaseCfg(
    #     prim_path="/World/DomeLight",
    #     spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=2000.0),
    # )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # quadruped robot
    robot: ArticulationCfg = ANYMAL_D_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    )

    # deformable ball
    ball: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/mmichelis/Documents/IsaacLab/scripts/demos/icosphere_3.usda",
            scale=[0.75, 0.75, 0.75],
            deformable_props=DeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 0.9)),
            physics_material=SurfaceDeformableBodyMaterialCfg(
                density=10.0,
                youngs_modulus=1e4,
                poissons_ratio=0.4,
                surface_thickness=0.1,
                surface_bend_stiffness=1e5,
                surface_shear_stiffness=1e5,
                surface_stretch_stiffness=1e5,
                static_friction=0.5,
                dynamic_friction=0.5,
            ),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.76),
        ),
    )

    # contact sensor
    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # ball center of mass in env frame
        ball_pos = ObsTerm(func=ball_pos_env, params={"asset_cfg": SceneEntityCfg("ball")})
        # ball velocity (so policy can anticipate ball dynamics)
        ball_vel = ObsTerm(func=ball_vel_env, params={"asset_cfg": SceneEntityCfg("ball")})
        # robot position relative to ball (so policy knows where it is on the ball)
        robot_rel_ball = ObsTerm(
            func=robot_pos_rel_ball,
            params={"robot_cfg": SceneEntityCfg("robot"), "ball_cfg": SceneEntityCfg("ball")},
        )
        # standard quadruped observations
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15), "yaw": (-0.5, 0.5)},
            "velocity_range": {
                "x": (-0.15, 0.15),
                "y": (-0.15, 0.15),
                "z": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.8, 1.2),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_ball = EventTerm(
        func=reset_ball,
        mode="reset",
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ball_cfg": SceneEntityCfg("ball"),
        },
    )

    # interval — gentle pushes only (strong pushes knock robot off ball)
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task: reward ball forward velocity
    ball_forward_vel = RewTerm(func=ball_vel_x, weight=4.0, params={"asset_cfg": SceneEntityCfg("ball")})
    # -- task: reward robot moving forward (encourages moving WITH the ball, not kicking it away)
    robot_forward = RewTerm(func=robot_forward_vel, weight=1.0, params={"asset_cfg": SceneEntityCfg("robot")})
    # -- task: penalize ball lateral drift
    ball_lateral_vel = RewTerm(func=ball_vel_y_penalty, weight=-1.0, params={"asset_cfg": SceneEntityCfg("ball")})
    # -- task: stay on the ball (penalize xy distance between robot and ball)
    stay_on_ball = RewTerm(
        func=robot_ball_xy_distance_sq,
        weight=-10.0,
        params={"robot_cfg": SceneEntityCfg("robot"), "ball_cfg": SceneEntityCfg("ball")},
    )
    # -- alive bonus
    alive = RewTerm(func=mdp.is_alive, weight=0.5)
    # -- termination penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # -- posture: maintain height (prevents leaning forward and toppling off)
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-2.0,
        params={"target_height": 2.5, "asset_cfg": SceneEntityCfg("robot")},
    )
    # -- smoothness penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.01)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-5.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-6)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    # -- contact penalties
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    fell_off_ball = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.75, "asset_cfg": SceneEntityCfg("robot")},
    )


##
# Environment configuration
##


@configclass
class QuadrupedYogaEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the quadruped yoga environment."""

    # Scene settings
    scene: QuadrupedYogaSceneCfg = QuadrupedYogaSceneCfg(num_envs=1024, env_spacing=4.0, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # viewer settings
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 6
        self.viewer.eye = (5.0, 8.0, 2.0)
        self.viewer.resolution = (1920, 1080)
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysxCfg()
        # sensor update periods
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
