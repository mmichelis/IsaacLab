# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Quadruped yoga ball environment with velocity direction commands.

The robot must roll the deformable ball in a commanded direction at a commanded speed,
similar to the standard velocity-tracking locomotion task but on a yoga ball.
"""

from __future__ import annotations

import math
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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import isaaclab.envs.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as loco_mdp

from isaaclab_physx.assets import DeformableObjectCfg
from isaaclab_physx.sim import DeformableBodyPropertiesCfg, SurfaceDeformableBodyMaterialCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.anymal import ANYMAL_D_CFG  # isort:skip



##
# Custom MDP terms
##


def ball_pos_env(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("ball")) -> torch.Tensor:
    """Ball center of mass position in the environment frame."""
    asset = env.scene[asset_cfg.name]
    return wp.to_torch(asset.data.root_pos_w) - env.scene.env_origins


def ball_vel_in_robot_frame(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Ball velocity in the robot's body frame (consistent with command frame)."""
    import isaaclab.utils.math as math_utils

    robot = env.scene[robot_cfg.name]
    ball = env.scene[ball_cfg.name]
    ball_vel_w = wp.to_torch(ball.data.root_vel_w)
    robot_quat = wp.to_torch(robot.data.root_quat_w)
    return math_utils.quat_apply_inverse(robot_quat, ball_vel_w)


def foot_contact_forces(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*FOOT"),
) -> torch.Tensor:
    """Net contact force magnitudes on each foot. Shape: (num_envs, 4)."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    forces = wp.to_torch(contact_sensor.data.net_forces_w)[:, sensor_cfg.body_ids]
    return torch.norm(forces, dim=-1)


def robot_pos_rel_ball(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Robot base position relative to ball center."""
    robot = env.scene[robot_cfg.name]
    ball = env.scene[ball_cfg.name]
    return wp.to_torch(robot.data.root_pos_w) - wp.to_torch(ball.data.root_pos_w)


def robot_ball_xy_distance_sq(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Squared horizontal distance between robot and ball."""
    rel = robot_pos_rel_ball(env, robot_cfg, ball_cfg)
    return (rel[:, :2] ** 2).sum(dim=-1)


def ball_vel_in_command_direction(
    env: ManagerBasedRLEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Reward ball velocity projected onto the commanded direction.

    Returns the dot product of ball velocity (body frame) with the command direction unit vector.
    Positive when ball moves in the right direction, scales linearly with speed.
    Zero when standing still. Negative when moving the wrong way.
    When command is near-zero (standing), returns zero (no reward or penalty).
    """
    import isaaclab.utils.math as math_utils

    robot = env.scene[robot_cfg.name]
    ball = env.scene[ball_cfg.name]
    # Ball velocity in robot body frame
    ball_vel_w = wp.to_torch(ball.data.root_vel_w)
    robot_quat = wp.to_torch(robot.data.root_quat_w)
    ball_vel_b = math_utils.quat_apply_inverse(robot_quat, ball_vel_w)[:, :2]
    # Command direction (body frame)
    command_xy = env.command_manager.get_command(command_name)[:, :2]
    command_norm = torch.norm(command_xy, dim=-1, keepdim=True).clamp(min=1e-3)
    command_dir = command_xy / command_norm
    # Dot product: ball speed in commanded direction
    vel_in_dir = (ball_vel_b * command_dir).sum(dim=-1)
    # Zero out reward when command is near-zero (standing envs)
    is_moving_cmd = torch.norm(command_xy, dim=-1) > 0.05
    return vel_in_dir * is_moving_cmd.float()


def track_ball_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Reward tracking of ball xy velocity to commanded velocity using exponential kernel.

    The command is in the robot's body frame, so ball world-frame velocity
    is transformed into the robot body frame before comparing.
    """
    import isaaclab.utils.math as math_utils

    robot = env.scene[robot_cfg.name]
    ball = env.scene[ball_cfg.name]
    # Transform ball world velocity into robot body frame
    ball_vel_w = wp.to_torch(ball.data.root_vel_w)
    robot_quat = wp.to_torch(robot.data.root_quat_w)
    ball_vel_b = math_utils.quat_apply_inverse(robot_quat, ball_vel_w)
    # Compare xy components against body-frame command
    command_vel_xy = env.command_manager.get_command(command_name)[:, :2]
    vel_error = torch.sum(torch.square(command_vel_xy - ball_vel_b[:, :2]), dim=1)
    return torch.exp(-vel_error / std**2)


def heading_alignment_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for aligning robot heading with the commanded heading target.

    Uses exponential kernel on the heading error. Directly rewards facing
    the right direction rather than rotating at the right speed.
    """
    import isaaclab.utils.math as math_utils

    command = env.command_manager.get_term(command_name)
    heading_error = math_utils.wrap_to_pi(
        command.heading_target - wp.to_torch(env.scene[asset_cfg.name].data.heading_w)
    )
    return torch.exp(-torch.square(heading_error) / std**2)


def reset_ball(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
):
    """Reset the deformable ball to its default shape.

    Must be called after the robot reset event so the robot's new position is available.
    """
    ball = env.scene[ball_cfg.name]
    nodal_state = wp.to_torch(ball.data.default_nodal_state_w)[env_ids].clone()
    nodal_state[..., 3:] = 0.0
    ball.write_nodal_state_to_sim_index(nodal_state, env_ids=env_ids)


##
# Scene definition
##


@configclass
class QuadrupedYogaDirectionSceneCfg(InteractiveSceneCfg):
    """Configuration for the quadruped yoga direction-tracking scene."""

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
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.95)),
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
                poissons_ratio=0.3,
                surface_thickness=0.1,
                surface_bend_stiffness=5e4,
                surface_shear_stiffness=5e4,
                surface_stretch_stiffness=5e4,
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
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = loco_mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=loco_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-0.3, 0.3),
            heading=(-math.pi, math.pi),
        ),
    )


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

        # ball state
        ball_pos = ObsTerm(func=ball_pos_env, params={"asset_cfg": SceneEntityCfg("ball")})
        ball_vel = ObsTerm(
            func=ball_vel_in_robot_frame,
            params={"robot_cfg": SceneEntityCfg("robot"), "ball_cfg": SceneEntityCfg("ball")},
        )
        # robot-ball relationship
        robot_rel_ball = ObsTerm(
            func=robot_pos_rel_ball,
            params={"robot_cfg": SceneEntityCfg("robot"), "ball_cfg": SceneEntityCfg("ball")},
        )
        # velocity command
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # standard quadruped proprioception
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

    # interval — gentle pushes only
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task: reward ball speed in the commanded direction (linear, strong gradient for movement)
    ball_direction_vel = RewTerm(
        func=ball_vel_in_command_direction,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "robot_cfg": SceneEntityCfg("robot"),
            "ball_cfg": SceneEntityCfg("ball"),
        },
    )
    # -- task: fine-tune velocity magnitude matching (exponential, secondary to direction reward)
    track_ball_lin_vel_xy = RewTerm(
        func=track_ball_lin_vel_xy_exp,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.5),
            "robot_cfg": SceneEntityCfg("robot"),
            "ball_cfg": SceneEntityCfg("ball"),
        },
    )
    # -- task: track commanded robot yaw rate
    # -- task: align robot heading with commanded heading
    heading_alignment = RewTerm(
        func=heading_alignment_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.5), "asset_cfg": SceneEntityCfg("robot")},
    )
    # -- task: stay on the ball
    stay_on_ball = RewTerm(
        func=robot_ball_xy_distance_sq,
        weight=-8.0,
        params={"robot_cfg": SceneEntityCfg("robot"), "ball_cfg": SceneEntityCfg("ball")},
    )
    # -- alive bonus
    alive = RewTerm(func=mdp.is_alive, weight=0.5)
    # -- termination penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # -- posture: maintain height
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.0,
        params={"target_height": 2.5, "asset_cfg": SceneEntityCfg("robot")},
    )
    # -- smoothness penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.25)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.01)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.5e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-6)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.025)
    # -- contact penalties # TODO: Do not work for rigid-deformable.
    # undesired_contacts = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=-0.5,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    # )


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
class QuadrupedYogaDirectionEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the quadruped yoga direction-tracking environment."""

    # Scene settings
    scene: QuadrupedYogaDirectionSceneCfg = QuadrupedYogaDirectionSceneCfg(
        num_envs=1024, env_spacing=4.0, replicate_physics=False
    )
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
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
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysxCfg()
        # sensor update periods
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
