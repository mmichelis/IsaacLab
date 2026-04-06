# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import math
import torch
import warp as wp

from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sensors import ContactSensorCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_physx.sim.spawners.materials.physics_materials_cfg import DeformableBodyMaterialCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

# import isaaclab.envs.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from isaaclab_physx.assets import DeformableObjectCfg
from isaaclab_physx.sim import DeformableBodyPropertiesCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import G1_CFG, G1_MINIMAL_CFG  # isort:skip
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


##
# Custom MDP terms
##

def sampled_beam_pos(
    env: ManagerBasedRLEnv,
    num_samples: int = 20,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("beam"),
) -> torch.Tensor:
    """Sample evenly-spaced points along the beam sorted by x-coordinate.

    Reduces the beam observation from all 402 vertices (1206 dims) to a compact
    representation of num_samples points (60 dims for 20 samples). Points are
    selected once at the first call by sorting vertices along x and sampling at
    even intervals. Returns positions in the environment frame.
    """
    deformable = env.scene[asset_cfg.name]
    nodal_pos = wp.to_torch(deformable.data.nodal_state_w)[..., :3] - env.scene.env_origins.unsqueeze(1)

    # Compute sample indices once from the first env's vertex layout
    if not hasattr(sampled_beam_pos, "_sample_indices"):
        x_coords = nodal_pos[0, :, 0]
        sorted_indices = torch.argsort(x_coords)
        step = max(1, len(sorted_indices) // num_samples)
        sampled_beam_pos._sample_indices = sorted_indices[::step][:num_samples]

    sampled = nodal_pos[:, sampled_beam_pos._sample_indices]
    return sampled.reshape(env.num_envs, -1)


def attach_beam(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    beam_cfg: SceneEntityCfg = SceneEntityCfg("beam"),
) -> None:
    """Attach the beam to the platforms at the start of the episode."""

    beam = env.scene[beam_cfg.name]

    if env_ids is None:
        nodal_state = wp.to_torch(beam.data.default_nodal_state_w).clone()
        nodal_kinematic_target = wp.to_torch(beam.data.nodal_kinematic_target).clone()
    else:
        nodal_state = wp.to_torch(beam.data.default_nodal_state_w)[env_ids].clone()
        nodal_kinematic_target = wp.to_torch(beam.data.nodal_kinematic_target)[env_ids].clone()

    # find attachment points at minimum and maximum x coordinates, indices in first environment are used for all since the beam is replicated
    min_x = torch.min(nodal_state[0, :, 0])
    max_x = torch.max(nodal_state[0, :, 0])
    eps = 1e-2
    fixed_vertices = torch.where(
        (nodal_state[0, :, 0] <= min_x + eps) | (nodal_state[0, :, 0] >= max_x - eps)
    )[0]

    # First set all vertices as free (flag=1.0), then fix endpoints (flag=0.0)
    nodal_kinematic_target[..., :, 3] = 1.0
    nodal_kinematic_target[..., fixed_vertices, :3] = nodal_state[..., fixed_vertices, :3]
    nodal_kinematic_target[..., fixed_vertices, 3] = 0.0
    beam.write_nodal_kinematic_target_to_sim_index(nodal_kinematic_target, env_ids=env_ids)


def reset_beam(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    beam_cfg: SceneEntityCfg = SceneEntityCfg("beam"),
) -> None:
    """Reset the beam to the initial state and re-attach endpoints."""

    beam = env.scene[beam_cfg.name]

    # Reset positions
    nodal_state = wp.to_torch(beam.data.default_nodal_state_w)[env_ids].clone()
    # Zero velocities
    nodal_state[..., 3:] = 0.0
    beam.write_nodal_state_to_sim_index(nodal_state, env_ids=env_ids)

    # Re-attach beam endpoints (kinematic targets must be re-written after state reset)
    attach_beam(env, env_ids, beam_cfg)


def robot_forward_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Robot base forward (x) velocity in world frame."""
    asset = env.scene[asset_cfg.name]
    return wp.to_torch(asset.data.root_vel_w)[:, 0]


def out_of_bounds(
    env: ManagerBasedRLEnv,
    margin: float = 0.5,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    platform_start_cfg: SceneEntityCfg = SceneEntityCfg("platform_start"),
    platform_end_cfg: SceneEntityCfg = SceneEntityCfg("platform_end"),
) -> torch.Tensor:
    """Terminate if robot leaves the bounding box around the course.

    The bounding box extends from start platform to end platform with a margin
    of ±margin meters in all horizontal directions.
    """
    robot = env.scene[robot_cfg.name]
    p_start = env.scene[platform_start_cfg.name]
    p_end = env.scene[platform_end_cfg.name]

    robot_pos = wp.to_torch(robot.data.root_pos_w) - env.scene.env_origins
    start_x = wp.to_torch(p_start.data.root_pos_w)[:, 0] - env.scene.env_origins[:, 0]
    end_x = wp.to_torch(p_end.data.root_pos_w)[:, 0] - env.scene.env_origins[:, 0]

    x_out = (robot_pos[:, 0] < start_x - margin) | (robot_pos[:, 0] > end_x + margin)
    y_out = robot_pos[:, 1].abs() > margin
    return x_out | y_out


def x_position_progress(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense reward for x-position progress. Returns robot x in env frame."""
    robot = env.scene[robot_cfg.name]
    return wp.to_torch(robot.data.root_pos_w)[:, 0] - env.scene.env_origins[:, 0]


def distance_to_end_platform(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    platform_cfg: SceneEntityCfg = SceneEntityCfg("platform_end"),
) -> torch.Tensor:
    """Distance from robot to end platform x-position. Shape: (num_envs, 1)."""
    robot = env.scene[robot_cfg.name]
    platform = env.scene[platform_cfg.name]
    robot_x = wp.to_torch(robot.data.root_pos_w)[:, 0]
    platform_x = wp.to_torch(platform.data.root_pos_w)[:, 0]
    return (platform_x - robot_x).unsqueeze(-1)


def reached_end_platform(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    platform_cfg: SceneEntityCfg = SceneEntityCfg("platform_end"),
) -> torch.Tensor:
    """Reward when the robot reaches the end platform.

    Returns 1.0 when the robot's x-position exceeds the platform's x-position,
    0.0 otherwise.
    """
    robot = env.scene[robot_cfg.name]
    platform = env.scene[platform_cfg.name]
    robot_x = wp.to_torch(robot.data.root_pos_w)[:, 0]
    platform_x = wp.to_torch(platform.data.root_pos_w)[:, 0]
    return (robot_x >= platform_x).float()


##
# Scene definition
##


@configclass
class HumanoidBalanceSceneCfg(InteractiveSceneCfg):
    """Configuration for a humanoid balance scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(100.0, 100.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
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

    # humanoid robot
    robot: ArticulationCfg = G1_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(pos=(-0.5, 0.0, 1.85)),
    )

    # deformable beam
    beam: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Beam",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/mmichelis/Documents/IsaacLab/scripts/demos/walking_beam_402v.usda",
            scale=[2.0, 1.0, 1.0],
            deformable_props=DeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.4, 0.2)),
            physics_material=DeformableBodyMaterialCfg(
                density=1000.0,
                youngs_modulus=1e8,
                poissons_ratio=0.4,
                static_friction=0.5,
                dynamic_friction=0.5,
            ),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(
            pos=(2.0, 0.0, 1.0),
        ),
    )

    # start platform
    platform_start: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PlatformStart",
        spawn=sim_utils.MeshCuboidCfg(
            size=(1.0, 1.0, 0.15),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.85, 0.65)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.55, 0.0, 1.0)),
        debug_vis=True,
    )

    # end platform
    platform_end: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PlatformEnd",
        spawn=sim_utils.MeshCuboidCfg(
            size=(1.0, 1.0, 0.15),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.65, 0.65)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(4.55, 0.0, 1.0)),
        debug_vis=True,
    )

    # contact sensor
    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # spatial awareness (critical for navigation)
        root_pos = ObsTerm(func=mdp.root_pos_w)
        dist_to_goal = ObsTerm(
            func=distance_to_end_platform,
            params={"robot_cfg": SceneEntityCfg("robot"), "platform_cfg": SceneEntityCfg("platform_end")},
        )
        # standard proprioception
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        # beam nodal positions
        beam_pos = ObsTerm(
            func=sampled_beam_pos,
            params={"num_samples": 20, "asset_cfg": SceneEntityCfg("beam")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

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
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.4, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    attachment = EventTerm(
        func=attach_beam,
        mode="startup",
        params={
            "beam_cfg": SceneEntityCfg("beam"),
        }
    )
    
    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.3, 0.3)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_beam = EventTerm(
        func=reset_beam,
        mode="reset",
        params={
            "beam_cfg": SceneEntityCfg("beam"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task: forward velocity (moderate weight to encourage walking, not jumping)
    robot_forward = RewTerm(func=robot_forward_vel, weight=2.5)
    # -- task: bonus for reaching end platform
    goal_reached = RewTerm(
        func=reached_end_platform,
        weight=10.0,
        params={"robot_cfg": SceneEntityCfg("robot"), "platform_cfg": SceneEntityCfg("platform_end")},
    )
    # -- penalties
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)
    # -- penalize vertical velocity (prevents jumping)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1)
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-5e-6,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])
        }
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-5e-6,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint"])
        }
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    # -- posture penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5)

    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_pitch_joint",
                    ".*_elbow_roll_joint",
                ],
            )
        },
    )
    joint_deviation_fingers = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_five_joint",
                    ".*_three_joint",
                    ".*_six_joint",
                    ".*_four_joint",
                    ".*_zero_joint",
                    ".*_one_joint",
                    ".*_two_joint",
                ],
            )
        },
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso_joint")},
    )



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fell_off = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    torso_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"), "threshold": 1.0},
    )
    out_of_bounds = DoneTerm(
        func=out_of_bounds,
        params={
            "margin": 0.5,
            "robot_cfg": SceneEntityCfg("robot"),
            "platform_start_cfg": SceneEntityCfg("platform_start"),
            "platform_end_cfg": SceneEntityCfg("platform_end"),
        },
    )


##
# Environment configuration
##


@configclass
class HumanoidBalanceEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the humanoid balance environment."""

    # Scene settings
    scene: HumanoidBalanceSceneCfg = HumanoidBalanceSceneCfg(num_envs=1024, env_spacing=7.0, replicate_physics=False)
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
        # self.viewer.origin_type = "asset_root"
        # self.viewer.asset_name = "robot"
        # self.viewer.env_index = 6
        # self.viewer.eye = (5.0, 8.0, 2.0)
        self.viewer.resolution = (1920, 1080)
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysxCfg()
        # sensor update periods
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
