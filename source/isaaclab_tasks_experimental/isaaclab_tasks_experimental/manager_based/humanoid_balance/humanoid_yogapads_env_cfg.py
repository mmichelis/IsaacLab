# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Humanoid balance environment with a 4x2 grid of yoga pads between two platforms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sensors import ContactSensorCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
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

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from isaaclab_physx.assets import DeformableObjectCfg
from isaaclab_physx.sim import DeformableBodyPropertiesCfg
from isaaclab_physx.sim.spawners.materials.physics_materials_cfg import DeformableBodyMaterialCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import G1_CFG  # isort:skip
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


##
# Layout constants
##

# Pad grid: 4 columns (x) x 2 rows (y)
# soft_platform_98v.usda: extent [-1,1]^3 with baked scale (1,1,0.5)
# At scale [0.5, 0.5, 0.5]: 1m x 1m x 0.5m pads
PAD_HALF_X = 0.5  # half-width in x after scaling
PAD_HALF_Y = 0.5  # half-width in y after scaling
PAD_HALF_Z = 0.25  # half-height in z after scaling
PAD_GAP = 0.1
PAD_SPACING_X = 2 * PAD_HALF_X + PAD_GAP  # 1.1m center-to-center
PAD_SPACING_Y = 2 * PAD_HALF_Y + PAD_GAP  # 1.1m center-to-center
PAD_Z = PAD_HALF_Z + 0.02  # center z: half-height + 1cm elevation

# Pad center positions (relative to box origin at x=0, y=0)
PAD_X_CENTERS = [PAD_HALF_X + i * PAD_SPACING_X for i in range(4)]  # [0.5, 1.6, 2.7, 3.8]
PAD_Y_CENTERS = [-PAD_SPACING_Y / 2, PAD_SPACING_Y / 2]  # [-0.55, 0.55]

# Box dimensions (flush with pad edges)
BOX_X_MIN = 0.0
BOX_X_MAX = PAD_X_CENTERS[-1] + PAD_HALF_X  # 4.3
BOX_Y_MIN = PAD_Y_CENTERS[0] - PAD_HALF_Y  # -1.05
BOX_Y_MAX = PAD_Y_CENTERS[1] + PAD_HALF_Y  # 1.05
BOX_X_LEN = BOX_X_MAX - BOX_X_MIN  # 4.3
BOX_Y_LEN = BOX_Y_MAX - BOX_Y_MIN  # 2.1

# Platform positions (5cm gap to box edges)
PLATFORM_GAP = 0.05
START_PLATFORM_X = BOX_X_MIN - PLATFORM_GAP - 0.5  # -0.55
END_PLATFORM_X = BOX_X_MAX + PLATFORM_GAP + 0.5  # 4.85
PAD_TOP_Z = PAD_Z + PAD_HALF_Z  # ~0.51
PLATFORM_Z = PAD_TOP_Z - 0.075  # platform center so top aligns with pad top


##
# Shared pad config
##

PAD_USD_PATH = "/home/mmichelis/Documents/IsaacLab/scripts/demos/soft_platform_386v.usda"

PAD_SPAWN_CFG = sim_utils.UsdFileCfg(
    usd_path=PAD_USD_PATH,
    scale=[0.5, 0.5, 0.5],
    deformable_props=DeformableBodyPropertiesCfg(),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.3, 0.7)),
    physics_material=DeformableBodyMaterialCfg(
        density=10.0,
        youngs_modulus=1e5,
        poissons_ratio=0.3,
        static_friction=0.6,
        dynamic_friction=0.6,
    ),
)


def _pad_cfg(name_suffix: str, x: float, y: float) -> DeformableObjectCfg:
    """Create a DeformableObjectCfg for a single yoga pad at (x, y, PAD_Z)."""
    return DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Pad_" + name_suffix,
        spawn=PAD_SPAWN_CFG,
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(x, y, PAD_Z)),
    )


##
# Custom MDP terms
##


def pad_centers_obs(env: ManagerBasedEnv) -> torch.Tensor:
    """Observe the center of mass of all 8 yoga pads. Shape: (num_envs, 24)."""
    centers = []
    for key in sorted(env.scene.deformable_objects.keys()):
        if key.startswith("pad_"):
            pad = env.scene.deformable_objects[key]
            pos = wp.to_torch(pad.data.root_pos_w) - env.scene.env_origins
            centers.append(pos)
    return torch.cat(centers, dim=-1)  # (N, 8*3)


def reset_all_pads(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
) -> None:
    """Reset all yoga pads to their default nodal state."""
    for key in sorted(env.scene.deformable_objects.keys()):
        if key.startswith("pad_"):
            pad = env.scene.deformable_objects[key]
            nodal_state = wp.to_torch(pad.data.default_nodal_state_w)[env_ids].clone()
            nodal_state[..., 3:] = 0.0
            pad.write_nodal_state_to_sim_index(nodal_state, env_ids=env_ids)


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
    """Terminate if robot leaves the bounding box around the course."""
    robot = env.scene[robot_cfg.name]
    p_start = env.scene[platform_start_cfg.name]
    p_end = env.scene[platform_end_cfg.name]

    robot_pos = wp.to_torch(robot.data.root_pos_w) - env.scene.env_origins
    start_x = wp.to_torch(p_start.data.root_pos_w)[:, 0] - env.scene.env_origins[:, 0]
    end_x = wp.to_torch(p_end.data.root_pos_w)[:, 0] - env.scene.env_origins[:, 0]

    x_out = (robot_pos[:, 0] < start_x - margin) | (robot_pos[:, 0] > end_x + margin)
    y_out = robot_pos[:, 1].abs() > (BOX_Y_MAX + margin)
    return x_out | y_out


def reached_end_platform(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    platform_cfg: SceneEntityCfg = SceneEntityCfg("platform_end"),
) -> torch.Tensor:
    """Reward when the robot reaches the end platform."""
    robot = env.scene[robot_cfg.name]
    platform = env.scene[platform_cfg.name]
    robot_x = wp.to_torch(robot.data.root_pos_w)[:, 0]
    platform_x = wp.to_torch(platform.data.root_pos_w)[:, 0]
    return (robot_x >= platform_x).float()


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


##
# Scene definition
##


@configclass
class HumanoidYogaPadsSceneCfg(InteractiveSceneCfg):
    """Configuration for a humanoid yoga pads balance scene."""

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
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # humanoid robot — spawns on start platform
    robot: ArticulationCfg = G1_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(pos=(START_PLATFORM_X+1.0, 0.0, PLATFORM_Z+0.9)),
    )

    # -- 4x2 yoga pads --
    pad_0_0: DeformableObjectCfg = _pad_cfg("0_0", PAD_X_CENTERS[0], PAD_Y_CENTERS[0])
    pad_0_1: DeformableObjectCfg = _pad_cfg("0_1", PAD_X_CENTERS[0], PAD_Y_CENTERS[1])
    pad_1_0: DeformableObjectCfg = _pad_cfg("1_0", PAD_X_CENTERS[1], PAD_Y_CENTERS[0])
    pad_1_1: DeformableObjectCfg = _pad_cfg("1_1", PAD_X_CENTERS[1], PAD_Y_CENTERS[1])
    pad_2_0: DeformableObjectCfg = _pad_cfg("2_0", PAD_X_CENTERS[2], PAD_Y_CENTERS[0])
    pad_2_1: DeformableObjectCfg = _pad_cfg("2_1", PAD_X_CENTERS[2], PAD_Y_CENTERS[1])
    pad_3_0: DeformableObjectCfg = _pad_cfg("3_0", PAD_X_CENTERS[3], PAD_Y_CENTERS[0])
    pad_3_1: DeformableObjectCfg = _pad_cfg("3_1", PAD_X_CENTERS[3], PAD_Y_CENTERS[1])

    # -- Platforms --
    platform_start: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PlatformStart",
        spawn=sim_utils.MeshCuboidCfg(
            size=(1.0, 1.0, 0.15),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.85, 0.65)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(START_PLATFORM_X, 0.0, PLATFORM_Z)),
    )

    platform_end: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PlatformEnd",
        spawn=sim_utils.MeshCuboidCfg(
            size=(1.0, 1.0, 0.15),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.65, 0.65)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(END_PLATFORM_X, 0.0, PLATFORM_Z)),
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

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.1, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # spatial awareness
        root_pos = ObsTerm(func=mdp.root_pos_w)
        dist_to_goal = ObsTerm(
            func=distance_to_end_platform,
            params={"robot_cfg": SceneEntityCfg("robot"), "platform_cfg": SceneEntityCfg("platform_end")},
        )
        # pad centers (8 pads * 3 coords = 24 dims)
        pad_positions = ObsTerm(func=pad_centers_obs)
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

    reset_pads = EventTerm(
        func=reset_all_pads,
        mode="reset",
        params={},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    robot_forward = RewTerm(func=robot_forward_vel, weight=5)
    goal_reached = RewTerm(
        func=reached_end_platform,
        weight=10.0,
        params={"robot_cfg": SceneEntityCfg("robot"), "platform_cfg": SceneEntityCfg("platform_end")},
    )
    # -- penalties
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-50.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1)
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-5e-6,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])
        },
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-5e-6,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint"])
        },
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
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
        params={"minimum_height": 0.7, "asset_cfg": SceneEntityCfg("robot", joint_names="torso_joint")},
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
class HumanoidYogaPadsEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the humanoid yoga pads balance environment."""

    # Scene settings — fewer envs due to 8 deformable pads per env
    scene: HumanoidYogaPadsSceneCfg = HumanoidYogaPadsSceneCfg(
        num_envs=256, env_spacing=8.0, replicate_physics=False
    )
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
        self.episode_length_s = 30.0
        # viewer settings
        self.viewer.resolution = (1920, 1080)
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysxCfg()
        # sensor update periods
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
