# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_physx.sim.spawners.materials.physics_materials_cfg import DeformableBodyMaterialCfg
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

import isaaclab.envs.mdp as mdp

from isaaclab_physx.assets import DeformableObjectCfg
from isaaclab_physx.sim import DeformableBodyPropertiesCfg, SurfaceDeformableBodyMaterialCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import G1_CFG  # isort:skip
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


##
# Custom MDP terms
##

def nodal_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Get nodal positions in environment frame of a deformable object as an observation."""
    deformable = env.scene[asset_cfg.name]
    nodal_state = wp.to_torch(deformable.data.nodal_state_w)[..., :3] - env.scene.env_origins.unsqueeze(1)

    return nodal_state.reshape(env.num_envs, -1)


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

    nodal_kinematic_target[..., fixed_vertices, :3] = nodal_state[..., fixed_vertices, :3]
    nodal_kinematic_target[..., fixed_vertices, 3] = 0.0
    beam.write_nodal_kinematic_target_to_sim_index(nodal_kinematic_target, env_ids=env_ids)


def reset_beam(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    beam_cfg: SceneEntityCfg = SceneEntityCfg("beam"),
) -> None:
    """Reset the beam to the initial state at the reset of the episode."""

    beam = env.scene[beam_cfg.name]

    # Reset positions
    nodal_state = wp.to_torch(beam.data.default_nodal_state_w)[env_ids].clone()
    # Zero velocities
    nodal_state[..., 3:] = 0.0

    beam.write_nodal_state_to_sim_index(nodal_state, env_ids=env_ids)


##
# Scene definition
##


@configclass
class HumanoidBalanceSceneCfg(InteractiveSceneCfg):
    """Configuration for a humanoid balance scene."""

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

    # humanoid robot
    robot: ArticulationCfg = G1_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(pos=(-0.5, 0.0, 2.0)),
    )

    # deformable beam
    beam: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Beam",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/mmichelis/Documents/IsaacLab/scripts/demos/walking_beam_402v.usda",
            scale=[1.0, 1.0, 1.0],
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
            pos=(1.0, 0.0, 1.0),
        ),
    )

    # start platform
    platform_start: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PlatformStart",
        spawn=sim_utils.MeshCuboidCfg(
            size=(1.0, 1.0, 0.15),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5),
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
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.65, 0.65)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.55, 0.0, 1.0)),
        debug_vis=True,
    )

    # contact sensor
    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
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
        
        # beam nodal positions
        beam_pos = ObsTerm(func=nodal_pos, params={"asset_cfg": SceneEntityCfg("beam")})

        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
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
            "pose_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15), "yaw": (-0.5, 0.5)},
            "velocity_range": {
                "x": (-0.15, 0.15),
                "y": (-0.15, 0.15),
                "z": (0.0, 0.0),
            },
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
    
    # -- alive bonus
    alive = RewTerm(func=mdp.is_alive, weight=0.5)
    # -- termination penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fell_off = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )


##
# Environment configuration
##


@configclass
class HumanoidBalanceEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the humanoid balance environment."""

    # Scene settings
    scene: HumanoidBalanceSceneCfg = HumanoidBalanceSceneCfg(num_envs=1024, env_spacing=5.0, replicate_physics=False)
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
        self.episode_length_s = 2.0
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
