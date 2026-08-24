# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for placing cloth in a bin with a Franka robot."""

from isaaclab_newton.physics import NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from ... import mdp
from .franka_cloth_env_cfg import (
    FrankaClothEnvCfg,
    FrankaClothRewardsCfg,
    FrankaClothSceneCfg,
)
from .franka_cloth_env_cfg import (
    PhysicsCfg as FrankaClothPhysicsCfg,
)
from .franka_soft_env_cfg import TABLE_SPAWN_CFG
from .franka_soft_env_cfg import CurriculumCfg as FrankaSoftCurriculumCfg
from .franka_soft_env_cfg import EventCfg as FrankaSoftEventCfg
from .franka_soft_env_cfg import TerminationsCfg as FrankaSoftTerminationsCfg

_BIN_BODY_PATTERN = r"/World/envs/env_[^/]+/Bin"
_TABLE_BODY_PATTERN = r"/World/envs/env_[^/]+/Table"
_BIN_X_BOUNDS = (0.24, 0.76)
_BIN_Y_BOUNDS = (-0.80, -0.49)
_BIN_Z_BOUNDS = (-0.26, -0.01)
_INSERTION_X_BOUNDS = _BIN_X_BOUNDS
_INSERTION_Y_BOUNDS = (-0.72, -0.57)
_INSERTION_Z_BOUNDS = (-0.26, 0.08)
_SUCCESS_THRESHOLD = 0.95


@configclass
class PhysicsCfg(FrankaClothPhysicsCfg):
    """Cloth physics presets with bin coupling for Newton."""

    newton_mjwarp_vbd_proxy: NewtonCfg = FrankaClothPhysicsCfg().newton_mjwarp_vbd_proxy
    default = newton_mjwarp_vbd_proxy

    def __post_init__(self) -> None:
        rigid_bodies = self.newton_mjwarp_vbd_proxy.solver_cfg.entries[0].bodies
        proxy_bodies = self.newton_mjwarp_vbd_proxy.solver_cfg.proxies[0].bodies
        rigid_bodies[:] = [body for body in rigid_bodies if "Support" not in body]
        proxy_bodies[:] = [body for body in proxy_bodies if "Support" not in body]
        rigid_bodies.extend((_BIN_BODY_PATTERN, _TABLE_BODY_PATTERN))
        proxy_bodies.extend((_BIN_BODY_PATTERN, _TABLE_BODY_PATTERN))
        self.default = self.newton_mjwarp_vbd_proxy


@configclass
class FrankaClothBinSceneCfg(FrankaClothSceneCfg):
    """Franka cloth scene with a bin beside the table."""

    support_neg_y: None = None
    support_pos_y: None = None

    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, -0.525)),
        spawn=TABLE_SPAWN_CFG.replace(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        ),
    )

    bin: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Bin",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/KLT_Bin/small_KLT.usd",
            scale=(2.0, 2.0, 2.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, -0.64784, -0.14636),
            rot=(0.0, 0.0, 0.70710678, 0.70710678),
        ),
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.deformable.newton_mjwarp_vbd_proxy.spawn.physics_material.particle_radius = 0.004
        self.deformable.default.spawn.physics_material.particle_radius = 0.004
        for deformable_cfg in (
            self.deformable.newton_mjwarp_vbd_proxy,
            self.deformable.physx,
            self.deformable.isaacsim_physx,
            self.deformable.ovphysx,
            self.deformable.default,
        ):
            deformable_cfg.init_state.pos = (0.4, 0.0, 0.002)
            deformable_cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.deformable.newton_mjwarp_vbd_proxy.init_state.pos = (0.4, 0.0, 0.0045)
        self.deformable.default.init_state.pos = (0.4, 0.0, 0.0045)
        # The base command visualizer normally renders this invisible collider.
        self.table.spawn.visible = True
        self.table.spawn.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5))


@configclass
class FrankaClothBinScenePresetCfg(PresetCfg):
    """Scene presets for placing cloth in a bin."""

    newton_mjwarp_vbd_proxy: FrankaClothBinSceneCfg = FrankaClothBinSceneCfg(
        num_envs=2048, env_spacing=2.0, replicate_physics=True
    )
    physx: FrankaClothBinSceneCfg = FrankaClothBinSceneCfg(num_envs=2048, env_spacing=2.0, replicate_physics=False)
    isaacsim_physx = physx
    ovphysx: FrankaClothBinSceneCfg = FrankaClothBinSceneCfg(num_envs=2048, env_spacing=2.0, replicate_physics=True)
    default = newton_mjwarp_vbd_proxy


@configclass
class FrankaClothBinRewardsCfg(FrankaClothRewardsCfg):
    """Rewards for placing cloth in the bin."""

    lifting_deformable: None = None
    deformable_goal_tracking: None = None
    reaching_deformable = FrankaClothRewardsCfg().reaching_deformable.replace(weight=0.5)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-5e-3)

    deformable_bin_distance = RewTerm(
        func=mdp.DeformableVertexDistanceToBoundsProgress,
        params={
            "x_bounds": _BIN_X_BOUNDS,
            "y_bounds": _BIN_Y_BOUNDS,
            "z_bounds": _BIN_Z_BOUNDS,
            "asset_cfg": SceneEntityCfg("deformable"),
        },
        weight=20.0,
    )

    deformable_insertion_alignment = RewTerm(
        func=mdp.DeformableVertexDistanceToBoundsProgress,
        params={
            "x_bounds": _INSERTION_X_BOUNDS,
            "y_bounds": _INSERTION_Y_BOUNDS,
            "z_bounds": _INSERTION_Z_BOUNDS,
            "asset_cfg": SceneEntityCfg("deformable"),
        },
        weight=5.0,
    )

    deformable_vertex_occupancy = RewTerm(
        func=mdp.DeformableVertexFractionInBounds,
        params={
            "x_bounds": _BIN_X_BOUNDS,
            "y_bounds": _BIN_Y_BOUNDS,
            "z_bounds": _BIN_Z_BOUNDS,
            "success_threshold": _SUCCESS_THRESHOLD,
            "asset_cfg": SceneEntityCfg("deformable"),
        },
        weight=5.0,
    )

    success_bonus = RewTerm(
        func=mdp.deformable_vertices_in_bounds_event,
        params={
            "x_bounds": _BIN_X_BOUNDS,
            "y_bounds": _BIN_Y_BOUNDS,
            "z_bounds": _BIN_Z_BOUNDS,
            "success_threshold": _SUCCESS_THRESHOLD,
            "asset_cfg": SceneEntityCfg("deformable"),
        },
        weight=20.0,
    )


@configclass
class FrankaClothBinTerminationsCfg(FrankaSoftTerminationsCfg):
    """Terminations for placing cloth in the bin."""

    deformable_out_of_bounds: None = None

    success = DoneTerm(
        func=mdp.deformable_vertices_in_bounds,
        params={
            "x_bounds": _BIN_X_BOUNDS,
            "y_bounds": _BIN_Y_BOUNDS,
            "z_bounds": _BIN_Z_BOUNDS,
            "success_threshold": _SUCCESS_THRESHOLD,
            "asset_cfg": SceneEntityCfg("deformable"),
        },
    )

    deformable_touches_ground = DoneTerm(
        func=mdp.deformable_touches_ground,
        params={
            "ground_height": -1.05,
            "tolerance": 0.005,
            "asset_cfg": SceneEntityCfg("deformable"),
        },
    )


@configclass
class FrankaClothBinCurriculumCfg(FrankaSoftCurriculumCfg):
    """Curriculum without lift-specific action-rate and gravity ramps."""

    action_rate: None = None
    gravity: None = None


@configclass
class FrankaClothBinEnvCfg(FrankaClothEnvCfg):
    """Manager-based RL environment for placing cloth in a bin."""

    scene: FrankaClothBinScenePresetCfg = FrankaClothBinScenePresetCfg()
    commands: None = None
    events: FrankaSoftEventCfg = FrankaSoftEventCfg()
    curriculum: FrankaClothBinCurriculumCfg = FrankaClothBinCurriculumCfg()
    rewards: FrankaClothBinRewardsCfg = FrankaClothBinRewardsCfg()
    terminations: FrankaClothBinTerminationsCfg = FrankaClothBinTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.sim.physics = PhysicsCfg()
        self.observations.policy.target_position = None
        self.viewer.eye = (2.2, -2.4, 1.6)
        self.viewer.lookat = (0.35, -0.10, 0.05)
        self.sim.default_visualizer_cfg.eye = self.viewer.eye
        self.sim.default_visualizer_cfg.lookat = self.viewer.lookat
