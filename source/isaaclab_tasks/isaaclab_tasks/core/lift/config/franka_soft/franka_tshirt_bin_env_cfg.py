# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for placing a deformable T-shirt in a bin."""

from pathlib import Path

from isaaclab_newton.physics import NewtonCfg
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import NewtonSurfaceDeformableBodyMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.assets.deformable_object import DeformableObjectCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from ... import mdp
from .franka_cloth_bin_env_cfg import (
    _BIN_X_BOUNDS,
    _BIN_Y_BOUNDS,
    _BIN_Z_BOUNDS,
    _SUCCESS_THRESHOLD,
    FrankaClothBinEnvCfg,
    FrankaClothBinRewardsCfg,
    FrankaClothBinSceneCfg,
    FrankaClothBinTerminationsCfg,
)
from .franka_cloth_bin_env_cfg import PhysicsCfg as FrankaClothBinPhysicsCfg
from .franka_cloth_env_cfg import FrankaClothSceneCfg
from .franka_soft_env_cfg import EventCfg as FrankaSoftEventCfg

_TSHIRT_USD_PATH = Path(__file__).parent / "assets" / "unisex_shirt_qem_937.usd"


@configclass
class PhysicsCfg(PresetCfg):
    """Newton cloth-bin physics with T-shirt self-contact."""

    newton_mjwarp_vbd_proxy: NewtonCfg = FrankaClothBinPhysicsCfg().newton_mjwarp_vbd_proxy
    default = newton_mjwarp_vbd_proxy

    def __post_init__(self) -> None:
        soft_entry = next(entry for entry in self.newton_mjwarp_vbd_proxy.solver_cfg.entries if entry.name == "soft")
        soft_entry.solver_cfg.particle_enable_self_contact = True
        soft_entry.solver_cfg.particle_self_contact_radius = 0.002
        soft_entry.solver_cfg.particle_self_contact_margin = 0.002
        soft_entry.solver_cfg.particle_collision_detection_interval = 1
        soft_entry.solver_cfg.particle_vertex_contact_buffer_size = 16
        soft_entry.solver_cfg.particle_edge_contact_buffer_size = 20
        soft_entry.solver_cfg.particle_topological_contact_filter_threshold = 1
        soft_entry.solver_cfg.particle_rest_shape_contact_exclusion_radius = 0.005
        self.default = self.newton_mjwarp_vbd_proxy


@configclass
class TShirtDeformableCfg(PresetCfg):
    """Decimated Newton T-shirt deformable preset."""

    newton_mjwarp_vbd_proxy: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Deformable",
        init_state=DeformableObjectCfg.InitialStateCfg(
            pos=(0.39952, 1.20300, 0.20566),
            rot=(0.0, 0.0, 1.0, 0.0),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_TSHIRT_USD_PATH),
            scale=(0.01, 0.01, 0.01),
            deformable_props=NewtonDeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.35, 0.85)),
            physics_material=NewtonSurfaceDeformableBodyMaterialCfg(
                density=1.0,
                particle_radius=0.004,
                tri_ke=5e2,
                tri_ka=5e2,
                tri_kd=1e-3,
                edge_ke=0.5,
                edge_kd=1e-3,
            ),
        ),
    )
    default = newton_mjwarp_vbd_proxy


@configclass
class FrankaTShirtBinSceneCfg(FrankaClothBinSceneCfg):
    """Franka bin scene using the decimated Newton T-shirt."""

    deformable: TShirtDeformableCfg = TShirtDeformableCfg()

    def __post_init__(self) -> None:
        FrankaClothSceneCfg.__post_init__(self)
        self.table.spawn.visible = True
        self.table.spawn.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5))


@configclass
class FrankaTShirtBinScenePresetCfg(PresetCfg):
    """Scene presets for placing the T-shirt in a bin."""

    newton_mjwarp_vbd_proxy: FrankaTShirtBinSceneCfg = FrankaTShirtBinSceneCfg(
        num_envs=32, env_spacing=2.0, replicate_physics=True
    )
    default = newton_mjwarp_vbd_proxy


@configclass
class FrankaTShirtBinEventCfg(FrankaSoftEventCfg):
    """Reset events with bounded T-shirt translation."""

    def __post_init__(self) -> None:
        self.reset_deformable.params["position_range"] = {
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (0.0, 0.0),
        }


@configclass
class FrankaTShirtBinRewardsCfg(FrankaClothBinRewardsCfg):
    """Cloth-bin rewards with area-weighted T-shirt occupancy."""

    deformable_vertex_occupancy: None = None
    deformable_area_occupancy = RewTerm(
        func=mdp.DeformableAreaFractionInBounds,
        params={
            "x_bounds": _BIN_X_BOUNDS,
            "y_bounds": _BIN_Y_BOUNDS,
            "z_bounds": _BIN_Z_BOUNDS,
            "success_threshold": _SUCCESS_THRESHOLD,
            "mesh_path": str(_TSHIRT_USD_PATH),
            "output": "fraction",
            "asset_cfg": SceneEntityCfg("deformable"),
        },
        weight=5.0,
    )
    success_bonus = RewTerm(
        func=mdp.DeformableAreaFractionInBounds,
        params={
            "x_bounds": _BIN_X_BOUNDS,
            "y_bounds": _BIN_Y_BOUNDS,
            "z_bounds": _BIN_Z_BOUNDS,
            "success_threshold": _SUCCESS_THRESHOLD,
            "mesh_path": str(_TSHIRT_USD_PATH),
            "output": "event",
            "asset_cfg": SceneEntityCfg("deformable"),
        },
        weight=20.0,
    )


@configclass
class FrankaTShirtBinTerminationsCfg(FrankaClothBinTerminationsCfg):
    """Cloth-bin terminations with area-weighted T-shirt success."""

    success = DoneTerm(
        func=mdp.DeformableAreaFractionInBounds,
        params={
            "x_bounds": _BIN_X_BOUNDS,
            "y_bounds": _BIN_Y_BOUNDS,
            "z_bounds": _BIN_Z_BOUNDS,
            "success_threshold": _SUCCESS_THRESHOLD,
            "mesh_path": str(_TSHIRT_USD_PATH),
            "output": "success",
            "asset_cfg": SceneEntityCfg("deformable"),
        },
    )


@configclass
class FrankaTShirtBinEnvCfg(FrankaClothBinEnvCfg):
    """Manager-based RL environment for placing a T-shirt in a bin."""

    scene: FrankaTShirtBinScenePresetCfg = FrankaTShirtBinScenePresetCfg()
    events: FrankaTShirtBinEventCfg = FrankaTShirtBinEventCfg()
    rewards: FrankaTShirtBinRewardsCfg = FrankaTShirtBinRewardsCfg()
    terminations: FrankaTShirtBinTerminationsCfg = FrankaTShirtBinTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.sim.physics = PhysicsCfg()
