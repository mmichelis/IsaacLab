# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka cable insertion environment."""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.cable_insertion.cable_insertion_env_cfg import (
    CableInsertionEventCfg,
    CableInsertionSceneCfg,
)
from isaaclab_tasks.core.peg_in_hole.config.franka.franka_env_cfg import FrankaPegInHoleEnvCfg
from isaaclab_tasks.core.peg_in_hole.peg_in_hole_env_cfg import PegInHolePhysicsCfg


@configclass
class FrankaCableInsertionEnvCfg(FrankaPegInHoleEnvCfg):
    """Franka peg-in-hole environment with a cable attached to the peg."""

    scene: CableInsertionSceneCfg = CableInsertionSceneCfg(num_envs=8192, env_spacing=2.0)
    events: CableInsertionEventCfg = CableInsertionEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.events.conditional_reset.params["buffer_size_per_group"] = 1
        self.scene.robot.spawn.activate_contact_sensors = False
        self.scene.panda_leftfinger_object_s = None
        self.scene.panda_rightfinger_object_s = None
        for term in (self.rewards.goal_distance, self.rewards.success, self.rewards.success_bonus):
            for parameter in ("contact_threshold", "thumb_name", "finger_names"):
                term.params.pop(parameter, None)

        physics_cfg = PegInHolePhysicsCfg().newton_mjwarp_vbd_proxy
        rigid_entry = next(entry for entry in physics_cfg.solver_cfg.entries if entry.name == "rigid")
        rigid_entry.solver_cfg.cone = "pyramidal"
        object_entry = next(entry for entry in physics_cfg.solver_cfg.entries if entry.name == "object")
        object_entry.bodies.append(r"/World/envs/env_.*/Cable")
        proxy = physics_cfg.solver_cfg.proxies[0]
        proxy.bodies[:2] = [
            r"/World/envs/env_.*/Robot/Geometry/.*panda_hand",
            r"/World/envs/env_.*/Robot/Geometry/.*panda_leftfinger",
            r"/World/envs/env_.*/Robot/Geometry/.*panda_rightfinger",
        ]
        self.sim.physics = physics_cfg
