# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CMA-ES stiffness/damping sysid env for the Franka FR3 WITH a Robotiq 2F-85 gripper.

Sibling of ``fr3_sysid_env_cfg.py``. Same replay + CMA-ES loop and the same
fitted ``fr3_joint.*`` PD gains, but the visible arm carries the Robotiq 2F-85
gripper the real data was collected with (held static / zero gripper motion).

The asset is the Franka Robotics ``franka.usd`` with the ``Gripper=Robotiq_2F_85``
variant baked, the whole closure flattened locally, and the arm joints renamed
``panda_jointN`` -> ``fr3_jointN`` (regenerate with
``scripts/sysid/build_fr3_robotiq_visual_asset.py``). The optimizer still writes
only the ``fr3_joint.*`` gains; the 6 Robotiq revolute joints are held rigid at
position 0 by a stiff PD "gripper_hold" group so zero gripper action keeps the
gripper static and it does not flop under wrist motion.
"""

from __future__ import annotations

import os

import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.sysid.sysid_env_cfg import (
    SysidActionsCfg,
    SysIdCfg,
    SysidEventCfg,
    SysidObservationsCfg,
    SysidRewardsCfg,
    SysidTerminationsCfg,
)
from isaaclab_tasks.utils import PresetCfg

# Reuse the FR3 dataset joint order, home pose and CMA-ES bounds unchanged — only
# the arm joints are fitted and their names/order are identical.
from .fr3_sysid_env_cfg import FR3_READY_POSE, FR3_SYSID_JOINT_ORDER, build_bounds  # noqa: F401

##
# Robot definition
##

# Visible arm+gripper: the Franka Robotics franka.usd with the Robotiq_2F_85
# variant baked and flattened, arm joints renamed panda_jointN -> fr3_jointN.
# Self-contained; regenerate with build_fr3_robotiq_visual_asset.py.
FR3_ROBOTIQ_USD_PATH = os.path.join(
    os.path.dirname(__file__), "assets", "franka_robotiq_visual", "fr3_robotiq_visual.usda"
)

# The 6 Robotiq 2F-85 revolute joints — held static, never fitted.
ROBOTIQ_HOLD_JOINTS = [
    "finger_joint",
    "right_outer_knuckle_joint",
    ".*_inner_finger_joint",
    ".*_inner_finger_knuckle_joint",
]

FR3_ROBOTIQ_SYSID_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=FR3_ROBOTIQ_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        # The franka.usd carries its own fixed base ("rootJoint"); do not re-assert.
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
        ),
    ),
    # Arm at the FR3 ready pose; gripper joints default to 0 (held there).
    init_state=ArticulationCfg.InitialStateCfg(joint_pos=dict(FR3_READY_POSE)),
    actuators={
        # Fitted arm group — identical to the gripper-less FR3 task. The optimizer
        # overwrites stiffness/damping per env every generation.
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint.*"],
            effort_limit_sim={"fr3_joint[1-4]": 187.0, "fr3_joint[5-7]": 112.0},
            velocity_limit_sim={
                "fr3_joint[1-4]": 2.62,
                "fr3_joint5": 5.26,
                "fr3_joint6": 4.18,
                "fr3_joint7": 5.26,
            },
            stiffness={"fr3_joint[1-4]": 600.0, "fr3_joint5": 250.0, "fr3_joint6": 150.0, "fr3_joint7": 50.0},
            damping={"fr3_joint[1-4]": 30.0, "fr3_joint5": 10.0, "fr3_joint6": 10.0, "fr3_joint7": 5.0},
        ),
        # Static holding group: stiff PD at position 0 so the gripper stays rigid
        # under wrist motion (zero gripper action). Never touched by the optimizer.
        "gripper_hold": ImplicitActuatorCfg(
            joint_names_expr=ROBOTIQ_HOLD_JOINTS,
            effort_limit_sim=100.0,
            velocity_limit_sim=10.0,
            stiffness=100.0,
            damping=10.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


##
# Physics preset — mjwarp/Newton is the default backend for this task.
##


@configclass
class FR3RobotiqSysidPhysicsCfg(PresetCfg):
    physx: PhysxCfg = PhysxCfg()

    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=50,
            nconmax=20,
            integrator="implicitfast",
            # Free-air chirps on a fixed-base arm never make contact; also spares
            # the packed Robotiq fingers from self-contact jitter.
            disable_contacts=True,
        ),
        num_substeps=1,
    )

    default = newton_mjwarp


##
# Scene
##


@configclass
class FR3RobotiqSysidSceneCfg(InteractiveSceneCfg):
    """Minimal scene: fixed-base FR3+Robotiq and a light. No ground — nothing touches it."""

    robot: ArticulationCfg = FR3_ROBOTIQ_SYSID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


##
# Environment
##


@configclass
class FR3RobotiqSysIdCfg(SysIdCfg):
    robot_name: str = "franka_robotiq"
    joint_order: list[str] = FR3_SYSID_JOINT_ORDER
    # Same compensated-plant assumption as the gripper-less FR3 task.
    zero_gravity: bool = True


@configclass
class FR3RobotiqSysIdEnvCfg(ManagerBasedRLEnvCfg):
    """CMA-ES sysid env for FR3+Robotiq implicit-actuator gains; driven by scripts/sysid/fit.py."""

    scene: FR3RobotiqSysidSceneCfg = FR3RobotiqSysidSceneCfg(num_envs=4096, env_spacing=1.5)
    observations: SysidObservationsCfg = SysidObservationsCfg()
    actions: SysidActionsCfg = SysidActionsCfg()
    rewards: SysidRewardsCfg = SysidRewardsCfg()
    events: SysidEventCfg = SysidEventCfg()
    terminations: SysidTerminationsCfg = SysidTerminationsCfg()

    sysid: FR3RobotiqSysIdCfg = FR3RobotiqSysIdCfg()

    def __post_init__(self) -> None:
        # Same timing as the gripper-less FR3 task: 1 kHz physics, 200 Hz command
        # hold via decimation. fit.py re-derives both from the dataset.
        self.decimation = 5
        self.sim.dt = 0.001  # 1 kHz; env step rate = sim.dt * decimation = 200 Hz
        self.sim.render_interval = self.decimation
        self.episode_length_s = 120.0
        self.sim.physics = FR3RobotiqSysidPhysicsCfg()
        if self.sysid.zero_gravity:
            self.sim.gravity = (0.0, 0.0, 0.0)
