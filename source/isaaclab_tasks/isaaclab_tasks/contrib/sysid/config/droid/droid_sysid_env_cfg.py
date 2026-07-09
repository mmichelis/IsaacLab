# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CMA-ES stiffness/damping sysid env for the Droid robot (Franka arm + Robotiq 2F-85).

Same replay + CMA-ES loop as the FR3 task, but the plant is the droid articulation
used by dextrah training (fabrics-sim ``droid_robotiq.usd``): a panda-named Franka
arm carrying a Robotiq 2F-85 (7 arm joints + 6 gripper joints, 5 of which mimic the
master via USD equality constraints). Fitting on this asset makes the fitted gains
directly transferable to the training env's ``DROID_CFG`` actuator groups.

The chirp datasets are recorded on an FR3 (columns ``fr3_joint*`` after contract
normalization); ``sysid.sim_joint_name_map`` maps them onto the articulation's
``panda_joint*`` names. The gripper joints are never in the dataset: the replay
holds them at their default (home) position with the training-time actuator gains.

PhysX is the only backend for this task — the droid USD carries no MuJoCo actuator
attributes, and PhysX is what dextrah trains on (the plant the gains must match).
"""

from __future__ import annotations

import os
import tempfile

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

##
# Robot definition
##


def _resolve_droid_usd_path() -> str:
    """Locate the fabrics-sim droid USD: $DROID_USD_PATH > container mount > host checkout."""
    fabrics_subpath = "src/fabrics_sim/models/robots/USD/droid_sim/droid_robotiq.usd"
    candidates = [
        os.environ.get("DROID_USD_PATH", ""),
        os.path.join("/workspace/fabrics-sim", fabrics_subpath),
        os.path.expanduser(
            os.path.join("~/workspaces/nvblox_next/submodules/fabrics-sim", fabrics_subpath)
        ),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(
        "droid_robotiq.usd not found — set DROID_USD_PATH or mount fabrics-sim at /workspace/fabrics-sim"
    )


DROID_USD_PATH = _resolve_droid_usd_path()

# The droid URDF's virtual fabric-control frames are massless by design. The
# USD authors physics:mass = 0 on them, and PhysX replaces a zero-mass dynamic
# body (no collision geometry to derive from) with a 1.0 kg RUNTIME fallback —
# 7 kg total, ~6 kg at the TCP. Authoring a tiny positive mass suppresses the
# fallback while keeping the bodies dynamically negligible.
_PHANTOM_FRAME_PRIMS = (
    "right_gripper",
    "right_gripper_x",
    "right_gripper_x_neg",
    "right_gripper_y",
    "right_gripper_y_neg",
    "right_gripper_z",
    "right_gripper_z_neg",
)
_PHANTOM_FRAME_MASS_KG = 1.0e-4


# Panda -> FR3 conversion: the droid USD carries Panda link masses and Panda
# joint limits, but the real arm is an FR3 (franka_description identified
# values). Masses in kg per link index 0-7; limits in degrees as USD authors
# them. The j6 window shift ([-1, 215] Panda vs [31.2, 258.8] FR3) is a hard
# feasibility difference, not just dynamics.
_FR3_LINK_MASSES_KG = (2.397, 2.927, 2.936, 2.245, 2.616, 2.327, 1.817, 0.627)
_FR3_JOINT_LIMITS_DEG = (
    (-157.2, 157.2),
    (-102.2, 102.2),
    (-166.2, 166.2),
    (-174.3, -8.7),
    (-160.8, 160.8),
    (31.2, 258.8),
    (-172.8, 172.8),
)


def _materialize_corrected_usd(enable_fr3_conversion: bool = False) -> str:
    """Write the corrected-plant overlay over the fabrics droid USD.

    Always fixes the phantom frame masses. With enable_fr3_conversion the layer
    additionally swaps the Panda link masses and joint limits for the FR3 ones.
    """
    overrides = [
        f'    over "{name}" {{\n        float physics:mass = {_PHANTOM_FRAME_MASS_KG}\n    }}'
        for name in _PHANTOM_FRAME_PRIMS
    ]
    if enable_fr3_conversion:
        for i, mass in enumerate(_FR3_LINK_MASSES_KG):
            overrides.append(f'    over "panda_link{i}" {{\n        float physics:mass = {mass}\n    }}')
        joint_overrides = []
        for i, (lo, hi) in enumerate(_FR3_JOINT_LIMITS_DEG, start=1):
            joint_overrides.append(
                f'        over "panda_joint{i}" {{\n'
                f"            float physics:lowerLimit = {lo}\n"
                f"            float physics:upperLimit = {hi}\n"
                "        }"
            )
        overrides.append('    over "joints" {\n' + "\n".join(joint_overrides) + "\n    }")
    body = "\n".join(overrides)
    layer = (
        '#usda 1.0\n(\n    defaultPrim = "droid"\n'
        f'    subLayers = [\n        @{DROID_USD_PATH}@\n    ]\n)\n\n'
        f'over "droid" {{\n{body}\n}}\n'
    )
    suffix = "fr3" if enable_fr3_conversion else "corrected"
    out_path = os.path.join(tempfile.gettempdir(), f"droid_robotiq_{suffix}_{os.getpid()}.usda")
    with open(out_path, "w") as f:
        f.write(layer)
    return out_path

# Dataset column order (contract-normalized FR3 names).
DROID_SYSID_JOINT_ORDER: list[str] = [f"fr3_joint{i}" for i in range(1, 8)]

# The droid USD names the arm joints panda_joint* (the arm is a Panda-derived
# Franka model); the recorded FR3 data maps 1:1 by joint number.
DROID_SIM_JOINT_NAME_MAP: dict[str, str] = {
    f"fr3_joint{i}": f"panda_joint{i}" for i in range(1, 8)
}

# Franka "ready" pose — the homing pose the data collection chirps around
# (isaac_ros_sysid config/robots/franka_fr3.yaml), on the articulation names.
# Gripper joints at 0.0 = open (dextrah's GRIPPER_HOME_Q).
DROID_READY_POSE: dict[str, float] = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.7853981633974483,
    "panda_joint3": 0.0,
    "panda_joint4": -2.356194490192345,
    "panda_joint5": 0.0,
    "panda_joint6": 1.5707963267948966,
    "panda_joint7": 0.7853981633974483,
    "robotiq_.*_joint": 0.0,
}

DROID_SYSID_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=DROID_USD_PATH,
        activate_contact_sensors=False,
        # disable_gravity mirrors dextrah's DROID_CFG: its geometric fabric is
        # gravity-free and expects the low-level controller to gravity-compensate,
        # which matches the sysid zero-gravity (compensated-plant) assumption.
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        # Solver iteration counts replicate dextrah's DROID_CFG — the fitted
        # gains must match the plant as the training env integrates it.
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=1,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(joint_pos=dict(DROID_READY_POSE)),
    actuators={
        # Arm: dextrah DROID_CFG training gains (uniform 400/80) as the asset
        # default — fit.py's "asset_default_gains" baseline then measures how far
        # the current training gains are from the recorded plant. The optimizer
        # overwrites these per env every generation.
        "panda_arm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-7]"],
            effort_limit_sim=None,
            velocity_limit_sim=None,
            stiffness=400.0,
            damping=80.0,
        ),
        # Robotiq master + 5 mimics: exactly DROID_CFG. Held at home during the
        # replay (never excited, never fitted).
        "robotiq": ImplicitActuatorCfg(
            joint_names_expr=["robotiq_.*_joint"],
            effort_limit_sim=5.0,
            velocity_limit_sim=2.0,
            stiffness=200.0,
            damping=10.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

##
# Physics preset — PhysX only (see module docstring).
##


@configclass
class DroidSysidPhysicsCfg(PresetCfg):
    physx: PhysxCfg = PhysxCfg()

    default = physx


##
# Scene
##


@configclass
class DroidSysidSceneCfg(InteractiveSceneCfg):
    """Minimal scene: fixed-base droid and a light. No ground — nothing touches it."""

    robot: ArticulationCfg = DROID_SYSID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


##
# Environment
##


@configclass
class DroidSysIdCfg(SysIdCfg):
    robot_name: str = "droid_franka_robotiq"
    joint_order: list[str] = DROID_SYSID_JOINT_ORDER
    sim_joint_name_map: dict[str, str] = DROID_SIM_JOINT_NAME_MAP
    # Same compensated-plant reasoning as the FR3 task; also matches dextrah's
    # DROID_CFG (disable_gravity on the arm bodies).
    zero_gravity: bool = True


@configclass
class DroidSysIdEnvCfg(ManagerBasedRLEnvCfg):
    """CMA-ES sysid env for the droid articulation; driven by scripts/sysid/fit.py."""

    scene: DroidSysidSceneCfg = DroidSysidSceneCfg(num_envs=256, env_spacing=1.5)
    observations: SysidObservationsCfg = SysidObservationsCfg()
    actions: SysidActionsCfg = SysidActionsCfg()
    rewards: SysidRewardsCfg = SysidRewardsCfg()
    events: SysidEventCfg = SysidEventCfg()
    terminations: SysidTerminationsCfg = SysidTerminationsCfg()

    sysid: DroidSysIdCfg = DroidSysIdCfg()

    def __post_init__(self) -> None:
        # 1 kHz physics with the command hold reproduced via decimation — same
        # timing model as the FR3 task; fit.py re-derives both from the dataset.
        self.decimation = 5
        self.sim.dt = 0.001
        self.sim.render_interval = self.decimation
        # fit.py overrides this from the trajectory length.
        self.episode_length_s = 120.0
        self.sim.physics = DroidSysidPhysicsCfg()
        if self.sysid.zero_gravity:
            self.sim.gravity = (0.0, 0.0, 0.0)


##
# Corrected-plant variant (see SIM2REAL_FINDINGS.md)
##

# MuJoCo menagerie franka_fr3 values — the community-calibrated reference for
# the real FR3's reflected rotor inertia and joint friction.
MENAGERIE_ARMATURE = {"panda_joint[1-4]": 0.195, "panda_joint[5-7]": 0.074}
MENAGERIE_FRICTION = {
    "panda_joint[1-4]": 1.137,
    "panda_joint5": 0.763,
    "panda_joint6": 0.44,
    "panda_joint7": 0.248,
}


@configclass
class DroidCorrectedSysIdCfg(DroidSysIdCfg):
    robot_name: str = "droid_franka_robotiq_corrected"


@configclass
class DroidCorrectedSysIdEnvCfg(DroidSysIdEnvCfg):
    """Droid sysid env on the physically-corrected plant.

    Three deltas from the as-is task, all findings-driven:
    - phantom-mass fix: virtual gripper frames get a tiny authored mass via the
      overlay layer, suppressing the 1 kg-per-body runtime fallback (~ -7 kg);
    - menagerie FR3 armature on the arm joints;
    - menagerie FR3 per-joint friction instead of zero.
    """

    sysid: DroidCorrectedSysIdCfg = DroidCorrectedSysIdCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        robot = self.scene.robot
        robot.spawn.usd_path = _materialize_corrected_usd()
        robot.actuators["panda_arm"].armature = dict(MENAGERIE_ARMATURE)
        robot.actuators["panda_arm"].friction = dict(MENAGERIE_FRICTION)


@configclass
class DroidFr3SysIdCfg(DroidSysIdCfg):
    robot_name: str = "droid_fr3_robotiq"


@configclass
class DroidFr3SysIdEnvCfg(DroidCorrectedSysIdEnvCfg):
    """Corrected plant plus the Panda -> FR3 conversion (link masses, joint limits)."""

    sysid: DroidFr3SysIdCfg = DroidFr3SysIdCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.robot.spawn.usd_path = _materialize_corrected_usd(enable_fr3_conversion=True)
