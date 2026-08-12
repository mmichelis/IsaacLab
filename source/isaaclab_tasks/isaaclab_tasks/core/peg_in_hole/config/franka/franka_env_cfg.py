# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import CollisionPropertiesCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.peg_in_hole import mdp
from isaaclab_tasks.core.peg_in_hole.peg_in_hole_env_cfg import PegInHoleEnvCfg
from isaaclab_tasks.utils import preset

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_MENAGERIE_CFG  # isort: skip


@configclass
class FrankaPegInHoleEnvCfg(PegInHoleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = FRANKA_PANDA_MENAGERIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.robot.actuators = {
            # inspired by libfranka's joint_impedance_control.cpp
            "panda_arm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-7]"],
                effort_limit_sim={"panda_joint[1-4]": 87.0, "panda_joint[5-7]": 12.0},
                velocity_limit_sim={"panda_joint[1-4]": 2.175, "panda_joint[5-7]": 2.61},
                stiffness={
                    "panda_joint[1-4]": 600.0,
                    "panda_joint5": 250.0,
                    "panda_joint6": 150.0,
                    "panda_joint7": 50.0,
                },
                damping={
                    "panda_joint[1-4]": 50.0,
                    "panda_joint5": 30.0,
                    "panda_joint6": 25.0,
                    "panda_joint7": 15.0,
                },
                armature={
                    "panda_joint[1-2]": 0.6057,
                    "panda_joint[3-4]": 0.4625,
                    "panda_joint[5-7]": 0.2055,
                },
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint1"],
                effort_limit_sim=70.0,
                velocity_limit=0.2,
                velocity_limit_sim=2.0,
                stiffness=350.0,
                damping=175.0,
                armature=0.1,
            ),
            "panda_finger2_passive": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint2"],
                effort_limit_sim=1.0,
                velocity_limit=0.2,
                velocity_limit_sim=2.0,
                stiffness=0.0,
                damping=0.0,
                armature=0.1,
            ),
        }

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.075
        )
        self.actions.gripper_action = mdp.JointPositionToLimitsActionCfg(
            asset_name="robot", joint_names=["panda_finger.*"], rescale_to_limits=True
        )
        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )
