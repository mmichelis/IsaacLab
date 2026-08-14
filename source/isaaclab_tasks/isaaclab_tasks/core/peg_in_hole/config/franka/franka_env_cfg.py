# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.sim.schemas import MujocoJointCfg

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.peg_in_hole import mdp
from isaaclab_tasks.core.peg_in_hole.peg_in_hole_env_cfg import PegInHoleEnvCfg
from isaaclab_tasks.utils import PresetCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_MENAGERIE_CFG  # isort: skip

LEFT_FINGER_CONTACT_SENSOR = "panda_leftfinger_object_s"
RIGHT_FINGER_CONTACT_SENSOR = "panda_rightfinger_object_s"


@configclass
class _JointActionsCfg:
    """Relative joint-position arm targets and a limit-rescaled gripper target."""

    arm_action = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=["panda_joint.*"], scale=0.05)
    gripper_action = mdp.JointPositionToLimitsActionCfg(
        asset_name="robot", joint_names=["panda_finger_joint1"], rescale_to_limits=True
    )


@configclass
class _IkActionsCfg:
    """Absolute end-effector pose targets and a binary gripper target."""

    arm_action = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.6},
        ),
        body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger_joint1"],
        open_command_expr={"panda_finger_joint1": 0.04},
        close_command_expr={"panda_finger_joint1": 0.015},
    )


@configclass
class ActionsCfg(PresetCfg):
    """Franka action-space presets."""

    joint: _JointActionsCfg = _JointActionsCfg()
    ik: _IkActionsCfg = _IkActionsCfg()
    default = joint


@configclass
class FrankaPegInHoleEnvCfg(PegInHoleEnvCfg):
    actions: ActionsCfg = ActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = FRANKA_PANDA_MENAGERIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.joint_drive_props = [MujocoJointCfg(actuatorgravcomp=True)]
        self.scene.robot.init_state.joint_pos["panda_joint2"] = 0.0
        self.scene.robot.spawn.activate_contact_sensors = True
        for finger_name, sensor_name in (
            ("panda_leftfinger", LEFT_FINGER_CONTACT_SENSOR),
            ("panda_rightfinger", RIGHT_FINGER_CONTACT_SENSOR),
        ):
            setattr(
                self.scene,
                sensor_name,
                ContactSensorCfg(
                    prim_path=(
                        "{ENV_REGEX_NS}/Robot/Geometry/panda_link0/panda_link1/panda_link2/"
                        "panda_link3/panda_link4/panda_link5/panda_link6/panda_link7/"
                        f"panda_hand/{finger_name}"
                    ),
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                ),
            )

        self.rewards.reaching_object.params["robot_cfg"] = SceneEntityCfg(
            "robot", body_names=["panda_leftfinger", "panda_rightfinger"]
        )
        for term in (self.rewards.goal_distance, self.rewards.success, self.rewards.success_bonus):
            term.params.update(
                {
                    "contact_threshold": 0.01,
                    "thumb_name": LEFT_FINGER_CONTACT_SENSOR,
                    "finger_names": [RIGHT_FINGER_CONTACT_SENSOR],
                }
            )

        self.scene.robot.actuators = {
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
