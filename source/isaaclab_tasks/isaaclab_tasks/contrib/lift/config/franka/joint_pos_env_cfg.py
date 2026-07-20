# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.sim.schemas import MujocoRigidBodyPropertiesCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import CollisionPropertiesCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg
from isaaclab_contrib.deformable.newton_manager_cfg import VBDSolverCfg

from isaaclab_tasks.contrib.lift import mdp
from isaaclab_tasks.contrib.lift.lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip


@configclass
class FrankaCubeLiftPhysicsCfg(LiftPhysicsCfg):
    """Physics presets for Franka cube lifting."""

    newton_mjwarp_vbd_proxy: NewtonCfg = NewtonCfg(
        solver_cfg=CouplerProxyCfg(
            entries=[
                CouplerEntryCfg(
                    name="rigid",
                    solver_cfg=MJWarpSolverCfg(cone="elliptic", ls_iterations=20, integrator="implicitfast"),
                    bodies=[r"/World/envs/env_.*/Robot", r"/World/envs/env_.*/Table"],
                ),
                CouplerEntryCfg(
                    name="object",
                    solver_cfg=VBDSolverCfg(iterations=10),
                    bodies=[r"/World/envs/env_.*/Object"],
                    include_static_shapes=True,
                ),
            ],
            proxies=[
                CouplerProxyMappingCfg(
                    source="rigid",
                    destination="object",
                    bodies=[
                        r"/World/envs/env_.*/Robot/panda_hand",
                        r"/World/envs/env_.*/Robot/panda_(left|right)finger",
                        r"/World/envs/env_.*/Table",
                    ],
                )
            ],
            iterations=1,
        ),
        num_substeps=2,
    )


@configclass
class FrankaCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.sim.physics = FrankaCubeLiftPhysicsCfg()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.rigid_props = preset(
            default=MujocoRigidBodyPropertiesCfg(disable_gravity=False, gravcomp=1.0),
            physx=self.scene.robot.spawn.rigid_props.replace(disable_gravity=True),
            newton_mjwarp=MujocoRigidBodyPropertiesCfg(disable_gravity=False, gravcomp=1.0),
        )

        self.scene.table = preset(
            default=self.scene.table,
            newton_mjwarp_vbd_proxy=ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(0.5, 0.0, -0.525), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
                ),
                spawn=sim_utils.CuboidCfg(
                    size=(1.3, 0.9, 1.05),
                    collision_props=CollisionPropertiesCfg(),
                    rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True),
                ),
                actuators={},
                articulation_root_prim_path="",
            ),
        )
        self.rewards.reaching_object.params["object_cfg"] = SceneEntityCfg("object", body_names="Object")
        self.rewards.object_goal_tracking_delta.params["object_cfg"] = SceneEntityCfg("object", body_names="Object")
        self.rewards.object_goal_tracking.params["object_cfg"] = SceneEntityCfg("object", body_names="Object")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.04
        )
        self.actions.gripper_action = mdp.JointPositionToLimitsActionCfg(
            asset_name="robot", joint_names=["panda_finger.*"], rescale_to_limits=True
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "panda_hand"

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[0, 0, 0, 1]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
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
