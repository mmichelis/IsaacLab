# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.coupling import CoupledProxySolverCfg
from isaaclab_contrib.deformable.newton_manager_cfg import CoupledNewtonCfg, NewtonModelCfg, VBDSolverCfg

from isaaclab_tasks.core.lift import mdp
from isaaclab_tasks.core.lift.config.franka_soft import mdp as soft_mdp
from isaaclab_tasks.core.lift.lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip


@configclass
class FrankaCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "panda_hand"

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Object",
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
            prim_path="/World/envs/env_.*/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="/World/envs/env_.*/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )

        # The object is a VBD body, so re-seed its VBD state on reset instead of writing the
        # rigid-body sim (reset_root_state_uniform would no-op on the solver side).
        # self.events.reset_object_position = EventTerm(
        #     func=soft_mdp.reset_rigid_body_uniform,
        #     mode="reset",
        #     params={
        #         "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
        #         "asset_cfg": SceneEntityCfg("object"),
        #     },
        # )
        # Proxy-coupled gripper: clear the teleport velocity left by the robot-joint reset, else the
        # fingers fling the object. Must run after reset_robot_joints / reset_all.
        # self.events.reset_proxy_velocity = EventTerm(func=soft_mdp.reset_proxy_body_prev, mode="reset")

        # self.sim.physics = CoupledNewtonCfg(
        #     scene_cfg=self.scene,
        #     solver_cfg=CoupledProxySolverCfg(
        #         src_solver_cfg=MJWarpSolverCfg(
        #             cone="elliptic",
        #             ls_parallel=True,
        #             ls_iterations=20,
        #             integrator="implicitfast",
        #         ),
        #         dst_solver_cfg=VBDSolverCfg(iterations=20, rigid_avbd_beta=1e3, rigid_contact_k_start=1e3),
        #         src_bodies=[SceneEntityCfg("robot")],
        #         dst_bodies=[SceneEntityCfg("object")],
        #         proxy_bodies=[
        #             SceneEntityCfg("robot", body_names=["panda_hand", "panda_(left|right)finger"]),
        #         ],
        #         # More relaxation passes tighten the proxy grip on the plug.
        #         proxy_iterations=4,
        #     ),
        #     model_cfg=NewtonModelCfg(
        #         shape_material_ke=1e5,
        #         shape_material_kd=1e-2,
        #         shape_material_mu=10.0,
        #     ),
        #     num_substeps=8,
        # )
        self.sim.physics = NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                cone="elliptic",
                ls_parallel=True,
                ls_iterations=20,
                integrator="implicitfast",
            ),
            num_substeps=8,
        )


@configclass
class FrankaCubeLiftEnvCfg_PLAY(FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
