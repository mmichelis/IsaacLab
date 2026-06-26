# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene-identical Franka cube-lift configs for PhysX vs. pure mjwarp comparison.

The rigid base sets up an all-rigid scene (robot, cuboid table, rigid DexCube) and
inherits the PhysX backend from :class:`LiftEnvCfg`. The mjwarp variant overrides
only the physics backend so the two are directly comparable for learning-curve
matching.
"""

import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.lift import mdp
from isaaclab_tasks.core.lift.config.franka_soft import mdp as soft_mdp
from isaaclab_tasks.core.lift.lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip


def clamp_object_linear_velocity(env, env_ids, asset_cfg: SceneEntityCfg, max_speed: float) -> None:
    """Clamp the object's linear speed each control step (mjwarp depenetration-cap analogue).

    Newton ignores PhysX's ``max_depenetration_velocity``, so the gripper can eject the cube at
    high velocity and a policy reward-hacks the lift term by punching it airborne. Capping the
    cube's linear speed stops the impulsive launch while leaving a slow, sustained grasped lift
    unaffected. Runs as an every-step interval event (``env_ids`` is None -> all envs).
    """
    asset = env.scene[asset_cfg.name]
    vel = asset.data.root_vel_w.torch.clone()  # [N, 6] = [lin (3), ang (3)]
    lin = vel[:, :3]
    speed = torch.linalg.vector_norm(lin, dim=-1, keepdim=True)
    scale = torch.clamp(max_speed / speed.clamp_min(1e-6), max=1.0)
    vel[:, :3] = lin * scale
    asset.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=torch.arange(env.num_envs, device=asset.device))


@configclass
class FrankaCubeLiftRigidEnvCfg(LiftEnvCfg):
    """All-rigid Franka cube lift on the default PhysX backend (the comparison baseline)."""

    def __post_init__(self):
        # post init of parent (sets PhysX backend, dt, decimation)
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot")

        self.scene.table = AssetBaseCfg(
            prim_path="/World/envs/env_.*/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.0, 0.0, 0.707, 0.707)),
            spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
        )

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

        # Shrink the command frame markers: the EE (current-pose) frame barely visible, and the
        # goal-pose frame half as thick as the default (0.1).
        self.commands.object_pose.current_pose_visualizer_cfg.markers["frame"].scale = (0.02, 0.02, 0.02)
        self.commands.object_pose.goal_pose_visualizer_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)

        # Set rigid Cube as object. The PhysX solver-iteration rigid_props below are honored by
        # PhysX and ignored by Newton; they are kept identical across both variants regardless.
        # A high-friction, zero-restitution contact material is set so the gripper can grip the
        # cube (Newton combines the two geoms' friction as the element-wise max, so this governs
        # the finger-cube contact). Both variants share it to keep the scene identical.
        self.scene.object = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0, dynamic_friction=1.0, restitution=0.0
                ),
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


@configclass
class FrankaCubeLiftMjwarpEnvCfg(FrankaCubeLiftRigidEnvCfg):
    """Same scene as the rigid baseline, simulated with the pure mjwarp Newton solver."""

    def __post_init__(self):
        super().__post_init__()

        # mjwarp's implicit PD drive is less damped than PhysX at the same gain, so the arm
        # overshoots more. Raise arm damping ~3x (4 -> 12) so the joint/EE motion ranges match
        # the PhysX baseline (gripper joints left as-is).
        for actuator_name in ("panda_shoulder", "panda_forearm"):
            self.scene.robot.actuators[actuator_name].damping = 16.0

        self.scene.table_collider = RigidObjectCfg(
            prim_path="/World/envs/env_.*/TableCollider",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.344, 0.0, -0.503)),
            spawn=sim_utils.CuboidCfg(
                size=(1.28, 0.91, 1.00),
                visible=False,
                collision_props=sim_utils.CollisionPropertiesCfg(),
                # Kinematic so mjwarp welds it as a per-env body (not a static world geom).
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            ),
        )

        # Pure mjwarp Newton backend (no coupling / VBD). Solver settings mirror the coupled
        # config's mjwarp source solver. Stiffer contacts (ke 2.5e3 -> 4e4, kd 100 -> 400, ~solref
        # 0.005s critically damped) stop the arm/cube sinking into the table; num_substeps 8 -> 16
        # keeps the stiffer contacts stable.
        self.sim.physics = NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                cone="elliptic",
                ls_parallel=True,
                ls_iterations=20,
                integrator="implicitfast",
            ),
            num_substeps=16,
            default_shape_cfg=NewtonShapeCfg(ke=4e4, kd=400.0),
        )

        # Cap the cube's linear speed every control step so the gripper cannot launch it (mjwarp
        # has no PhysX-style depenetration-velocity cap). Prevents the punch reward-hack while a
        # slow grasped lift is unaffected. interval_range_s=(0, 0) fires every step.
        self.events.clamp_object_velocity = EventTerm(
            func=clamp_object_linear_velocity,
            mode="interval",
            interval_range_s=(0.0, 0.0),
            params={"asset_cfg": SceneEntityCfg("object"), "max_speed": 1.0},
        )

        # Reset envs whose coupled solve destabilizes: any robot/cube body exceeding 1e2 m/s.
        self.terminations.body_velocity_out_of_bounds = DoneTerm(
            func=soft_mdp.body_velocity_out_of_bounds,
            params={"max_velocity": 1e2},
        )

        self.terminations.object_out_of_bounds = DoneTerm(
            func=soft_mdp.object_outside_table_bounds,
            params={
                "x_bounds": (0.0, 0.9),
                "y_bounds": (-0.5, 0.5),
                "z_bounds": (-0.1, 2.0),
                "asset_cfg": SceneEntityCfg("object"),
            },
        )
