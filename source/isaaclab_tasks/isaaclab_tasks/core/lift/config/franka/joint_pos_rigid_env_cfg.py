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

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg
from isaaclab_newton.sensors.contact_sensor import ContactSensorCfg
from isaaclab_visualizers.newton import NewtonVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import CollisionPropertiesCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.coupling import CoupledAdmmSolverCfg, CoupledProxySolverCfg
from isaaclab_contrib.deformable.newton_manager_cfg import (
    CoupledNewtonCfg,
    NewtonModelCfg,
    VBDSolverCfg,
)

from isaaclab_tasks.core.lift import mdp
from isaaclab_tasks.core.lift.lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


@configclass
class FrankaCubeLiftRigidEnvCfg(LiftEnvCfg):
    """All-rigid Franka cube lift on the default PhysX backend (the comparison baseline)."""

    def __post_init__(self):
        # post init of parent (sets PhysX backend, dt, decimation)
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/envs/env_.*/Robot")

        # Replace the world-welded USD table with a static cuboid collider (same footprint and
        # top height as the mjwarp variant). PhysX positions per-env static geoms via the cloner.
        self.scene.table = AssetBaseCfg(
            prim_path="/World/envs/env_.*/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, -0.525), rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=sim_utils.CuboidCfg(
                # 90 deg z rotation baked into the footprint (swapped x/y) so it holds regardless
                # of how the static per-env table prim is positioned.
                size=(1.3, 0.9, 1.05),
                collision_props=CollisionPropertiesCfg(),
            ),
        )

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.015},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "panda_hand"

        # Set rigid Cube as object.
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
        self.scene.robot.spawn.rigid_props = sim_utils.MujocoRigidBodyPropertiesCfg(gravcomp=1.0)

        self.scene.robot.actuators["panda_shoulder"].stiffness = 1000.0
        self.scene.robot.actuators["panda_shoulder"].damping = 60.0
        self.scene.robot.actuators["panda_shoulder"].armature = 0.1
        self.scene.robot.actuators["panda_forearm"].stiffness = 1000.0
        self.scene.robot.actuators["panda_forearm"].damping = 60.0
        self.scene.robot.actuators["panda_forearm"].armature = 0.1
        self.scene.robot.actuators["panda_hand"].stiffness = 350.0
        self.scene.robot.actuators["panda_hand"].damping = 20.0
        self.scene.robot.actuators["panda_hand"].armature = 0.1

        # Drive the arm with joint position deltas (added to the current joint positions each
        # step) instead of absolute position commands.
        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.05
        )

        # Camera pose (lowered eye framing the tabletop workspace). When Kit is running,
        # ViewportCameraController pushes ViewerCfg.eye/lookat onto the Newton viewer via
        # sim.set_camera_view(), overriding NewtonVisualizerCfg. So ViewerCfg is the master pose
        # (origin_type "world" -> absolute world coords); the Newton cfg carries the same values
        # for the kitless path and seeds the --video recorder. Record at 1080p.
        eye, lookat = (3.5, -3.0, 1.5), (0.4, 0.0, -0.25)
        self.viewer.eye = eye
        self.viewer.lookat = lookat
        self.sim.visualizer_cfgs = [NewtonVisualizerCfg(eye=eye, lookat=lookat, window_width=1920, window_height=1080)]
        self.video_recorder.window_width = 1920
        self.video_recorder.window_height = 1080

        # mjwarp does not position per-env static geoms, so re-create the table as a jointless
        # articulation, which mjwarp positions per-env.
        self.scene.table = ArticulationCfg(
            prim_path="/World/envs/env_.*/Table",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.5, 0.0, -0.525), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
            ),
            spawn=sim_utils.CuboidCfg(
                # 90 deg z rotation baked into the footprint (swapped x/y) to match the rigid table.
                size=(1.3, 0.9, 1.05),
                collision_props=CollisionPropertiesCfg(),
                rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True),
            ),
            actuators={},
            articulation_root_prim_path="",
        )

        # Pure mjwarp solver (the coupled-proxy variant lives in FrankaCubeLiftProxyEnvCfg).
        self.sim.physics = NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                cone="elliptic",
                integrator="implicitfast",
                use_mujoco_contacts=False,
            ),
            num_substeps=2,
            collision_decimation=1,
            default_shape_cfg=NewtonShapeCfg(ke=4e4, kd=400.0),
        )


@configclass
class FrankaCubeLiftProxyEnvCfg(FrankaCubeLiftMjwarpEnvCfg):
    """Same scene as the mjwarp variant, but the object is coupled to a VBD proxy solver.

    The robot is simulated with mjwarp (source solver) while the DexCube and the static
    table are handed to a VBD (destination) solver; a proxy coupler exchanges contacts
    between the two so a soft/deformable-style object can interact with the rigid mjwarp
    scene. The table is a body-less static collider owned by VBD as world geometry.

    The cube's reset is overridden: since it is owned by the VBD solver, IsaacLab's default
    rigid-body reset never reaches its state, so we re-seed the VBD body state instead.
    """

    def __post_init__(self):
        super().__post_init__()

        # Replace the inherited kinematic mjwarp table with a body-less static collider. With no
        # rigid body its collision shape gets body == -1, so the coupled manager auto-routes it into
        # the VBD (dst) solver as static world geometry (the cube rests on it there). The robot no
        # longer collides with the table in mjwarp.
        self.scene.table = AssetBaseCfg(
            prim_path="/World/envs/env_.*/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, -0.525), rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=sim_utils.CuboidCfg(
                size=(1.3, 0.9, 1.05),
                collision_props=CollisionPropertiesCfg(),
            ),
        )

        # The DexCube now lives on the VBD (dst) side of the coupled solver, so the inherited
        # rigid-body reset (reset_root_state_uniform) never touches its state. Re-seed the VBD
        # body_q/joint_q (and AVBD companions) via the soft-object reset path, keeping the same
        # sampling range, and clear the proxy teleport velocity left by the robot-joint reset.
        from isaaclab.managers import EventTermCfg as EventTerm
        from isaaclab.managers import SceneEntityCfg

        from isaaclab_tasks.core.lift.config.franka_soft import mdp as soft_mdp

        self.events.reset_object_position = EventTerm(
            func=soft_mdp.reset_rigid_body_uniform,
            mode="reset",
            params={
                "pose_range": self.events.reset_object_position.params["pose_range"],
                "asset_cfg": SceneEntityCfg("object"),
            },
        )
        # Runs after reset_all/reset_object_position: snaps each proxy body's body_q_prev onto
        # its post-reset pose so the coupler does not fling the cube from the arm teleport.
        self.events.reset_proxy_velocity = EventTerm(func=soft_mdp.reset_proxy_body_prev, mode="reset")

        self.sim.physics = CoupledNewtonCfg(
            solver_cfg=CoupledProxySolverCfg(
                src_solver_cfg=MJWarpSolverCfg(
                    cone="elliptic",
                    ls_iterations=20,
                    integrator="implicitfast",
                ),
                dst_solver_cfg=VBDSolverCfg(
                    iterations=10,
                    rigid_avbd_beta=0.0,
                ),
                src_bodies=["/World/envs/env_.*/Robot"],
                dst_bodies=["/World/envs/env_.*/Object"],
                proxy_bodies=[
                    "/World/envs/env_.*/Robot/panda_hand",
                    "/World/envs/env_.*/Robot/panda_(left|right)finger",
                ],
                proxy_iterations=5,
            ),
            model_cfg=NewtonModelCfg(
                shape_material_ke=8e3,
            ),
            num_substeps=2,
        )


@configclass
class FrankaCubeLiftAdmmEnvCfg(FrankaCubeLiftMjwarpEnvCfg):
    """Same scene as the mjwarp variant, but the object is coupled to a VBD solver via ADMM.

    The counterpart to :class:`FrankaCubeLiftProxyEnvCfg`: the robot is simulated with mjwarp
    (source solver) and the DexCube plus the static table are handed to a VBD (destination)
    solver, but here the two are coupled by the linearized ADMM solver instead of the proxy
    coupler. A single source-to-destination contact pair transmits the grasp forces, so no
    proxy bodies are needed. The table is a body-less static collider owned by VBD as world
    geometry.

    The cube's reset is overridden: since it is owned by the VBD solver, IsaacLab's default
    rigid-body reset never reaches its state, so we re-seed the VBD body state instead.
    """

    def __post_init__(self):
        super().__post_init__()

        # Replace the inherited kinematic mjwarp table with a body-less static collider. With no
        # rigid body its collision shape gets body == -1, so the coupled manager auto-routes it into
        # the VBD (dst) solver as static world geometry (the cube rests on it there). The robot no
        # longer collides with the table in mjwarp.
        self.scene.table = AssetBaseCfg(
            prim_path="/World/envs/env_.*/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, -0.525), rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=sim_utils.CuboidCfg(
                size=(1.3, 0.9, 1.05),
                collision_props=CollisionPropertiesCfg(),
            ),
        )

        # The DexCube now lives on the VBD (dst) side of the coupled solver, so the inherited
        # rigid-body reset (reset_root_state_uniform) never touches its state. Re-seed the VBD
        # body_q via the soft-object reset path, keeping the same sampling range. Unlike the proxy
        # variant there are no proxy bodies, so no proxy-velocity reset is needed.
        from isaaclab.managers import EventTermCfg as EventTerm
        from isaaclab.managers import SceneEntityCfg

        from isaaclab_tasks.core.lift.config.franka_soft import mdp as soft_mdp

        self.events.reset_object_position = EventTerm(
            func=soft_mdp.reset_rigid_body_uniform,
            mode="reset",
            params={
                "pose_range": self.events.reset_object_position.params["pose_range"],
                "asset_cfg": SceneEntityCfg("object"),
            },
        )

        self.sim.physics = CoupledNewtonCfg(
            solver_cfg=CoupledAdmmSolverCfg(
                src_solver_cfg=MJWarpSolverCfg(
                    cone="elliptic",
                    ls_iterations=20,
                    integrator="implicitfast",
                    use_mujoco_contacts=False,
                ),
                dst_solver_cfg=VBDSolverCfg(
                    iterations=20,
                    rigid_avbd_beta=0.0,
                ),
                src_bodies=["/World/envs/env_.*/Robot"],
                dst_bodies=["/World/envs/env_.*/Object"],
                rigid_contact_matching="sticky",
                iterations=8,
                rho=5e2,
                gamma=0.01,
                baumgarte=0.0,
            ),
            model_cfg=NewtonModelCfg(
                shape_material_ke=8e3,
            ),
            num_substeps=2,
        )


@configclass
class FrankaCubeLiftMjwarpIkAbsEnvCfg(FrankaCubeLiftMjwarpEnvCfg):
    """Pure-mjwarp cube lift driven by task-space absolute-pose IK.

    Same scene and solver as :class:`FrankaCubeLiftMjwarpEnvCfg`, but the arm is
    commanded with a differential IK action so the task-space pick-and-lift state
    machine (``scripts/environments/state_machine/lift_franka_soft.py``) can drive it.
    """

    def __post_init__(self):
        super().__post_init__()

        # Swap joint-position control for 7-dim absolute EE pose via differential IK.
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )

        # Restore the binary gripper: the state machine emits a single open/close command.
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.015},
        )

        # Soften the arm PD so the IK target is tracked gently
        for actuator_name in ("panda_shoulder", "panda_forearm"):
            self.scene.robot.actuators[actuator_name].stiffness = 100.0
            self.scene.robot.actuators[actuator_name].damping = 40.0

        # Contact sensor on the gripper fingers, filtered to the cube, to read the
        # gripper-on-cube normal contact forces (per finger).
        self.scene.gripper_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_.*finger",
            update_period=0.0,
            history_length=1,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )
