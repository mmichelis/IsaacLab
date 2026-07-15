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
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import NewtonDeformableBodyMaterialCfg
from isaaclab_visualizers.newton import NewtonVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.assets.deformable_object import DeformableObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import CollisionPropertiesCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.coupling import (
    CouplerAdmmCfg,
    CouplerEntryCfg,
    CouplerProxyCfg,
    CouplerProxyMappingCfg,
)
from isaaclab_contrib.deformable.newton_manager_cfg import NewtonModelCfg, VBDSolverCfg

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

        # Camera pose (lowered eye framing the tabletop workspace).
        eye, lookat = (2.75, -2.0, 1.0), (0.5, 0.0, -0.5)
        self.viewer.eye = eye
        self.viewer.lookat = lookat
        self.sim.visualizer_cfgs = [NewtonVisualizerCfg(eye=eye, lookat=lookat, window_width=1920, window_height=1080)]
        self.video_recorder.window_width = 1920
        self.video_recorder.window_height = 1080


@configclass
class FrankaCubeLiftMjwarpEnvCfg(FrankaCubeLiftRigidEnvCfg):
    """Same scene as the rigid baseline, simulated with the pure mjwarp Newton solver."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.rigid_props = sim_utils.MujocoRigidBodyPropertiesCfg(gravcomp=1.0)

        # Drive the arm with joint position deltas (added to the current joint positions each
        # step) instead of absolute position commands.
        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.05
        )
        # Continuous gripper control: action in [-1, 1] maps to the finger joint limits
        # (closed to open) instead of a binary open/close command.
        self.actions.gripper_action = mdp.JointPositionToLimitsActionCfg(
            asset_name="robot", joint_names=["panda_finger.*"], rescale_to_limits=True
        )

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
        )


@configclass
class FrankaCubeLiftMjwarpGraspEnvCfg(FrankaCubeLiftMjwarpEnvCfg):
    """Pure-mjwarp cube lift with a bilateral contact reward for grasping."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.gripper_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_.*finger",
            update_period=0.0,
            history_length=1,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )
        self.rewards.grasping_object = RewTerm(
            func=mdp.object_is_grasped,
            weight=5.0,
            params={"force_threshold": 0.1, "sensor_cfg": SceneEntityCfg("gripper_contact")},
        )


@configclass
class FrankaCubeLiftProxyEnvCfg(FrankaCubeLiftMjwarpEnvCfg):
    """Same scene as the mjwarp variant, but the object is coupled to a VBD proxy solver.

    The robot and kinematic table are simulated with mjwarp while the DexCube is handed
    to VBD. The table and gripper are exposed as proxies so both solvers resolve their
    respective contacts.

    """

    def __post_init__(self):
        super().__post_init__()

        self.sim.physics = NewtonCfg(
            solver_cfg=CouplerProxyCfg(
                scene_cfg=self.scene,
                entries=[
                    CouplerEntryCfg(
                        name="rigid",
                        solver_cfg=MJWarpSolverCfg(
                            cone="elliptic",
                            ls_iterations=20,
                            integrator="implicitfast",
                        ),
                        bodies=[SceneEntityCfg("robot"), SceneEntityCfg("table")],
                    ),
                    CouplerEntryCfg(
                        name="object",
                        solver_cfg=VBDSolverCfg(iterations=10),
                        bodies=[SceneEntityCfg("object")],
                        include_static_shapes=True,
                    ),
                ],
                proxies=[
                    CouplerProxyMappingCfg(
                        source="rigid",
                        destination="object",
                        bodies=[
                            SceneEntityCfg(
                                "robot",
                                body_names=["panda_hand", "panda_(left|right)finger"],
                            ),
                            SceneEntityCfg("table"),
                        ],
                    )
                ],
                iterations=1,
            ),
            num_substeps=2,
        )
        self.rewards.object_goal_tracking_delta.params["object_cfg"] = SceneEntityCfg("object", body_names="Object")


@configclass
class FrankaCubeLiftDeformableProxyEnvCfg(FrankaCubeLiftProxyEnvCfg):
    """Proxy-coupled variant where the cube is a volumetric deformable body instead of rigid.

    Mirrors :class:`FrankaCubeLiftProxyEnvCfg` (mjwarp robot source, proxy gripper coupling, VBD
    destination), but the rigid DexCube is replaced by a solid VBD deformable cuboid of the same
    edge length (0.048 m) and a stiff material (Young's modulus 1e8 Pa). Since the object is now a
    particle deformable it auto-routes to the VBD solver, so the coupling needs no explicit
    destination body selector. The object observation becomes the mean of the cube's vertices and
    the reset re-seeds the nodal state instead of a rigid body pose.
    """

    def __post_init__(self):
        super().__post_init__()

        self.rewards.object_goal_tracking_delta.params["object_cfg"] = SceneEntityCfg("object")

        from isaaclab.managers import EventTermCfg as EventTerm
        from isaaclab.managers import ObservationTermCfg as ObsTerm

        # Replace the rigid DexCube with a solid VBD deformable cuboid of the same footprint
        YOUNGS_MODULUS = 5.0e6  # [Pa]
        POISSONS_RATIO = 0.45
        self.scene.object = DeformableObjectCfg(
            prim_path="/World/envs/env_.*/Object",
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.055)),
            spawn=sim_utils.MeshCuboidCfg(
                size=(0.048, 0.048, 0.048),
                deformable_props=NewtonDeformableBodyPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.6, 0.9)),
                physics_material=NewtonDeformableBodyMaterialCfg(
                    density=1.0,
                    k_mu=YOUNGS_MODULUS / (2.0 * (1.0 + POISSONS_RATIO)),
                    k_lambda=YOUNGS_MODULUS * POISSONS_RATIO / ((1.0 + POISSONS_RATIO) * (1.0 - 2.0 * POISSONS_RATIO)),
                    particle_radius=0.002,
                ),
            ),
        )

        self.sim.physics = NewtonCfg(
            solver_cfg=CouplerProxyCfg(
                scene_cfg=self.scene,
                entries=[
                    CouplerEntryCfg(
                        name="rigid",
                        solver_cfg=MJWarpSolverCfg(
                            cone="elliptic",
                            ls_iterations=20,
                            integrator="implicitfast",
                        ),
                        bodies=[SceneEntityCfg("robot"), SceneEntityCfg("table")],
                    ),
                    CouplerEntryCfg(
                        name="soft",
                        solver_cfg=VBDSolverCfg(iterations=10),
                        all_particles=True,
                        include_static_shapes=True,
                    ),
                ],
                proxies=[
                    CouplerProxyMappingCfg(
                        source="rigid",
                        destination="soft",
                        bodies=[
                            SceneEntityCfg(
                                "robot",
                                body_names=["panda_hand", "panda_(left|right)finger"],
                            ),
                            SceneEntityCfg("table"),
                        ],
                    )
                ],
                iterations=8,
                model_cfg=NewtonModelCfg(
                    soft_contact_ke=1e4,
                    soft_contact_kd=1e-1,
                    soft_contact_mu=5.0,
                ),
            ),
            default_shape_cfg=NewtonShapeCfg(ke=1e4, kd=1e-1, mu=5.0),
            num_substeps=2,
        )

        # Object observation: mean of the deformable cube's vertices in the robot root frame.
        self.observations.policy.object_position = ObsTerm(
            func=mdp.deformable_com_in_robot_root_frame,
            params={"asset_cfg": SceneEntityCfg("object")},
        )
        # Reconstruct a rigid orientation for the deformable cube by fitting a coordinate frame to
        # a few of its vertices (Kabsch), so the same policy that expects object_orientation works.
        self.observations.policy.object_orientation = ObsTerm(
            func=mdp.DeformableOrientationInRobotRootFrame,
            params={"asset_cfg": SceneEntityCfg("object"), "num_points": 8},
        )

        # Re-seed the deformable node cloud using the inherited x/y sampling range.
        pose_range = self.events.reset_object_position.params["pose_range"]
        self.events.reset_object_position = EventTerm(
            func=mdp.reset_nodal_state_uniform,
            mode="reset",
            params={
                "position_range": {
                    "x": pose_range.get("x", (0.0, 0.0)),
                    "y": pose_range.get("y", (0.0, 0.0)),
                    "z": (0.0, 0.0),
                },
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object"),
            },
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

        self.sim.physics = NewtonCfg(
            solver_cfg=CouplerAdmmCfg(
                scene_cfg=self.scene,
                entries=[
                    CouplerEntryCfg(
                        name="rigid",
                        solver_cfg=MJWarpSolverCfg(
                            cone="elliptic",
                            ls_iterations=20,
                            integrator="implicitfast",
                            use_mujoco_contacts=False,
                        ),
                        bodies=[SceneEntityCfg("robot")],
                    ),
                    CouplerEntryCfg(
                        name="object",
                        solver_cfg=VBDSolverCfg(iterations=20),
                        bodies=[SceneEntityCfg("object")],
                        include_static_shapes=True,
                    ),
                ],
                contact_pairs=[("rigid", "object")],
                rigid_contact_matching="sticky",
                iterations=8,
                rho=5e2,
                gamma=0.01,
                baumgarte=0.0,
            ),
            default_shape_cfg=NewtonShapeCfg(ke=8e3),
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

        # Soften the arm PD so the IK target is tracked gently
        # for actuator_name in ("panda_shoulder", "panda_forearm"):
        #     self.scene.robot.actuators[actuator_name].stiffness = 100.0
        #     self.scene.robot.actuators[actuator_name].damping = 40.0

        # Contact sensor on the gripper fingers, filtered to the cube, to read the
        # gripper-on-cube normal contact forces (per finger).
        self.scene.gripper_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_.*finger",
            update_period=0.0,
            history_length=1,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )
