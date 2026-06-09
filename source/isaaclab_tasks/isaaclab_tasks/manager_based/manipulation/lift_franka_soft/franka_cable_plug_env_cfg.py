# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Panda manipulating a Newton cable plug.

The cable is welded at one end to a kinematic anchor fixed above the tabletop
and at the other end to a rigid plug body. The RL task is to bring the plug to
a sampled target pose using the Franka end-effector.

This environment inherits aggressively from :class:`FrankaCableEnvCfg`: it
reuses the robot, actions, physics solver, and reward shape, and only adds the
two attachment bodies plus rewires reward and observation terms from the cable
to the plug.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING

import math
import torch
from isaaclab_newton.physics import MJWarpSolverCfg
from isaaclab_newton.sim.spawners.materials import NewtonCableMaterialCfg
from isaaclab_visualizers.kit.kit_visualizer_cfg import KitVisualizerCfg
from isaaclab_visualizers.newton.newton_visualizer_cfg import NewtonVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_from_angle_axis

from isaaclab_contrib.cable.cable_object_cfg import CableAttachmentCfg, CableObjectCfg
from isaaclab_contrib.coupling import CoupledAdmmSolverCfg
from isaaclab_contrib.deformable.newton_manager_cfg import (
    CoupledNewtonCfg,
    NewtonModelCfg,
    VBDSolverCfg,
)

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from . import mdp
from .franka_soft_env_cfg import FrankaSoftEnvCfg, _FrankaSoftSceneCfg

##
# Cable / attachment geometry constants
##

# Number of cable segments. Matches the parent cable env so the same physics tuning applies.
_NUM_POINTS = 28

# Per-segment length [m]. Matches the parent cable env.
_SEGMENT_LENGTH = 0.02

# Cable width [m].
_CABLE_WIDTH = 0.01

# Anchor pose in the env-local frame [m]. Positioned above the tabletop, in front of the robot.
_ANCHOR_POS = (0.15, 0.3, 0.2)

# Plug body parameters. Mass is the midpoint of the demo's [0.005, 0.05] kg range.
_PLUG_RADIUS = 0.01
_PLUG_HEIGHT = 0.04
_PLUG_MASS = 0.21

# Plug rest pose [m]: directly below the anchor at the cable's natural extent.
_PLUG_INIT_POS = (_ANCHOR_POS[0] + (_NUM_POINTS - 2) * _SEGMENT_LENGTH, _ANCHOR_POS[1], _ANCHOR_POS[2])

# Target-hole pose [m]: center of the socket. Placed in front of the plug's rest position;
# the socket opens along -x so the plug (oriented along x) can be inserted by pushing in +x.
_TARGET_HOLE_POS = (_PLUG_INIT_POS[0] - 0.15, _PLUG_INIT_POS[1] - 0.2, _PLUG_INIT_POS[2])
# Inner clear opening in the y-z plane [m]: slightly larger than the plug diameter (2*_PLUG_RADIUS).
_TARGET_HOLE_INNER = 0.03
# Wall thickness [m].
_TARGET_HOLE_WALL_THICKNESS = 0.003
# Socket depth along x [m]: matches the plug height so the plug can fully insert.
_TARGET_HOLE_DEPTH = _PLUG_HEIGHT


##
# Target-hole geometry: four independent kinematic cuboid walls forming a square
# socket. The opening faces -x; walls extend along +x by ``_TARGET_HOLE_DEPTH``.
# Each wall is offset from the socket center by ``_WALL_OFFSET`` along its axis.
# Modeled as four separate ``RigidObjectCfg`` entries — simpler than a custom
# compound-shape spawner, and the per-step cost of writing four kinematic poses
# instead of one is negligible.
##

_WALL_SPAN = _TARGET_HOLE_INNER + 2.0 * _TARGET_HOLE_WALL_THICKNESS
_WALL_OFFSET = (_TARGET_HOLE_INNER + _TARGET_HOLE_WALL_THICKNESS) / 2.0

# Per-wall size [m] (x, y, z). Top/bottom span y; left/right span z.
_WALL_SIZE_TB = (_TARGET_HOLE_DEPTH, _WALL_SPAN, _TARGET_HOLE_WALL_THICKNESS)
_WALL_SIZE_LR = (_TARGET_HOLE_DEPTH, _TARGET_HOLE_WALL_THICKNESS, _TARGET_HOLE_INNER)

# Per-wall offset from socket center in the socket frame [m].
_WALL_OFFSETS: dict[str, tuple[float, float, float]] = {
    "target_hole_top": (0.0, 0.0, _WALL_OFFSET),
    "target_hole_bottom": (0.0, 0.0, -_WALL_OFFSET),
    "target_hole_left": (0.0, -_WALL_OFFSET, 0.0),
    "target_hole_right": (0.0, _WALL_OFFSET, 0.0),
}


def _wall_cfg(size: tuple[float, float, float], offset: tuple[float, float, float], prim_name: str) -> RigidObjectCfg:
    """Kinematic cuboid wall offset from the socket center by ``offset``."""
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/{prim_name}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=tuple(c + o for c, o in zip(_TARGET_HOLE_POS, offset)),
        ),
    )


##
# Scene
##


@configclass
class _FrankaCablePlugSceneCfg(_FrankaSoftSceneCfg):
    """Scene for the MJWarp Franka environment grasping a rigid VBD body attached to a VBD cable."""

    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    anchor: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Anchor",
        spawn=sim_utils.SphereCfg(
            radius=0.02,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=_ANCHOR_POS),
    )

    # Square socket: four kinematic cuboid walls. Opening faces -x (toward the plug);
    # walls extend along +x by _TARGET_HOLE_DEPTH.
    target_hole_top: RigidObjectCfg = _wall_cfg(_WALL_SIZE_TB, _WALL_OFFSETS["target_hole_top"], "TargetHoleTop")
    target_hole_bottom: RigidObjectCfg = _wall_cfg(
        _WALL_SIZE_TB, _WALL_OFFSETS["target_hole_bottom"], "TargetHoleBottom"
    )
    target_hole_left: RigidObjectCfg = _wall_cfg(_WALL_SIZE_LR, _WALL_OFFSETS["target_hole_left"], "TargetHoleLeft")
    target_hole_right: RigidObjectCfg = _wall_cfg(_WALL_SIZE_LR, _WALL_OFFSETS["target_hole_right"], "TargetHoleRight")

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Plug",
        spawn=sim_utils.CylinderCfg(
            radius=_PLUG_RADIUS,
            height=_PLUG_HEIGHT,
            # spawn=sim_utils.CuboidCfg(
            #     size=(2*_PLUG_RADIUS, 2*_PLUG_RADIUS, _PLUG_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=_PLUG_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.2, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=_PLUG_INIT_POS,
            rot=quat_from_angle_axis(torch.tensor(torch.pi / 2.0), torch.tensor([0.0, 1.0, 0.0])),
        ),
    )

    cable: CableObjectCfg = CableObjectCfg(
        prim_path="/World/envs/env_.*/Cable",
        init_state=CableObjectCfg.InitialStateCfg(pos=_ANCHOR_POS),
        spawn=sim_utils.CableCfg(
            positions=[(i * _SEGMENT_LENGTH, 0.0, 0.0) for i in range(_NUM_POINTS)],
            width=_CABLE_WIDTH,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.85, 0.1)),
            physics_material=NewtonCableMaterialCfg(
                stretch_stiffness=1.0e3,
                stretch_damping=1.0e-1,
                bend_stiffness=5.0e-2,
                bend_damping=5.0e-4,
                density=1.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        attachments=[
            CableAttachmentCfg(
                target_prim_path="/World/envs/env_.*/Anchor",
                cable_anchor=0,
            ),
            CableAttachmentCfg(
                target_prim_path="/World/envs/env_.*/Plug",
                cable_anchor=-2,
            ),
        ],
    )

    def __post_init__(self):
        super().__post_init__()
        self.robot.spawn.rigid_props.disable_gravity = True
        self.robot.spawn.rigid_props = sim_utils.MujocoRigidBodyPropertiesCfg(gravcomp=1.0)

        # increase franka gripper stiffness
        self.robot.actuators["panda_hand"].effort_limit_sim = 1500.0
        self.robot.actuators["panda_hand"].stiffness = 1500.0
        self.robot.actuators["panda_hand"].damping = 150.0


##
# MDP overrides
##


# Lazily-built subclass of :class:`~isaaclab.envs.mdp.commands.pose_command.UniformPoseCommand`
# that drags one or more rigid scene assets along with the sampled command.
#
# The class itself is constructed on first instantiation (env build time, after
# ``SimulationApp`` is up) because importing ``UniformPoseCommand`` at module
# load pulls in :class:`~isaaclab.assets.Articulation`, which loads
# ``omni.kit.app`` C extensions and corrupts Kit's plugin loader.
_uniform_pose_command_with_targets_cls: type | None = None


def _make_uniform_pose_command_with_targets(cfg, env):
    """Build (once) and instantiate the multi-target pose command class."""
    global _uniform_pose_command_with_targets_cls
    if _uniform_pose_command_with_targets_cls is None:
        from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand
        from isaaclab.utils.math import combine_frame_transforms, quat_apply

        class _UniformPoseCommandWithTargets(UniformPoseCommand):
            """:class:`UniformPoseCommand` that drags a set of rigid scene assets along with the sampled command.

            On every resample (reset or mid-episode), each named asset's root pose in the world
            frame is written to ``command_pose + target_offset_b + R(command_quat) * local_offset``
            expressed in the robot's root frame and then transformed to world. All assets share
            the command's orientation. Useful for compound fixtures (e.g. a socket built from
            several kinematic walls) that must follow a sampled goal pose together.
            """

            def __init__(self, cfg, env):
                super().__init__(cfg, env)
                self._targets = [env.scene[name] for name, _ in cfg.targets]
                # Stack local offsets to shape (n_targets, 3) for batched rotation.
                self._local_offsets = torch.tensor([offset for _, offset in cfg.targets], device=self.device)
                self._target_offset_b = torch.tensor(cfg.target_offset_b, device=self.device).view(1, 3)

            def _resample_command(self, env_ids):
                super()._resample_command(env_ids)
                # Socket center pose in the robot's root frame, then transform to world.
                center_pos_b = self.pose_command_b[env_ids, :3] + self._target_offset_b
                center_quat_b = self.pose_command_b[env_ids, 3:]
                center_pos_w, center_quat_w = combine_frame_transforms(
                    self.robot.data.root_pos_w.torch[env_ids],
                    self.robot.data.root_quat_w.torch[env_ids],
                    center_pos_b,
                    center_quat_b,
                )
                # Write each wall's pose: position = center + R(center_quat) * local_offset;
                # orientation = center orientation.
                for target, local_offset in zip(self._targets, self._local_offsets):
                    offset_w = quat_apply(center_quat_w, local_offset.expand_as(center_pos_w))
                    wall_pos_w = center_pos_w + offset_w
                    target.write_root_pose_to_sim_index(
                        root_pose=torch.cat([wall_pos_w, center_quat_w], dim=-1), env_ids=env_ids
                    )

        _uniform_pose_command_with_targets_cls = _UniformPoseCommandWithTargets
    return _uniform_pose_command_with_targets_cls(cfg, env)


@configclass
class _UniformPoseCommandWithTargetsCfg(UniformPoseCommandCfg):
    """Configuration for the lazily-built multi-target pose command."""

    class_type: Callable = _make_uniform_pose_command_with_targets

    targets: list[tuple[str, tuple[float, float, float]]] = MISSING
    """Per-target ``(scene_asset_name, local_offset_from_socket_center [m])`` pairs."""

    target_offset_b: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Offset added to the command position [m] in the robot root frame to locate the socket center."""


@configclass
class CommandsCfg:
    """Plug goal pose sampled in the robot root frame.

    Ranges are tightened compared to :class:`FrankaCableEnvCfg.CommandsCfg` to keep
    targets inside the half-sphere reachable by the plug while the cable is
    anchored above the tabletop. The four target-hole walls follow each sample,
    rigidly offset from the socket center, which is shifted by
    :attr:`_UniformPoseCommandWithTargetsCfg.target_offset_b` in +x from the
    command pose so the plug can be pushed into it.
    """

    object_pose = _UniformPoseCommandWithTargetsCfg(
        asset_name="robot",
        body_name="panda_hand",
        targets=[(name, offset) for name, offset in _WALL_OFFSETS.items()],
        target_offset_b=(0.1, 0.0, 0.0),
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(_PLUG_INIT_POS[0] - 0.1, _PLUG_INIT_POS[0] + 0.0),
            pos_y=(_PLUG_INIT_POS[1] - 0.05, _PLUG_INIT_POS[1] + 0.05),
            pos_z=(_PLUG_INIT_POS[2] - 0.05, _PLUG_INIT_POS[2] + 0.05),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
        goal_pose_visualizer_cfg=VisualizationMarkersCfg(
            prim_path="/Visuals/Command/goal_pose",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.0,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.2), opacity=0.4),
                ),
            },
        ),
        current_pose_visualizer_cfg=VisualizationMarkersCfg(
            prim_path="/Visuals/Command/body_pose",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=1e-6,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0), opacity=0.0),
                ),
            },
        ),
    )


@configclass
class ActionsCfg:
    """7-dim absolute end-effector pose (xyz + quaternion) via differential IK + 1-dim binary gripper."""

    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.05},
        ),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.05},
        close_command_expr={"panda_finger_.*": 0.005},
    )


@configclass
class ObservationsCfg:
    """Policy observations for the cable plug task."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        plug_position = ObsTerm(
            func=mdp.object_com_in_robot_root_frame,
            params={"asset_cfg": SceneEntityCfg("object")},
        )
        target_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset events for the cable plug task."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.9, 1.1), "velocity_range": (0.0, 0.0)},
    )
    reset_assembly = EventTerm(
        func=mdp.reset_cable_assembly_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.02, 0.02),
                "z": (-0.02, 0.02),
                "yaw": (-math.pi / 18.0, math.pi / 18.0),
            },
            "cable_cfg": SceneEntityCfg("cable"),
            "anchor_cfg": SceneEntityCfg("anchor"),
            "plug_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class RewardsCfg:
    """Reach-and-track reward shaping for the rigid plug."""

    reaching_plug = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.1, "asset_cfg": SceneEntityCfg("object")},
        weight=5.0,
    )
    lifting_plug = RewTerm(
        func=mdp.object_lifted,
        params={"minimal_height": 0.04, "asset_cfg": SceneEntityCfg("object")},
        weight=5.0,
    )
    plug_goal_tracking = RewTerm(
        func=mdp.object_com_goal_distance,
        params={
            "std": 0.3,
            "minimal_height": 0.05,
            "command_name": "object_pose",
            "asset_cfg": SceneEntityCfg("object"),
        },
        weight=16.0,
    )
    plug_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_com_goal_distance,
        params={
            "std": 0.05,
            "minimal_height": 0.05,
            "command_name": "object_pose",
            "asset_cfg": SceneEntityCfg("object"),
        },
        weight=5.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-2)
    gripper_close = RewTerm(
        func=mdp.gripper_close_action,
        params={"action_name": "gripper_action"},
        weight=-1.0,
    )
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-2)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1e-4)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1e-4)


@configclass
class TerminationsCfg:
    """Time-out only; the cable is anchored so the plug cannot escape the workspace."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##


@configclass
class FrankaCablePlugEnvCfg(FrankaSoftEnvCfg):
    """Franka Panda manipulating a cable with a fixed anchor and a rigid plug."""

    scene: _FrankaCablePlugSceneCfg = _FrankaCablePlugSceneCfg(num_envs=128, env_spacing=2.5, replicate_physics=True)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # general settings
        self.decimation = 1
        self.episode_length_s = 6.0

        # simulation settings
        self.sim.dt = 1 / 60.0
        self.sim.render_interval = self.decimation
        self.sim.gravity = (0.0, 0.0, -9.81)

        view = dict(eye=(1.4, 1.0, 0.6), lookat=(0.35, 0.0, 0.1), window_width=1600, window_height=1600)
        self.sim.visualizer_cfgs = [KitVisualizerCfg(**view), NewtonVisualizerCfg(**view)]

        # The proxy-coupled solver from FrankaCableEnvCfg is reused. Both the kinematic anchor
        # and the rigid plug are connected to the cable via VBD attachments, so the solver
        # needs to know about them on the VBD side.
        self.sim.physics = CoupledNewtonCfg(
            scene_cfg=self.scene,
            solver_cfg=CoupledAdmmSolverCfg(
                src_solver_cfg=MJWarpSolverCfg(
                    cone="elliptic",
                    ls_parallel=True,
                    ls_iterations=20,
                    integrator="implicitfast",
                ),
                dst_solver_cfg=VBDSolverCfg(iterations=20, rigid_avbd_beta=1e3, rigid_contact_k_start=1e3),
                src_bodies=[SceneEntityCfg("robot")],
                dst_bodies=[
                    SceneEntityCfg("object"),
                    SceneEntityCfg("anchor"),
                    SceneEntityCfg("target_hole_top"),
                    SceneEntityCfg("target_hole_bottom"),
                    SceneEntityCfg("target_hole_left"),
                    SceneEntityCfg("target_hole_right"),
                    SceneEntityCfg("cable"),
                ],
                iterations=5,
                rho=30.0,
                gamma=0.1,
                baumgarte=0.005,
                contact_distance=0.003,
            ),
            model_cfg=NewtonModelCfg(
                shape_material_ke=1e5,
                shape_material_kd=1e-2,
                shape_material_mu=1.0,
            ),
            num_substeps=10,
        )
