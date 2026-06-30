# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Panda grasping a Newton cable plug and inserting it into a socket.

The cable is welded to a kinematic anchor above the tabletop at one end and to a
rigid plug at the other. The RL task brings the plug to a sampled target pose.
"""

from __future__ import annotations

import math
from dataclasses import MISSING

import torch
from isaaclab_newton.physics import MJWarpSolverCfg
from isaaclab_newton.sim.spawners.materials import NewtonCableMaterialCfg
from isaaclab_visualizers.kit.kit_visualizer_cfg import KitVisualizerCfg
from isaaclab_visualizers.newton.newton_visualizer_cfg import NewtonVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply, quat_from_angle_axis, subtract_frame_transforms

from isaaclab_contrib.cable.cable_object_cfg import CableAttachmentCfg, CableObjectCfg
from isaaclab_contrib.coupling import CoupledProxySolverCfg
from isaaclab_contrib.deformable.newton_manager_cfg import (
    CoupledNewtonCfg,
    NewtonModelCfg,
    VBDSolverCfg,
)

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from . import mdp
from .franka_soft_env_cfg import FrankaSoftEnvCfg, _FrankaSoftSceneCfg
from .mdp.rewards import _SOCKET_CFGS, _socket_frame_w

##
# Cable / attachment geometry constants
##

# Cable: 28 segments x 0.02 m, 0.01 m wide (matches the parent cable env).
_NUM_POINTS = 28
_SEGMENT_LENGTH = 0.02
_CABLE_WIDTH = 0.01

# Newton rigid bodies the cable resolves to -- fewer than the _NUM_POINTS=28 control points, per the
# rod builder's internal mapping. The no-cable variant emits a zero cable_poses slot of this size so
# its observation space matches the cable env. test_cable_plug_nocable_parity pins it to the live cable.
_CABLE_NUM_BODIES = 26

# Per-episode reset transform applied to the cable assembly (cable + anchor + plug).
_RESET_POSE_RANGE = {
    # "x": (-0.05, 0.05),
    # "y": (-0.3, 0.3),
    # "z": (-0.05, 0.05),
    # "yaw": (-math.pi / 3.0, math.pi / 3.0),
}

# Franka shoulder (J1/J2 axis intersection) in the robot root frame [m]; the reachable workspace is
# ~a spherical shell about it, so reset sampling uses it as the sphere origin.
_SHOULDER_OFFSET = (0.0, 0.0, 0.333)

# Reachable-workspace bounds (shoulder-centered shell): r [m], polar theta [rad], azimuth phi [rad].
# All reset sampling is clipped to these so the plug and goal stay reachable.
_FRANKA_WORKSPACE = {
    "r": (0.15, 0.75),
    "theta": (math.pi / 12.0, math.pi / 2.0 + math.pi / 3.0),
    "phi": (-math.pi / 4.0, math.pi / 4.0),
    "shoulder_offset": _SHOULDER_OFFSET,
}


def _clip_to_workspace(pose_range: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
    """Clip a spherical pose range's ``r``/``theta``/``phi`` to :data:`_FRANKA_WORKSPACE`."""
    clipped = dict(pose_range)
    for key in ("r", "theta", "phi"):
        if key in clipped:
            lo, hi = clipped[key]
            w_lo, w_hi = _FRANKA_WORKSPACE[key]
            clipped[key] = (max(lo, w_lo), min(hi, w_hi))
    return clipped


# No-cable plug reset: the plug spawns at the default grasp point, jittered in shoulder-centered
# spherical coords (r [m], polar theta [rad], azimuth phi [rad]) so the arm only closes to grab. The
# default is the FK grasp point (panda_hand + ee offset) at the Franka default config.
_DEFAULT_FRANKA_POSE = (0.46630, 1.45799, 0.0)
# Default plug orientation (euler xyz [rad]) at the default config; the reset starts at this exact
# pose (degenerate ranges) and the curriculum widens roll/pitch/yaw from here.
_DEFAULT_PLUG_RPY = (0.04440, -0.77480, math.pi)
_PLUG_GRASP_RANGE = _clip_to_workspace(
    {
        "r": (_DEFAULT_FRANKA_POSE[0], _DEFAULT_FRANKA_POSE[0]),
        "theta": (_DEFAULT_FRANKA_POSE[1], _DEFAULT_FRANKA_POSE[1]),
        "phi": (_DEFAULT_FRANKA_POSE[2], _DEFAULT_FRANKA_POSE[2]),
        "roll": (_DEFAULT_PLUG_RPY[0], _DEFAULT_PLUG_RPY[0]),
        "pitch": (_DEFAULT_PLUG_RPY[1], _DEFAULT_PLUG_RPY[1]),
        "yaw": (_DEFAULT_PLUG_RPY[2], _DEFAULT_PLUG_RPY[2]),
    }
)

# Goal pose reset: position sampled in the same shoulder-centered spherical shell as the plug (keeps
# the goal reachable), with the socket's insertion orientation kept wide.
# Pitch is -90 deg from the default plug orientation so the socket faces +x.
_GOAL_SPHERICAL_RANGE = _clip_to_workspace(
    {
        "r": _PLUG_GRASP_RANGE["r"],
        "theta": _PLUG_GRASP_RANGE["theta"],
        "phi": _PLUG_GRASP_RANGE["phi"],
        "pitch": (_DEFAULT_PLUG_RPY[1] - math.pi / 2, _DEFAULT_PLUG_RPY[1] - math.pi / 2),
        "yaw": (_DEFAULT_PLUG_RPY[2], _DEFAULT_PLUG_RPY[2]),
    }
)

# Curriculum: linearly widen the plug/goal reset ranges from their tight initial values (above) to
# these wider, still workspace-clipped, final bounds over the first _CURRICULUM_NUM_STEPS env steps.
# The plug widens position (r/theta/phi) and orientation (roll/pitch/yaw about the default tilt); the
# goal widens position only (its pitch/yaw stay at initial).
# Scale to the training budget: common_step_counter advances num_steps_per_env per iteration, so a
# full run (24 * 50000 iters) reaches ~1.2e6; saturate partway so training continues at full difficulty.
_CURRICULUM_NUM_STEPS = 2e4
_PLUG_GRASP_RANGE_FINAL = _clip_to_workspace(
    {
        "r": (0.15, 0.75),
        "theta": (math.pi / 4.0, math.pi / 2.0 + math.pi / 6.0),
        "phi": (-math.pi / 4.0, math.pi / 4.0),
        "roll": (_DEFAULT_PLUG_RPY[0] - math.pi / 4, _DEFAULT_PLUG_RPY[0] + math.pi / 4),
        "pitch": (_DEFAULT_PLUG_RPY[1] - math.pi / 4, _DEFAULT_PLUG_RPY[1] + math.pi / 4),
        "yaw": (_DEFAULT_PLUG_RPY[2] - math.pi / 4, _DEFAULT_PLUG_RPY[2] + math.pi / 4),
    }
)
_GOAL_SPHERICAL_RANGE_FINAL = _clip_to_workspace(
    {
        "r": (0.15, 0.75),
        "theta": (math.pi / 4.0, math.pi / 2.0 + math.pi / 6.0),
        "phi": (-math.pi / 4.0, math.pi / 4.0),
        "pitch": (_DEFAULT_PLUG_RPY[1] - 1.5 * math.pi, _DEFAULT_PLUG_RPY[1] - math.pi / 4),
        "yaw": (_DEFAULT_PLUG_RPY[2] - math.pi / 3, _DEFAULT_PLUG_RPY[2] + math.pi / 3),
    }
)

# Episode length [s] ramped by the curriculum: short episodes early (fast grasp feedback), long ones
# once the task widens.
_EPISODE_LENGTH_INITIAL = 0.5
_EPISODE_LENGTH_FINAL = 4.0

# Kinematic anchor, above the tabletop in front of the robot [m].
_ANCHOR_POS = (0.15, 0.0, 0.2)

# Light plug so the grasp holds and cable tension stays low [m, m, kg].
_PLUG_RADIUS = 0.01
_PLUG_HEIGHT = 0.04
_PLUG_MASS = 0.05

# Taut plug reach from the anchor [m]; the plug spawns here and hangs taut under gravity.
_CABLE_REACH = (_NUM_POINTS - 2) * _SEGMENT_LENGTH
_PLUG_INIT_POS = (_ANCHOR_POS[0] + _CABLE_REACH, _ANCHOR_POS[1], _ANCHOR_POS[2])

# Socket dimensions [m]. Its pose is sampled per episode in a reset event (see EventCfg).
_TARGET_HOLE_INNER = 0.03  # clear opening, > plug diameter
_TARGET_HOLE_WALL_THICKNESS = 0.003
_TARGET_HOLE_DEPTH = _PLUG_HEIGHT

# Socket center relative to the sampled goal, in the goal's local frame [m]: the goal (the staging
# point the plug tracks) sits one offset in front of the socket opening (which faces -x).
_SOCKET_OFFSET_B = (0.1, 0.0, 0.0)
_STAGING_OFFSET = tuple(-v for v in _SOCKET_OFFSET_B)  # goal relative to socket center, in the bore frame


##
# Socket: four kinematic cuboid walls forming a square opening that faces -x.
##

_WALL_SPAN = _TARGET_HOLE_INNER + 2.0 * _TARGET_HOLE_WALL_THICKNESS
_WALL_OFFSET = (_TARGET_HOLE_INNER + _TARGET_HOLE_WALL_THICKNESS) / 2.0

# Per-wall size [m] (x, y, z): top/bottom span y, left/right span z.
_WALL_SIZE_TB = (_TARGET_HOLE_DEPTH, _WALL_SPAN, _TARGET_HOLE_WALL_THICKNESS)
_WALL_SIZE_LR = (_TARGET_HOLE_DEPTH, _TARGET_HOLE_WALL_THICKNESS, _TARGET_HOLE_INNER)

# Per-wall offset from the socket center [m].
_WALL_OFFSETS: dict[str, tuple[float, float, float]] = {
    "target_hole_top": (0.0, 0.0, _WALL_OFFSET),
    "target_hole_bottom": (0.0, 0.0, -_WALL_OFFSET),
    "target_hole_left": (0.0, -_WALL_OFFSET, 0.0),
    "target_hole_right": (0.0, _WALL_OFFSET, 0.0),
}


def _wall_cfg(size: tuple[float, float, float], offset: tuple[float, float, float], prim_name: str) -> RigidObjectCfg:
    """Kinematic cuboid wall at ``offset`` from the socket center (a reset event repositions it).

    The reset writes the wall's VBD ``body_q`` directly (see :func:`reset_socket_pose_uniform`); a
    kinematic body is never integrated, so that pose persists and the wall stays immovable under the
    plug during insertion.
    """
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/{prim_name}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=offset),
    )


##
# Scene
##


@configclass
class _FrankaCablePlugSceneCfg(_FrankaSoftSceneCfg):
    """Franka, anchor, plug, socket walls, and the cable joining anchor to plug."""

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

    target_hole_top: RigidObjectCfg = _wall_cfg(_WALL_SIZE_TB, _WALL_OFFSETS["target_hole_top"], "TargetHoleTop")
    target_hole_bottom: RigidObjectCfg = _wall_cfg(
        _WALL_SIZE_TB, _WALL_OFFSETS["target_hole_bottom"], "TargetHoleBottom"
    )
    target_hole_left: RigidObjectCfg = _wall_cfg(_WALL_SIZE_LR, _WALL_OFFSETS["target_hole_left"], "TargetHoleLeft")
    target_hole_right: RigidObjectCfg = _wall_cfg(_WALL_SIZE_LR, _WALL_OFFSETS["target_hole_right"], "TargetHoleRight")

    table_collider: RigidObjectCfg = RigidObjectCfg(
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

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Plug",
        spawn=sim_utils.CylinderCfg(
            radius=_PLUG_RADIUS,
            height=_PLUG_HEIGHT,
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
                density=10.0,
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
        # World gravity with full gravity compensation, so the low-PD IK does not fight sag.
        self.robot.spawn.rigid_props = sim_utils.MujocoRigidBodyPropertiesCfg(gravcomp=1.0)

        # Stiffer gripper.
        self.robot.actuators["panda_hand"].effort_limit_sim = 1500.0
        self.robot.actuators["panda_hand"].stiffness = 1500.0
        self.robot.actuators["panda_hand"].damping = 150.0

        for actuator_name in ("panda_shoulder", "panda_forearm"):
            self.robot.actuators[actuator_name].damping = 16.0


##
# MDP overrides
##


class _SocketPoseCommand(CommandTerm):
    """Goal pose (robot root frame) read back from the socket walls placed by the reset event.

    The socket is sampled and built once per episode in
    :func:`~isaaclab_tasks.core.lift.config.franka_soft.mdp.reset_socket_pose_uniform`,
    so this command samples nothing. Each step it reads the four walls and exposes the staging point
    (one offset in front of the opening, which faces -x) as the goal the plug tracks, letting the
    stock command-based observation and reward terms apply unchanged. Subclasses
    :class:`~isaaclab.managers.CommandTerm` directly to avoid importing :class:`UniformPoseCommand`,
    whose :class:`~isaaclab.assets.Articulation` import corrupts the Kit loader before app start.
    """

    cfg: _SocketPoseCommandCfg

    def __init__(self, cfg: _SocketPoseCommandCfg, env) -> None:
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.object = env.scene[cfg.object_name]
        self._staging_offset = torch.tensor(cfg.staging_offset, device=self.device)
        # commands: (x, y, z, qx, qy, qz, qw) in the robot root frame; mirror in world for markers.
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.pose_command_b

    def _refresh(self) -> None:
        socket_pos_w, socket_quat_w = _socket_frame_w(self._env, _SOCKET_CFGS)
        self.pose_command_w[:, :3] = socket_pos_w + quat_apply(
            socket_quat_w, self._staging_offset.expand_as(socket_pos_w)
        )
        self.pose_command_w[:, 3:] = socket_quat_w
        self.pose_command_b[:, :3], self.pose_command_b[:, 3:] = subtract_frame_transforms(
            self.robot.data.root_pos_w.torch,
            self.robot.data.root_quat_w.torch,
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
        )

    def _update_metrics(self) -> None:
        self.metrics["position_error"] = torch.linalg.norm(
            self.pose_command_w[:, :3] - self.object.data.root_pos_w.torch, dim=-1
        )

    def _resample_command(self, env_ids) -> None:
        self._refresh()  # socket is fixed per episode; the reset event already placed the walls

    def _update_command(self) -> None:
        self._refresh()

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        from isaaclab.markers import VisualizationMarkers  # lazy: avoid pulling Kit at module load

        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        elif hasattr(self, "goal_pose_visualizer"):
            self.goal_pose_visualizer.set_visibility(False)
            self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        if not self.robot.is_initialized:
            return
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
        self.current_pose_visualizer.visualize(self.object.data.root_pos_w.torch, self.object.data.root_quat_w.torch)


@configclass
class _SocketPoseCommandCfg(CommandTermCfg):
    """Configuration for the socket-derived goal pose command."""

    class_type: type = _SocketPoseCommand
    asset_name: str = MISSING
    """Robot entity providing the root frame."""
    object_name: str = MISSING
    """Plug entity, drawn by the current-pose marker and used for the tracking metric."""
    staging_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Goal offset from the socket center [m], in the bore frame."""
    goal_pose_visualizer_cfg: VisualizationMarkersCfg = MISSING
    current_pose_visualizer_cfg: VisualizationMarkersCfg = MISSING


@configclass
class CommandsCfg:
    """Plug goal pose (robot root frame), read from the socket walls placed by the reset event."""

    object_pose = _SocketPoseCommandCfg(
        asset_name="robot",
        object_name="object",
        staging_offset=_STAGING_OFFSET,
        resampling_time_range=(1.0e9, 1.0e9),  # fixed per episode; the reset event sets the socket
        debug_vis=True,
        goal_pose_visualizer_cfg=FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose").replace(
            markers={"frame": FRAME_MARKER_CFG.markers["frame"].replace(scale=(0.05, 0.05, 0.05))}
        ),
        current_pose_visualizer_cfg=VisualizationMarkersCfg(
            prim_path="/Visuals/Command/body_pose",
            markers={"frame": FRAME_MARKER_CFG.markers["frame"].replace(scale=(0.001, 0.001, 0.001))},
        ),
    )


@configclass
class ActionsCfg:
    """7-dim arm joint position + 1-dim binary gripper command."""

    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.007},
    )


@configclass
class ObservationsCfg:
    """Policy observations for the cable plug task."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        plug_pose = ObsTerm(
            func=mdp.body_poses_in_robot_root_frame,
            params={"asset_cfg": SceneEntityCfg("object")},
        )
        cable_poses = ObsTerm(
            func=mdp.body_poses_in_robot_root_frame,
            params={"asset_cfg": SceneEntityCfg("cable")},
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
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )
    reset_assembly = EventTerm(
        func=mdp.reset_cable_assembly_uniform,
        mode="reset",
        params={
            "pose_range": _RESET_POSE_RANGE,
            "cable_cfg": SceneEntityCfg("cable"),
            "anchor_cfg": SceneEntityCfg("anchor"),
            "plug_cfg": SceneEntityCfg("object"),
        },
    )
    # Plug-only reset, populated by the no-cable variant (mutually exclusive with reset_assembly).
    reset_plug: EventTerm | None = None
    # Sample the goal pose (robot frame) and place the kinematic socket walls in front of it.
    reset_socket = EventTerm(
        func=mdp.reset_socket_pose_uniform,
        mode="reset",
        params={
            "pose_range": _GOAL_SPHERICAL_RANGE,
            "socket_offset_b": _SOCKET_OFFSET_B,
            "wall_offsets": _WALL_OFFSETS,
            "shoulder_offset": _SHOULDER_OFFSET,
        },
    )
    # Clear the proxy teleport velocity from the arm-joint reset above; must run after it.
    reset_proxy_velocity = EventTerm(func=mdp.reset_proxy_body_prev, mode="reset")


@configclass
class RewardsCfg:
    """Reach-and-track reward shaping for the rigid plug."""

    reaching_plug = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.1, "asset_cfg": SceneEntityCfg("object")},
        weight=1.0,
    )
    # reach_threshold gates the grasp on the plug center sitting at the grasp point (ee_frame TCP): a held
    # plug measures ~0 m from it, while pressing the socket walls (whose net force the finger sensor cannot
    # tell from the plug's) leaves the plug outside this radius, so 0.02 m credits grasping the plug only.
    grasping_plug = RewTerm(
        func=mdp.object_grasped,
        params={"force_threshold": 0.1, "reach_threshold": 0.02, "asset_cfg": SceneEntityCfg("object")},
        weight=10.0,
    )
    plug_goal_tracking = RewTerm(
        func=mdp.object_grasped_goal_distance,
        params={
            "std": 0.25,
            "minimal_height": 0.0,
            "command_name": "object_pose",
            "force_threshold": 0.1,
            "reach_threshold": 0.02,
            "asset_cfg": SceneEntityCfg("object"),
        },
        weight=5.0,
    )
    # Dense peg-in-hole shaping (grasp-gated): centers, aligns, then seats the plug in the bore. Drives the
    # final ~10 cm from the staging goal into the socket, which goal-tracking alone does not (its tanh kernel
    # saturates at the staging point). Weight 3 -> max ~9, kept below grasping (10) so the grasp holds first.
    # plug_socket_insertion = RewTerm(
    #     func=mdp.plug_socket_insertion,
    #     params={
    #         "std": 0.02,
    #         "depth": _TARGET_HOLE_DEPTH,
    #         "radius": _TARGET_HOLE_INNER / 2.0,
    #         "min_axis_cos": 0.9,
    #         "asset_cfg": SceneEntityCfg("object"),
    #     },
    #     weight=3.0,
    # )
    # Sparse success bonus when the plug is grasped and its center is seated inside the socket bore. Grasp-
    # gated so it cannot be earned by dropping a free plug in. Weight 50 -- a crisp success signal, not the
    # old 500 (which would spike the value targets and destabilize PPO).
    # plug_inserted = RewTerm(
    #     func=mdp.plug_inserted,
    #     params={
    #         "depth_tol": _TARGET_HOLE_DEPTH / 2.0,
    #         "radius": _TARGET_HOLE_INNER / 2.0,
    #         "force_threshold": 0.1,
    #         "reach_threshold": 0.02,
    #     },
    #     weight=50.0,
    # )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-2)
    # gripper_close = RewTerm(
    #     func=mdp.gripper_close_amount,
    #     params={"action_name": "gripper_action"},
    #     weight=-1e-2,
    # )
    # joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4)
    # joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1e-6)
    # joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1e-6)


@configclass
class TerminationsCfg:
    """Time-out plus a velocity-divergence guard that resets blown-up envs."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Plug dropped below the tabletop.
    plug_below_table = DoneTerm(
        func=mdp.object_com_below_minimum,
        params={"minimum_height": 0.0, "asset_cfg": SceneEntityCfg("object")},
    )

    # Plug left the table footprint.
    plug_outside_table = DoneTerm(
        func=mdp.object_outside_table_bounds,
        params={
            "x_bounds": (0.0, 1.0),
            "y_bounds": (-0.5, 0.5),
            "z_bounds": (0.0, 1.0),
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    # Thresholds well above the worst seen under extreme random actions (~30 rad/s, ~5 m/s).
    velocity_divergence = DoneTerm(
        func=mdp.assembly_velocity_out_of_bounds,
        params={"max_joint_vel": 50.0, "max_body_vel": 20.0},
    )


@configclass
class CurriculumCfg:
    """Widen the goal (and, without the cable, plug) reset ranges over training steps."""

    reset_goal_range = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "events.reset_socket.params.pose_range",
            "modify_fn": mdp.step_widen_pose_range,
            "modify_params": {
                "initial_range": _GOAL_SPHERICAL_RANGE,
                "final_range": _GOAL_SPHERICAL_RANGE_FINAL,
                "num_steps": _CURRICULUM_NUM_STEPS,
            },
        },
    )
    # Plug-only widening, populated by the no-cable variant (reset_plug exists only there).
    reset_plug_range: CurrTerm | None = None

    # Ramp the episode length from short to long over training.
    episode_length = CurrTerm(
        func=mdp.modify_env_param,
        params={
            "address": "cfg.episode_length_s",
            "modify_fn": mdp.step_interpolate_value,
            "modify_params": {
                "initial_value": _EPISODE_LENGTH_INITIAL,
                "final_value": _EPISODE_LENGTH_FINAL,
                "num_steps": _CURRICULUM_NUM_STEPS,
            },
        },
    )

    # Report the widening progress (0->1) as Curriculum/progress in the training output.
    progress = CurrTerm(
        func=mdp.curriculum_progress,
        params={"num_steps": _CURRICULUM_NUM_STEPS},
    )


##
# Environment configuration
##


@configclass
class FrankaCablePlugEnvCfg(FrankaSoftEnvCfg):
    """Franka Panda manipulating a cable with a fixed anchor and a rigid plug."""

    with_cable: bool = True
    """If ``False``, drop the cable and anchor and manipulate the free rigid plug alone. The
    observation and action spaces are unchanged (the absent cable poses become zeros), so a policy
    trained without the cable can be deployed directly on the cable task."""

    scene: _FrankaCablePlugSceneCfg = _FrankaCablePlugSceneCfg(num_envs=1024, env_spacing=2.5, replicate_physics=True)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def _disable_cable(self) -> None:
        """Remove the cable and anchor; manipulate the free plug, keeping the observation space."""
        self.scene.cable = None
        self.scene.anchor = None
        # Spawn the plug at the gripper, so a zero-offset reset only needs the fingers to close. The
        # arm must therefore reset to its default config (a zero default-offset action holds it
        # there); otherwise it would drift off the spawned plug before the fingers close.
        self.events.reset_assembly = None
        self.events.reset_plug = EventTerm(
            func=mdp.reset_plug_uniform,
            mode="reset",
            params={
                "pose_range": _PLUG_GRASP_RANGE,
                "plug_cfg": SceneEntityCfg("object"),
                "shoulder_offset": _SHOULDER_OFFSET,
            },
        )
        # Widen the plug grasp range over training (the goal range is widened in the base cfg).
        self.curriculum.reset_plug_range = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "events.reset_plug.params.pose_range",
                "modify_fn": mdp.step_widen_pose_range,
                "modify_params": {
                    "initial_range": _PLUG_GRASP_RANGE,
                    "final_range": _PLUG_GRASP_RANGE_FINAL,
                    "num_steps": _CURRICULUM_NUM_STEPS,
                },
            },
        )
        # Keep the cable_poses slot (zeros) so the observation space matches the cable env.
        self.observations.policy.cable_poses = ObsTerm(
            func=mdp.zero_body_poses, params={"num_bodies": _CABLE_NUM_BODIES}
        )
        # The divergence guard no longer has a cable to bound; check the plug alone.
        self.terminations.velocity_divergence.params["asset_cfgs"] = (SceneEntityCfg("object"),)

    def __post_init__(self) -> None:
        super().__post_init__()

        # general settings
        self.decimation = 1
        # Curriculum ramps this up to _EPISODE_LENGTH_FINAL; start short for fast early feedback.
        self.episode_length_s = _EPISODE_LENGTH_INITIAL

        # simulation settings
        self.sim.dt = 1 / 60.0
        self.sim.render_interval = self.decimation
        self.sim.gravity = (0.0, 0.0, 0.0)

        view = dict(eye=(1.4, 1.0, 0.6), lookat=(0.35, 0.0, 0.1), window_width=1600, window_height=1600)
        self.sim.visualizer_cfgs = [KitVisualizerCfg(**view), NewtonVisualizerCfg(**view)]

        # Reconfigure scene/events/observations before building the solver so dst_bodies is correct.
        if not self.with_cable:
            self._disable_cable()

        # VBD-side bodies: plug + walls always; anchor + cable only when the cable is present.
        dst_bodies = [
            SceneEntityCfg("object"),
            SceneEntityCfg("target_hole_top"),
            SceneEntityCfg("target_hole_bottom"),
            SceneEntityCfg("target_hole_left"),
            SceneEntityCfg("target_hole_right"),
        ]
        if self.with_cable:
            dst_bodies += [SceneEntityCfg("anchor"), SceneEntityCfg("cable")]

        # Proxy-coupled solver (mirrors FrankaCableEnvCfg): arm in MJWarp, plug/walls (and the
        # cable assembly when present) in VBD, gripper fingers as VBD proxies so VBD's
        # contact/friction grasps the plug.
        self.sim.physics = CoupledNewtonCfg(
            scene_cfg=self.scene,
            solver_cfg=CoupledProxySolverCfg(
                src_solver_cfg=MJWarpSolverCfg(
                    cone="elliptic",
                    ls_parallel=True,
                    ls_iterations=20,
                    integrator="implicitfast",
                ),
                dst_solver_cfg=VBDSolverCfg(iterations=20, rigid_avbd_beta=1e3, rigid_contact_k_start=1e3),
                src_bodies=[SceneEntityCfg("robot"), SceneEntityCfg("table_collider")],
                dst_bodies=dst_bodies,
                proxy_bodies=[
                    SceneEntityCfg("robot", body_names=["panda_hand", "panda_(left|right)finger"]),
                ],
                # More relaxation passes tighten the proxy grip on the plug.
                proxy_iterations=4,
            ),
            model_cfg=NewtonModelCfg(
                shape_material_ke=1e5,
                shape_material_kd=1e-2,
                shape_material_mu=10.0,
            ),
            num_substeps=8,
        )


@configclass
class FrankaCablePlugNoCableEnvCfg(FrankaCablePlugEnvCfg):
    """Cable-plug task without the cable: only the free rigid plug and socket are simulated.

    Keeps the observation and action spaces of :class:`FrankaCablePlugEnvCfg`, so a policy trained
    here can be deployed directly on the cable env.
    """

    with_cable: bool = False


def _pin_full_difficulty(cfg: FrankaCablePlugEnvCfg) -> None:
    """Pin the reset ranges to the curriculum's final (full-difficulty) bounds and disable it.

    For eval the difficulty should not ramp from scratch with ``common_step_counter``, so the goal
    (and, without the cable, plug) reset ranges and the episode length are set to what the curriculum
    reaches at the end of training, and the now-redundant curriculum terms are removed.
    """
    cfg.events.reset_socket.params["pose_range"] = {**_GOAL_SPHERICAL_RANGE, **_GOAL_SPHERICAL_RANGE_FINAL}
    if cfg.events.reset_plug is not None:
        cfg.events.reset_plug.params["pose_range"] = {**_PLUG_GRASP_RANGE, **_PLUG_GRASP_RANGE_FINAL}
    cfg.episode_length_s = _EPISODE_LENGTH_FINAL
    cfg.curriculum.reset_goal_range = None
    cfg.curriculum.reset_plug_range = None
    cfg.curriculum.episode_length = None
    cfg.curriculum.progress = None


@configclass
class FrankaCablePlugEnvCfg_PLAY(FrankaCablePlugEnvCfg):
    """Eval cfg for the cable variant: reset ranges pinned to full difficulty, curriculum disabled."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        _pin_full_difficulty(self)


@configclass
class FrankaCablePlugNoCableEnvCfg_PLAY(FrankaCablePlugNoCableEnvCfg):
    """Eval cfg for the no-cable variant: reset ranges pinned to full difficulty, curriculum disabled."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        _pin_full_difficulty(self)
