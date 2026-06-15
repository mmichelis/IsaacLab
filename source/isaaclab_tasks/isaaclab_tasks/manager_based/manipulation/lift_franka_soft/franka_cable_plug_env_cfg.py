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
from collections.abc import Callable
from dataclasses import MISSING

import torch
from isaaclab_newton.physics import MJWarpSolverCfg
from isaaclab_newton.sim.spawners.materials import NewtonCableMaterialCfg
from isaaclab_visualizers.kit.kit_visualizer_cfg import KitVisualizerCfg
from isaaclab_visualizers.newton.newton_visualizer_cfg import NewtonVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_from_angle_axis

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

##
# Cable / attachment geometry constants
##

# Cable: 28 segments x 0.02 m, 0.01 m wide (matches the parent cable env).
_NUM_POINTS = 28
_SEGMENT_LENGTH = 0.02
_CABLE_WIDTH = 0.01

# Kinematic anchor, above the tabletop in front of the robot [m].
_ANCHOR_POS = (0.15, 0.0, 0.2)

# Light plug so the grasp holds and cable tension stays low [m, m, kg].
_PLUG_RADIUS = 0.01
_PLUG_HEIGHT = 0.04
_PLUG_MASS = 0.05

# Taut plug reach from the anchor [m]; the plug spawns here and hangs taut under gravity.
_CABLE_REACH = (_NUM_POINTS - 2) * _SEGMENT_LENGTH
_PLUG_INIT_POS = (_ANCHOR_POS[0] + _CABLE_REACH, _ANCHOR_POS[1], _ANCHOR_POS[2])

# Goal well inside the cable reach, so insertion slackens (not stretches) the cable [m].
_GOAL_POS = (_ANCHOR_POS[0] + 0.5 * _CABLE_REACH, _ANCHOR_POS[1], _ANCHOR_POS[2] - 0.05)

# Socket dimensions [m]. Its pose follows the goal at runtime (offset +x; see CommandsCfg).
_TARGET_HOLE_INNER = 0.03  # clear opening, > plug diameter
_TARGET_HOLE_WALL_THICKNESS = 0.003
_TARGET_HOLE_DEPTH = _PLUG_HEIGHT


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
    """Kinematic cuboid wall at ``offset`` from the socket center (the command repositions it)."""
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


##
# MDP overrides
##


# Built lazily on first use: importing UniformPoseCommand at module load pulls in
# Articulation, whose Kit C-extensions corrupt the plugin loader before the app starts.
_uniform_pose_command_with_targets_cls: type | None = None


def _make_uniform_pose_command_with_targets(cfg, env):
    """Build (once) and instantiate the multi-target pose command class."""
    global _uniform_pose_command_with_targets_cls
    if _uniform_pose_command_with_targets_cls is None:
        from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand
        from isaaclab.utils.math import combine_frame_transforms, quat_apply

        class _UniformPoseCommandWithTargets(UniformPoseCommand):
            """:class:`UniformPoseCommand` that moves rigid assets with the sampled command.

            On each resample, every target is placed at ``command +
            R(command_quat) * (target_offset_b + local_offset)`` and shares the command
            orientation, so the socket walls follow the goal pose together. The socket offset is
            applied in the goal's local frame, so it points along the sampled goal's x axis.
            """

            def __init__(self, cfg, env):
                super().__init__(cfg, env)
                self._targets = [env.scene[name] for name, _ in cfg.targets]
                self._local_offsets = torch.tensor([offset for _, offset in cfg.targets], device=self.device)
                self._target_offset_b = torch.tensor(cfg.target_offset_b, device=self.device).view(1, 3)

            def _resample_command(self, env_ids):
                super()._resample_command(env_ids)
                # Socket center in robot frame (offset applied in the goal's local frame), then to world.
                center_quat_b = self.pose_command_b[env_ids, 3:]
                center_pos_b = self.pose_command_b[env_ids, :3] + quat_apply(
                    center_quat_b, self._target_offset_b.expand(len(env_ids), 3)
                )
                center_pos_w, center_quat_w = combine_frame_transforms(
                    self.robot.data.root_pos_w.torch[env_ids],
                    self.robot.data.root_quat_w.torch[env_ids],
                    center_pos_b,
                    center_quat_b,
                )
                # Place each wall at center + R(center_quat) * local_offset.
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
    """Offset to the socket center [m], applied in the sampled goal's local frame."""


@configclass
class CommandsCfg:
    """Plug goal pose (robot root frame); the socket walls follow it, offset +x."""

    object_pose = _UniformPoseCommandWithTargetsCfg(
        asset_name="robot",
        body_name="panda_hand",
        targets=[(name, offset) for name, offset in _WALL_OFFSETS.items()],
        target_offset_b=(0.1, 0.0, 0.0),
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(_GOAL_POS[0] - 0.03, _GOAL_POS[0] + 0.07),
            pos_y=(_GOAL_POS[1] - 0.1, _GOAL_POS[1] + 0.1),
            pos_z=(_GOAL_POS[2] + 0.0, _GOAL_POS[2] + 0.15),
            roll=(0.0, 0.0),
            pitch=(-math.pi / 9, math.pi / 9),
            yaw=(-math.pi / 4, math.pi / 4),
        ),
        goal_pose_visualizer_cfg=FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose").replace(
            markers={"frame": FRAME_MARKER_CFG.markers["frame"].replace(scale=(0.05, 0.05, 0.05))}
        ),
        # Shrink the current-pose (EE) frame to ~invisible so only the goal frame is shown.
        current_pose_visualizer_cfg=FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/body_pose").replace(
            markers={"frame": FRAME_MARKER_CFG.markers["frame"].replace(scale=(0.001, 0.001, 0.001))}
        ),
    )


@configclass
class ActionsCfg:
    """7-dim arm joint position + 2-dim continuous gripper joint position."""

    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
    )
    gripper_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        scale=0.04,
        use_default_offset=True,
        clip={"panda_finger_.*": (0.007, 0.04)},
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
        params={"position_range": (0.9, 1.1), "velocity_range": (0.0, 0.0)},
    )
    reset_assembly = EventTerm(
        func=mdp.reset_cable_assembly_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.3, 0.3),
                "z": (-0.05, 0.05),
                "yaw": (-math.pi / 3.0, math.pi / 3.0),
            },
            "cable_cfg": SceneEntityCfg("cable"),
            "anchor_cfg": SceneEntityCfg("anchor"),
            "plug_cfg": SceneEntityCfg("object"),
        },
    )
    # Clear the proxy teleport velocity from the arm-joint reset above; must run after it.
    reset_proxy_velocity = EventTerm(func=mdp.reset_proxy_body_prev, mode="reset")


@configclass
class RewardsCfg:
    """Reach-grasp-and-track reward shaping for the rigid plug."""

    reaching_plug = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.1, "asset_cfg": SceneEntityCfg("object")},
        weight=8.0,
    )
    # Reward closing the gripper only when the EE is near the plug (bootstraps grasping).
    grasp_plug = RewTerm(
        func=mdp.grasp_plug,
        params={"std": 0.05, "action_name": "gripper_action", "asset_cfg": SceneEntityCfg("object")},
        weight=4.0,
    )
    # Track the goal only while the plug is held near the EE (else the reward is farmable).
    # Coarse kernel: far-field gradient that pulls the held plug toward the goal.
    plug_goal_tracking = RewTerm(
        func=mdp.object_com_goal_distance,
        params={
            "std": 0.3,
            "minimal_height": 0.05,
            "command_name": "object_pose",
            "asset_cfg": SceneEntityCfg("object"),
            "held_distance": 0.08,
        },
        weight=12.0,
    )
    # Fine kernel: sharp near-goal peak that rewards closing the last few cm.
    plug_goal_tracking_fine = RewTerm(
        func=mdp.object_com_goal_distance,
        params={
            "std": 0.08,
            "minimal_height": 0.05,
            "command_name": "object_pose",
            "asset_cfg": SceneEntityCfg("object"),
            "held_distance": 0.08,
        },
        weight=18.0,
    )
    # Align the plug orientation with the goal orientation (held gate).
    plug_goal_orientation = RewTerm(
        func=mdp.object_goal_orientation,
        params={
            "command_name": "object_pose",
            "asset_cfg": SceneEntityCfg("object"),
            "held_distance": 0.08,
        },
        weight=12.0,
    )
    # Sparse bonus for bringing the held plug within 5 cm of the goal position.
    plug_near_goal = RewTerm(
        func=mdp.object_near_goal,
        params={
            "threshold": 0.05,
            "command_name": "object_pose",
            "asset_cfg": SceneEntityCfg("object"),
            "held_distance": 0.08,
        },
        weight=5.0,
    )
    # Sparse bonus when the plug center is inside the socket bore.
    plug_inserted = RewTerm(
        func=mdp.plug_inserted,
        params={"depth_tol": _TARGET_HOLE_DEPTH / 2.0, "radius": _TARGET_HOLE_INNER / 2.0},
        weight=10.0,
    )

    # Stronger motion penalties for a slower, smoother arm (less cable clipping).
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1.5e-3)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-2e-3)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1e-6)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.5e-4)


@configclass
class TerminationsCfg:
    """Time-out plus a velocity-divergence guard that resets blown-up envs."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Plug dropped below the tabletop.
    plug_below_table = DoneTerm(
        func=mdp.object_com_below_minimum,
        params={"minimum_height": 0.0, "asset_cfg": SceneEntityCfg("object")},
    )

    # Thresholds well above the worst seen under extreme random actions (~30 rad/s, ~5 m/s).
    velocity_divergence = DoneTerm(
        func=mdp.assembly_velocity_out_of_bounds,
        params={"max_joint_vel": 25.0, "max_body_vel": 10.0},
    )


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

        # Proxy-coupled solver (mirrors FrankaCableEnvCfg): arm in MJWarp, plug/anchor/walls/cable
        # in VBD, gripper fingers exposed as VBD proxies so VBD's contact/friction grasps the plug.
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
