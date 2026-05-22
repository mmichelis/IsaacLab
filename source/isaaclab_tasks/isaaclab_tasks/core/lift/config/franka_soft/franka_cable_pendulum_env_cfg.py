# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Panda manipulating a Newton cable pendulum.

The cable is welded at one end to a kinematic anchor fixed above the tabletop
and at the other end to a rigid plug body. The RL task is to bring the plug to
a sampled target pose using the Franka end-effector.

This environment inherits aggressively from :class:`FrankaCableEnvCfg`: it
reuses the robot, actions, physics solver, and reward shape, and only adds the
two attachment bodies plus rewires reward and observation terms from the cable
to the plug.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.cable.cable_object_cfg import CableAttachmentCfg, CableObjectCfg
from isaaclab_contrib.deformable.newton_manager_cfg import CoupledNewtonCfg, ProxyCoupledMJWarpVBDSolverCfg

from isaaclab_newton.sim.spawners.materials import NewtonCableMaterialCfg

from . import mdp
from .franka_cable_env_cfg import FrankaCableEnvCfg, _FrankaCableSceneCfg

##
# Cable / attachment geometry constants
##

# Number of cable segments. Matches the parent cable env so the same physics tuning applies.
_NUM_POINTS = 20

# Per-segment length [m]. Matches the parent cable env.
_SEGMENT_LENGTH = 0.02

# Cable width [m].
_CABLE_WIDTH = 0.01

# Anchor pose in the env-local frame [m]. Positioned above the tabletop, in front of the robot.
_ANCHOR_POS = (0.5, 0.0, 0.5)

# Plug rest pose [m]: directly below the anchor at the cable's natural extent.
_PLUG_INIT_POS = (_ANCHOR_POS[0], _ANCHOR_POS[1], _ANCHOR_POS[2] - (_NUM_POINTS - 1) * _SEGMENT_LENGTH)

# Plug body parameters. Mass is the midpoint of the demo's [0.005, 0.05] kg range.
_PLUG_RADIUS = 0.04
_PLUG_HEIGHT = 0.04
_PLUG_MASS = 0.02


def _y_axis_quat(angle_rad: float) -> tuple[float, float, float, float]:
    """Quaternion ``(x, y, z, w)`` for a rotation about +Y, matching :attr:`AssetBaseCfg.InitialStateCfg.rot`."""
    return (0.0, math.sin(0.5 * angle_rad), 0.0, math.cos(0.5 * angle_rad))


##
# Scene
##


@configclass
class _FrankaCablePendulumSceneCfg(_FrankaCableSceneCfg):
    """Scene for the Franka cable pendulum environment.

    Inherits ``robot``, ``ee_frame``, ``table``, ``ground``, ``sky_light`` and
    the actuator tuning from :class:`_FrankaCableSceneCfg`. Replaces the cable
    spawn to lay it out vertically and wire it to two new attachment bodies:
    a kinematic ``anchor`` above the tabletop and a rigid ``plug`` at the
    cable's other end.
    """

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

    plug: RigidObjectCfg = RigidObjectCfg(
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
            rot=_y_axis_quat(-math.pi / 2.0),
        ),
    )

    object: CableObjectCfg = CableObjectCfg(
        prim_path="/World/envs/env_.*/Cable",
        init_state=CableObjectCfg.InitialStateCfg(pos=_ANCHOR_POS),
        spawn=sim_utils.CableCfg(
            positions=[(0.0, 0.0, -i * _SEGMENT_LENGTH) for i in range(_NUM_POINTS)],
            width=_CABLE_WIDTH,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.85, 0.1)),
            physics_material=NewtonCableMaterialCfg(
                stretch_stiffness=1.0e6,
                stretch_damping=1.0e-1,
                bend_stiffness=5.0e-3,
                bend_damping=2.0e-3,
                density=100.0,
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
                cable_anchor=-1,
            ),
        ],
    )


##
# MDP overrides
##


@configclass
class CommandsCfg:
    """Plug goal pose sampled in the robot root frame.

    Ranges are tightened compared to :class:`FrankaCableEnvCfg.CommandsCfg` to keep
    targets inside the half-sphere reachable by the plug while the cable is
    anchored above the tabletop.
    """

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="panda_hand",
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.3, 0.6),
            pos_y=(-0.25, 0.25),
            pos_z=(0.05, 0.4),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
        goal_pose_visualizer_cfg=VisualizationMarkersCfg(
            prim_path="/Visuals/Command/goal_pose",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.02,
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
class ObservationsCfg:
    """Policy observations for the cable pendulum task."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        plug_position = ObsTerm(
            func=mdp.object_com_in_robot_root_frame,
            params={"asset_cfg": SceneEntityCfg("plug")},
        )
        target_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset events for the cable pendulum task."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.9, 1.1), "velocity_range": (0.0, 0.0)},
    )


@configclass
class RewardsCfg:
    """Reach-and-track reward shaping for the rigid plug."""

    reaching_plug = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.1, "asset_cfg": SceneEntityCfg("plug")},
        weight=5.0,
    )
    lifting_plug = RewTerm(
        func=mdp.object_lifted,
        params={"minimal_height": 0.04, "asset_cfg": SceneEntityCfg("plug")},
        weight=5.0,
    )
    plug_goal_tracking = RewTerm(
        func=mdp.object_com_goal_distance,
        params={
            "std": 0.3,
            "minimal_height": 0.05,
            "command_name": "object_pose",
            "asset_cfg": SceneEntityCfg("plug"),
        },
        weight=16.0,
    )
    plug_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_com_goal_distance,
        params={
            "std": 0.05,
            "minimal_height": 0.05,
            "command_name": "object_pose",
            "asset_cfg": SceneEntityCfg("plug"),
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
class FrankaCablePendulumEnvCfg(FrankaCableEnvCfg):
    """Franka Panda manipulating a cable with a fixed anchor and a rigid plug."""

    scene: _FrankaCablePendulumSceneCfg = _FrankaCablePendulumSceneCfg(
        num_envs=128, env_spacing=2.5, replicate_physics=True
    )
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # The proxy-coupled solver from FrankaCableEnvCfg is reused. Both the kinematic anchor
        # and the rigid plug are connected to the cable via VBD attachments, so the solver
        # needs to know about them on the VBD side.
        assert isinstance(self.sim.physics, CoupledNewtonCfg)
        solver_cfg = self.sim.physics.solver_cfg
        assert isinstance(solver_cfg, ProxyCoupledMJWarpVBDSolverCfg)
        solver_cfg.vbd_bodies = [
            SceneEntityCfg("object"),
            SceneEntityCfg("anchor"),
            SceneEntityCfg("plug"),
        ]
