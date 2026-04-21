# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka cable-reach environment: grasp a cable handle and move it to a 6D target."""

from __future__ import annotations

import math

from isaaclab_physx.physics import PhysxCfg

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from . import mdp as cable_mdp
from .cable_asset_cfg import build_cable_articulation_cfg

# Stiffer PD gains (K=400, D=80) from the HIGH_PD preset give the IK-differential
# controller enough bandwidth to track commanded EE deltas. We override the preset's
# ``disable_gravity=True`` default: the cable task cares about lifting a weighted
# handle under gravity, and disabling gravity would give the policy a free lift.
# ``.copy()`` deep-copies so modifying ``spawn.rigid_props`` does not leak into the
# module-level ``FRANKA_PANDA_HIGH_PD_CFG`` singleton used by other tasks.
FRANKA_CABLE_REACH_CFG = FRANKA_PANDA_HIGH_PD_CFG.copy()
FRANKA_CABLE_REACH_CFG.spawn.rigid_props.disable_gravity = False

##
# Scene
##


# Handle rest position on the table surface. Matches the lift task's world-frame
# convention where the ground plane sits at z=-1.05 and the table's top surface is
# approximately at z=0. See ``ObjectTableSceneCfg`` in the lift task for reference.
HANDLE_REST_POS = (0.45, 0.0, 0.02)


# Cached SceneEntityCfg specs. These are passed in term ``params`` so the manager
# resolves ``joint_ids`` before the term function is called — the function defaults
# alone would stay unresolved.
#
# The cable cfg does not carry ``body_names``: the handle is the articulation root
# (the ``<freejoint/>`` in the generated MJCF sits on the handle body), so the handle
# pose is read via ``root_pos_w`` / ``root_quat_w`` rather than a body-indexed lookup.
_CABLE = SceneEntityCfg("cable")
_ROBOT_FINGERS = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"])
_ROBOT_ARM = SceneEntityCfg("robot", joint_names=["panda_joint.*"])


def _ee_marker_cfg():
    cfg = FRAME_MARKER_CFG.copy()
    cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    cfg.prim_path = "/Visuals/FrameTransformer"
    return cfg


@configclass
class CableReachSceneCfg(InteractiveSceneCfg):
    """Scene: Franka on a table with a procedurally spawned cable lying on top."""

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        # Rotation is (x, y, z, w). (0, 0, 0.707, 0.707) = +90° around world z,
        # matching the lift task's table orientation (spins the table yaw-wise, keeps
        # the top surface horizontal at z=0). Earlier (0.707, 0, 0, 0.707) was +90°
        # around x which tipped the table onto its side.
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.0, 0.0, 0.707, 0.707)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        ),
    )

    robot: ArticulationCfg = FRANKA_CABLE_REACH_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    cable: ArticulationCfg = build_cable_articulation_cfg(
        prim_path="{ENV_REGEX_NS}/Cable",
        init_pos=HANDLE_REST_POS,
        # IsaacLab quaternion convention is (x, y, z, w). (0,0,0,1) is identity.
        init_rot=(0.0, 0.0, 0.0, 1.0),
        link_radius=0.005,  # 10 mm diameter — visible in the viewer against the Franka
    )

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=_ee_marker_cfg(),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.1034)),
            ),
        ],
    )


##
# MDP
##


@configclass
class CommandsCfg:
    """6D target pose for the cable handle, expressed in the robot root frame."""

    handle_pose = base_mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="panda_hand",
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=base_mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.35, 0.55),
            pos_y=(-0.15, 0.15),
            pos_z=(0.15, 0.40),
            # Narrowed from ±π/3 to ±π/6 — ±π/3 on all three axes is largely
            # unreachable when pinch-grasping a horizontal cable handle, and the
            # orientation reward was silently pulling the policy toward infeasible
            # targets. Widen again once position tracking is reliable.
            roll=(-math.pi / 6, math.pi / 6),
            pitch=(-math.pi / 6, math.pi / 6),
            yaw=(-math.pi / 6, math.pi / 6),
        ),
    )


@configclass
class ActionsCfg:
    # Task-space IK-differential control: the policy outputs a 6D EE-pose delta
    # (Δpos[m], Δaxis-angle[rad]) relative to the *current* EE pose. This removes the
    # joint-space default-pose anchor that made lifting hard under the previous
    # ``JointPositionActionCfg`` — the policy no longer needs to command large
    # joint-offset actions just to hold the grasp configuration.
    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
        ),
        # Per-step EE delta: 5 cm translation, 0.1 rad rotation at 50 Hz control.
        scale=(0.05, 0.05, 0.05, 0.1, 0.1, 0.1),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
    )
    gripper_action = base_mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=base_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=base_mdp.joint_vel_rel)
        # End-effector pose directly, rather than forcing the network to learn FK.
        ee_pose = ObsTerm(func=cable_mdp.ee_pose_in_robot_frame)
        handle_pose = ObsTerm(
            func=cable_mdp.handle_pose_in_robot_frame, params={"cable_cfg": _CABLE}
        )
        handle_vel = ObsTerm(func=cable_mdp.handle_velocity, params={"cable_cfg": _CABLE})
        ee_to_handle = ObsTerm(
            func=cable_mdp.ee_to_handle_position, params={"cable_cfg": _CABLE}
        )
        handle_to_target = ObsTerm(
            func=cable_mdp.handle_to_target_position, params={"cable_cfg": _CABLE}
        )
        # EE→target delta short-circuits the "sum two vectors" burden the network
        # would otherwise take on (ee_to_handle + handle_to_target).
        ee_to_target = ObsTerm(func=cable_mdp.ee_to_target_position)
        # Axis-angle rotation from handle to target — analogous to the position
        # delta, gives the network a direct orientation-error signal.
        orientation_error = ObsTerm(
            func=cable_mdp.target_orientation_error, params={"cable_cfg": _CABLE}
        )
        target_pose = ObsTerm(func=base_mdp.generated_commands, params={"command_name": "handle_pose"})
        # Exposes the (binary) grasp gate so the critic can attribute the discrete
        # reward-stack transitions that share this gate.
        grasped = ObsTerm(
            func=cable_mdp.grasp_indicator,
            params={"robot_cfg": _ROBOT_FINGERS, "cable_cfg": _CABLE},
        )
        last_action = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_scene = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")

    reset_cable = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "yaw": (-math.pi / 6, math.pi / 6),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cable"),
        },
    )

    reset_robot_joints = EventTerm(
        func=base_mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.9, 1.1),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": _ROBOT_ARM,
        },
    )


@configclass
class RewardsCfg:
    """Staged dense reward: reach → grasp → lift → move-to-target."""

    # Stage 1: approach the handle.
    reach_handle = RewTerm(
        func=cable_mdp.ee_to_handle_distance_tanh,
        params={"std": 0.1, "cable_cfg": _CABLE},
        weight=1.0,
    )

    # Stage 2: grasp bonus.
    grasp = RewTerm(
        func=cable_mdp.is_grasped,
        params={"robot_cfg": _ROBOT_FINGERS, "cable_cfg": _CABLE},
        weight=2.0,
    )

    # Stage 3: continuous lift-height reward — a dedicated bootstrapping signal that
    # rewards raising the handle off the table (gated by grasp). Without this, the
    # policy gets stuck in the "grasp and sit still" optimum: the target-tracking
    # gradient alone is not enough, because a first attempt to lift briefly perturbs
    # the handle horizontally before it rises, and PPO's short-horizon advantage
    # punishes the dip. This term gives purely-vertical credit for any lift progress.
    lift_progress = RewTerm(
        func=cable_mdp.lift_progress,
        params={
            "rest_height": 0.02,
            "max_lift": 0.3,
            "robot_cfg": _ROBOT_FINGERS,
            "cable_cfg": _CABLE,
        },
        weight=5.0,
    )

    # Stage 3+4: target-position tracking with a coarse and a fine tanh. Weights
    # chosen to dominate the reach+grasp signals so the policy has a clear incentive
    # to move the handle to the target rather than settle into "grasp and sit still".
    # Mirrors the Franka cube-lift task's proven balance (coarse:fine ≈ 16:5).
    target_position_coarse = RewTerm(
        func=cable_mdp.handle_target_position_tanh,
        params={
            "std": 0.3,
            "command_name": "handle_pose",
            "robot_cfg": _ROBOT_FINGERS,
            "cable_cfg": _CABLE,
        },
        weight=16.0,
    )

    target_position_fine = RewTerm(
        func=cable_mdp.handle_target_position_tanh,
        params={
            "std": 0.05,
            "command_name": "handle_pose",
            "robot_cfg": _ROBOT_FINGERS,
            "cable_cfg": _CABLE,
        },
        weight=5.0,
    )

    # Stage 4: orientation target tracking (gated by is_grasped). Deliberately soft
    # (wide std, low weight) until position tracking is reliable — previously the
    # dominant term pulled the policy toward infeasible ±π/3 targets and starved the
    # position gradient.
    target_orientation = RewTerm(
        func=cable_mdp.handle_target_orientation_tanh,
        params={
            "std": 1.0,
            "command_name": "handle_pose",
            "robot_cfg": _ROBOT_FINGERS,
            "cable_cfg": _CABLE,
        },
        weight=0.5,
    )

    # Sparse success bonus. Thresholds relaxed from (0.02 m / 0.1 rad) to
    # (0.05 m / 0.3 rad) so the bonus fires occasionally during early training and
    # provides a real credit-assignment signal — the tighter thresholds were
    # effectively untrainable from cold start.
    success = RewTerm(
        func=cable_mdp.success_bonus,
        params={
            "pos_threshold": 0.05,
            "rot_threshold": 0.3,
            "minimal_height": 0.04,
            "command_name": "handle_pose",
            "robot_cfg": _ROBOT_FINGERS,
            "cable_cfg": _CABLE,
        },
        weight=10.0,
    )

    # Regularization. Kept gentle — previous values (-1e-2 / -1e-3) made "don't
    # move" the dominant local optimum and collapsed exploration before the policy
    # ever discovered moving the handle toward the target.
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=base_mdp.joint_vel_l2,
        params={"asset_cfg": _ROBOT_ARM},
        weight=-1e-4,
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)

    # Physics-solver failure detection. ``time_out=False`` marks the episode as a
    # genuine failure, so the framework resets the env instead of letting NaN state
    # persist and poison subsequent observations.
    invalid_state = DoneTerm(
        func=cable_mdp.invalid_cable_state,
        params={"cable_cfg": _CABLE, "robot_cfg": SceneEntityCfg("robot")},
        time_out=False,
    )


##
# Top-level env cfg
##


@configclass
class FrankaCableReachEnvCfg(ManagerBasedRLEnvCfg):
    """Franka grasps a cable handle and moves it to a randomized 6D target pose."""

    # replicate_physics=False forces per-env asset re-instantiation rather than the
    # physics-backend clone path. Needed because the URDF-imported cable articulation
    # has a floating-base root joint whose body rel isn't rebound by the cloner.
    scene: CableReachSceneCfg = CableReachSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 0.01  # 100 Hz physics, 50 Hz control
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysxCfg(
            bounce_threshold_velocity=0.01,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=16 * 1024,
            friction_correlation_distance=0.00625,
        )
