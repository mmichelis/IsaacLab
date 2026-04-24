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
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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
# Stronger gripper (2e3 → 5e3 Nm/rad) gives ~50 N total clamp force at a 1 cm gap
# instead of ~20 N. For a cable weight of ~1 N and expected lift acceleration
# forces of ~15 N at the current IK speeds, this turns a ~25% margin into a ~3×
# safety factor — the grip should now survive the motion transients that were
# slipping the handle out during lift attempts.
FRANKA_CABLE_REACH_CFG.actuators["panda_hand"].stiffness = 5e3
FRANKA_CABLE_REACH_CFG.actuators["panda_hand"].damping = 2e2

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

    # NOTE: kept the name ``cable`` for minimal churn, but this is now just a
    # standalone rigid box (no chain attached). Simplifies the physics by removing
    # the articulated cable that was adding mass/inertia resistance during lift
    # — once grasp-and-move works on the bare handle, the cable can be added back.
    cable: RigidObjectCfg = build_cable_articulation_cfg(
        prim_path="{ENV_REGEX_NS}/Cable",
        init_pos=HANDLE_REST_POS,
        # IsaacLab quaternion convention is (x, y, z, w). (0,0,0,1) is identity.
        init_rot=(0.0, 0.0, 0.0, 1.0),
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
    # Joint-space control: policy outputs 7-dim arm-joint-delta + 1-dim gripper. This
    # matches the proven Franka cube-lift task. Earlier I tried IK-differential control
    # with ``use_relative_mode=True``, but it had a fatal issue: with zero action the
    # EE drifts ~10 cm in x and 12 cm in z over 1 s under gravity (because each step
    # re-reads the drifted current pose and sets the target = current + Δ). The policy
    # can't learn "hold still" because there's no anchor — any attempt is swamped by
    # sag drift. In joint-space action, ``action = 0`` commands the default joint pose
    # which the high PD gains actively hold against gravity, giving the policy a stable
    # rest state to learn from.
    arm_action = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=0.5,
        use_default_offset=True,
    )
    gripper_action = base_mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.05},
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

    # Stage 1a: coarse approach reward — gradient from far away, pulls the EE
    # toward the handle. Tanh std of 0.15 means the reward is still meaningful
    # from ~20 cm out.
    reach_handle_coarse = RewTerm(
        func=cable_mdp.ee_to_handle_distance_tanh,
        params={"std": 0.15, "cable_cfg": _CABLE},
        weight=1.0,
    )

    # Stage 1b: fine-grained centering — sharp gradient near the handle forces
    # the EE to actually reach the handle's CENTER (not just "within 5 cm of
    # somewhere on it"). Without this the policy was hovering near the handle
    # and snapping the gripper shut off-center, missing the grasp. Std 0.03
    # means within ~3 cm of the handle center earns significant reward; beyond
    # that the term saturates to 0.
    reach_handle_fine = RewTerm(
        func=cable_mdp.ee_to_handle_distance_tanh,
        params={"std": 0.03, "cable_cfg": _CABLE},
        weight=2.0,
    )

    # Stage 2: grasp bonus — uses the LOOSE geometric check (proximity + fingers
    # in the gripping range), not the strict velocity-correlation ``is_grasped``.
    # The strict version was rejecting real grips during the brief finger-contact
    # transient (handle moves a few mm when fingers first touch, velocities
    # mismatch, check fails), so the policy learned "don't close — it only loses
    # my reach reward." The geometric check fires as soon as the policy
    # completes a close on a nearby handle, giving it a direct reward signal for
    # the close action. Physical verification still happens via ``handle_tracks_
    # gripper`` and ``handle_above_table`` (which use strict ``is_grasped``
    # internally), which only light up if the grip is actually holding.
    grasp = RewTerm(
        func=cable_mdp.is_grasped_geometric,
        params={"robot_cfg": _ROBOT_FINGERS, "cable_cfg": _CABLE},
        weight=2.0,
    )

    # Stage 2b: "don't drag on the table" penalty. The policy has been letting
    # the gripper sag onto the table surface instead of hovering above. We
    # penalize the fingertip z-position being below 1 cm above the table
    # surface — the handle center sits at z ≈ 0.02, so this threshold leaves
    # plenty of room for a real grasp while punishing the "rest the fingers on
    # the table" behaviour.
    ee_too_low = RewTerm(
        func=cable_mdp.ee_below_threshold,
        params={"min_z": 0.01},
        weight=-5.0,
    )

    # Stage 2c: "keep the gripper OPEN unless you're on the handle" penalty.
    # Continuous: proportional to how closed the fingers are × how far the
    # gripper is from a real grasp. Zero when the gripper is open; zero when
    # the gripper is closed on the handle; max (≈ 1.0) when the gripper is
    # closed on empty air. Directly addresses the policy's learned "snap shut
    # on approach" behaviour that makes the grasp reward impossible to earn.
    gripper_closed_without_grasp = RewTerm(
        func=cable_mdp.gripper_closed_without_grasp,
        params={
            "open_threshold": 0.035,
            "robot_cfg": _ROBOT_FINGERS,
            "cable_cfg": _CABLE,
        },
        weight=-3.0,
    )

    # Stage 3a: reward the handle for TRACKING the gripper's velocity. Peak when
    # the handle and gripper share the same linear velocity while the gripper is
    # in motion AND the fingers are clamped on something near the EE. Fades away
    # when the handle slips, is pushed, or is flicked. Also fades to 0 when the
    # gripper is static — you can't farm this by grabbing and sitting still.
    handle_tracks_gripper = RewTerm(
        func=cable_mdp.handle_tracks_gripper,
        params={
            "lin_vel_std": 0.10,
            "motion_scale": 0.05,
            "robot_cfg": _ROBOT_FINGERS,
            "cable_cfg": _CABLE,
        },
        weight=3.0,
    )

    # Stage 3b: explicit lift-off-the-table reward. ``handle_tracks_gripper`` pays
    # for ANY velocity-matched motion, including horizontal dragging across the
    # table — previous runs showed the policy learning exactly that. Adding this
    # altitude term gives the policy an explicit gradient telling it to lift the
    # handle off the table, which it needs to do anyway to reach the ≥15 cm
    # minimum target altitude. Gated on the strict ``is_grasped`` (velocity-
    # correlation check) so fake lifts don't register.
    handle_above_table = RewTerm(
        func=cable_mdp.handle_above_table,
        params={
            "rest_height": 0.02,
            "max_lift": 0.20,
            "robot_cfg": _ROBOT_FINGERS,
            "cable_cfg": _CABLE,
        },
        weight=5.0,
    )

    # Sparse binary "handle is off the table" reward. Fires +1 per step once the
    # handle centre is more than 5 cm above the table (well clear of rest at
    # ~2 cm, so small bumps don't count) AND the strict ``is_grasped`` holds
    # (rigid-body velocity correlation — rejects flicks/pushes). Complements the
    # continuous ``handle_above_table`` reward with an all-or-nothing "you've
    # actually got it off the table" signal — useful because step-function
    # bonuses are what RL agents LATCH onto once discovered.
    handle_lifted = RewTerm(
        func=cable_mdp.is_lifted,
        params={
            "minimal_height": 0.05,
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
        weight=10.0,
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
