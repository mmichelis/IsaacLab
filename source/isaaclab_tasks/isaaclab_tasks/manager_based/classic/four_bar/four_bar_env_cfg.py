# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

from isaaclab_newton.physics import KaminoSolverCfg, NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.manager_based.classic.four_bar.mdp as mdp
from isaaclab_tasks.utils import PresetCfg


def _newton_four_bar_usd_path() -> str:
    """Return Newton's packaged four-bar linkage USD path."""
    spec = find_spec("newton")
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError("The 'newton' package is required for the four-bar linkage asset.")
    return str(Path(spec.origin).resolve().parent / "examples" / "assets" / "boxes_fourbar.usda")


def _four_bar_kamino_cfg() -> NewtonCfg:
    """Return the Kamino physics configuration for the four-bar linkage."""
    return NewtonCfg(
        solver_cfg=KaminoSolverCfg(
            integrator="moreau",
            use_collision_detector=True,
            sparse_jacobian=True,
            constraints_alpha=0.1,
            padmm_max_iterations=100,
            padmm_primal_tolerance=1e-4,
            padmm_dual_tolerance=1e-4,
            padmm_compl_tolerance=1e-4,
            padmm_rho_0=0.05,
            padmm_eta=1e-5,
            padmm_use_acceleration=True,
            padmm_warmstart_mode="containers",
            padmm_contact_warmstart_method="geom_pair_net_force",
            padmm_use_graph_conditionals=False,
            collision_detector_pipeline="unified",
            collision_detector_max_contacts_per_pair=8,
        ),
        num_substeps=2,
        debug_mode=False,
        use_cuda_graph=False,
        usd_joint_ordering=None,
        skip_validation_joints=True,
    )


FOUR_BAR_USD_PATH = _newton_four_bar_usd_path()


@configclass
class FourBarPhysicsCfg(PresetCfg):
    """Physics backend presets for the four-bar linkage."""

    default: NewtonCfg = _four_bar_kamino_cfg()
    newton_kamino: NewtonCfg = _four_bar_kamino_cfg()


@configclass
class FourBarSceneCfg(InteractiveSceneCfg):
    """Configuration for the four-bar linkage scene."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(100.0, 100.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=2.0,
                dynamic_friction=2.0,
                restitution=0.0,
            ),
        ),
    )

    robot: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=FOUR_BAR_USD_PATH,
            func="isaaclab_tasks.manager_based.classic.four_bar.spawn:spawn_four_bar_robot",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.FourBarJointEffortActionCfg(
        asset_name="robot",
        joint_ids=(0, 2),
        scale=0.25,
        gait_effort=16.0,
        gait_period_s=0.75,
        action_clip=(-1.0, 1.0),
        effort_clip=(-16.0, 16.0),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy group."""

        body_pos_rel = ObsTerm(func=mdp.four_bar_body_pos_rel)
        body_lin_vel = ObsTerm(func=mdp.four_bar_body_lin_vel)
        body_ang_vel = ObsTerm(func=mdp.four_bar_body_ang_vel)
        joint_pos = ObsTerm(func=mdp.four_bar_joint_pos)
        joint_vel = ObsTerm(func=mdp.four_bar_joint_vel)
        gait_phase = ObsTerm(func=mdp.gait_phase, params={"period_s": 0.75})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for reset events."""

    material = EventTerm(
        func=mdp.set_four_bar_material,
        mode="startup",
        params={"friction": 2.0, "restitution": 0.0},
    )

    reset_linkage = EventTerm(
        func=mdp.reset_four_bar_configuration,
        mode="reset",
        params={
            "joint_angle_range": (0.0, 0.0),
            "joint_velocity_range": (0.0, 0.0),
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {},
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    forward_progress = RewTerm(func=mdp.body_progress_x, weight=300.0)
    forward_velocity = RewTerm(func=mdp.root_lin_vel_x, weight=5.0)
    gait_action_alignment = RewTerm(
        func=mdp.gait_action_alignment,
        weight=0.0,
        params={"period_s": 0.75},
    )
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.005)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
    invalid_state = RewTerm(
        func=mdp.invalid_state_penalty,
        weight=-20.0,
        params={"max_body_distance": 10.0, "max_body_velocity": 1000.0, "max_joint_velocity": 1000.0},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    invalid_state = DoneTerm(
        func=mdp.invalid_newton_state,
        params={"max_body_distance": 10.0, "max_body_velocity": 1000.0, "max_joint_velocity": 1000.0},
    )
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class FourBarEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the four-bar linkage locomotion environment."""

    scene: FourBarSceneCfg = FourBarSceneCfg(num_envs=16, env_spacing=2.0, clone_in_fabric=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        from isaaclab_visualizers.newton import NewtonVisualizerCfg  # noqa: PLC0415

        self.decimation = 2
        self.episode_length_s = 5.0
        self.viewer.eye = (0.75, -1.0, 0.55)
        self.viewer.lookat = (0.0, 0.0, 0.06)
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physics = FourBarPhysicsCfg()
        self.sim.visualizer_cfgs = NewtonVisualizerCfg(
            eye=(0.45, -0.65, 0.35),
            lookat=(0.0, 0.0, 0.16),
        )
