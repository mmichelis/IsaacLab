# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Waterhose manipulation environment skeleton.

The MDP config groups below (commands, actions, observations, events, rewards,
terminations) are copied from the Franka cable-plug task
(:mod:`isaaclab_tasks.manager_based.manipulation.lift_franka_soft`) so they can
be edited freely to fit this environment. The reward/observation *functions*
themselves are still imported from ``lift_franka_soft.mdp``.

The scene (:class:`WaterhoseSceneCfg`) is intentionally empty apart from a sky
light and a ground plane; all manipulated assets are expected to be loaded
externally. Because the
config groups and the coupled MJWarp+VBD physics block reference scene entities
by name (``robot``, ``object``, ``anchor``, ``cable``,
``target_hole_{top,bottom,left,right}``), the module imports cleanly, but the
environment will only build and run once assets with those names are added to
the scene.
"""

from __future__ import annotations

import torch
from isaaclab_newton.physics import NewtonCollisionPipelineCfg
from isaaclab_visualizers.kit.kit_visualizer_cfg import KitVisualizerCfg
from isaaclab_visualizers.newton.newton_visualizer_cfg import NewtonVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.deformable.newton_manager_cfg import (
    CoupledNewtonCfg,
    NewtonModelCfg,
    VBDSolverCfg,
)

from ..lift_franka_soft import mdp

##
# Scene
##


@configclass
class WaterhoseSceneCfg(InteractiveSceneCfg):
    """Minimal scene: a sky light and a ground plane. Manipulated assets are loaded externally."""

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -1.05]),
        spawn=GroundPlaneCfg(),
    )


##
# MDP overrides
##


@configclass
class CommandsCfg:
    """Cable goal pose sampled in the robot root frame."""

    # object_pose = mdp.UniformPoseCommandCfg(
    #     asset_name="robot",
    #     body_name="panda_hand",
    #     resampling_time_range=(5.0, 5.0),
    #     debug_vis=True,
    #     ranges=mdp.UniformPoseCommandCfg.Ranges(
    #         pos_x=(0.4, 0.6),
    #         pos_y=(-0.25, 0.25),
    #         pos_z=(0.1, 0.3),
    #         roll=(0.0, 0.0),
    #         pitch=(0.0, 0.0),
    #         yaw=(0.0, 0.0),
    #     ),
    #     goal_pose_visualizer_cfg=VisualizationMarkersCfg(
    #         prim_path="/Visuals/Command/goal_pose",
    #         markers={
    #             "sphere": sim_utils.SphereCfg(
    #                 radius=0.02,
    #                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.2), opacity=0.01),
    #             ),
    #         },
    #     ),
    #     # Hide the EE frame
    #     current_pose_visualizer_cfg=VisualizationMarkersCfg(
    #         prim_path="/Visuals/Command/body_pose",
    #         markers={
    #             "sphere": sim_utils.SphereCfg(
    #                 radius=1e-6,
    #                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0), opacity=0.0),
    #             ),
    #         },
    #     ),
    # )


@configclass
class ActionsCfg:
    """7-dim absolute end-effector pose (xyz + quaternion) via differential IK + 1-dim binary gripper."""

    # arm_action = DifferentialInverseKinematicsActionCfg(
    #     asset_name="robot",
    #     joint_names=["panda_joint.*"],
    #     body_name="panda_hand",
    #     controller=DifferentialIKControllerCfg(
    #         command_type="pose",
    #         use_relative_mode=False,
    #         ik_method="dls",
    #         ik_params={"lambda_val": 0.05},
    #     ),
    #     body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
    # )
    # gripper_action = mdp.BinaryJointPositionActionCfg(
    #     asset_name="robot",
    #     joint_names=["panda_finger.*"],
    #     open_command_expr={"panda_finger_.*": 0.05},
    #     close_command_expr={"panda_finger_.*": 0.005},
    # )


def dummy_obs(env) -> torch.Tensor:
    """Placeholder observation term: a per-env zero scalar, shape [num_envs, 1].

    Returns a valid 2-D tensor the observation manager can concatenate. Replace
    with real observation terms once the scene assets are loaded.
    """
    return torch.zeros(env.num_envs, 1, device=env.device)

@configclass
class ObservationsCfg:
    """Policy observations for the cable plug task."""

    @configclass
    class PolicyCfg(ObsGroup):
        # joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # plug_position = ObsTerm(
        #     func=mdp.object_com_in_robot_root_frame,
        #     params={"asset_cfg": SceneEntityCfg("object")},
        # )
        # target_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        # actions = ObsTerm(func=mdp.last_action)
        actions = ObsTerm(func=dummy_obs)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset events for the cable plug task."""

    # reset_robot_joints = EventTerm(
    #     func=mdp.reset_joints_by_scale,
    #     mode="reset",
    #     params={"position_range": (0.9, 1.1), "velocity_range": (0.0, 0.0)},
    # )


@configclass
class RewardsCfg:
    """Reach-and-track reward shaping for the rigid plug."""

    # joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-2)
    # joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1e-4)
    # joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1e-4)


@configclass
class TerminationsCfg:
    """Time-out only; the cable is anchored so the plug cannot escape the workspace."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##


@configclass
class WaterhoseEnvCfg(ManagerBasedRLEnvCfg):
    """Waterhose environment reusing the cable-plug MDP on an externally loaded scene."""

    scene: WaterhoseSceneCfg = WaterhoseSceneCfg(num_envs=8, env_spacing=2.5, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        # general settings
        self.decimation = 1
        self.episode_length_s = 1.0

        # simulation settings
        self.sim.dt = 1 / 60.0
        self.sim.render_interval = self.decimation
        self.sim.gravity = (0.0, 0.0, -9.81)

        view = dict(eye=(1.4, 1.0, 0.6), lookat=(0.35, 0.0, 0.1), window_width=1600, window_height=1600)
        self.sim.visualizer_cfgs = [KitVisualizerCfg(**view), NewtonVisualizerCfg(**view)]

        # self.sim.physics = CoupledNewtonCfg(
        #     scene_cfg=self.scene,
        #     solver_cfg=CoupledAdmmSolverCfg(
        #         src_solver_cfg=MJWarpSolverCfg(
        #             cone="elliptic",
        #             ls_parallel=True,
        #             ls_iterations=20,
        #             integrator="implicitfast",
        #         ),
        #         dst_solver_cfg=VBDSolverCfg(iterations=20, rigid_avbd_beta=1e3, rigid_contact_k_start=1e3),
        #         src_bodies=[SceneEntityCfg("robot")],
        #         dst_bodies=[
        #             SceneEntityCfg("object"),
        #             SceneEntityCfg("anchor"),
        #             SceneEntityCfg("cable"),
        #         ],
        #         iterations=5,
        #         rho=30.0,
        #         gamma=0.1,
        #         baumgarte=0.005,
        #         contact_distance=0.003,
        #     ),
        #     model_cfg=NewtonModelCfg(
        #         shape_material_ke=1e5,
        #         shape_material_kd=1e-2,
        #         shape_material_mu=1.0,
        #     ),
        #     num_substeps=10,
        # )
        self.sim.physics = CoupledNewtonCfg(
            solver_cfg=VBDSolverCfg(
                iterations=20,
                rigid_body_contact_buffer_size=1024,
                rigid_contact_k_start=1.0e1,
                rigid_avbd_beta=1e2,
            ),
            num_substeps=8,
            collision_cfg=NewtonCollisionPipelineCfg(rigid_contact_max=65536),
            model_cfg=NewtonModelCfg(
                shape_material_ke=1.0e3,
                shape_material_kd=1.0e0,
                shape_material_mu=1.0,
            ),
        )
