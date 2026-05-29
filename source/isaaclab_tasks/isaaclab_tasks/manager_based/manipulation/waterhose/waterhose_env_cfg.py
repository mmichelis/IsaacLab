# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Waterhose manipulation environment.

MDP config groups are copied from the Franka cable-plug task
(:mod:`isaaclab_tasks.manager_based.manipulation.lift_franka_soft`), whose
``mdp`` functions are reused by import.
"""

from __future__ import annotations

import os

import torch
from isaaclab_newton.physics import NewtonCollisionPipelineCfg
from isaaclab_newton.sim.spawners.materials.physics_materials_cfg import NewtonCableMaterialCfg
from isaaclab_visualizers.kit.kit_visualizer_cfg import KitVisualizerCfg
from isaaclab_visualizers.newton.newton_visualizer_cfg import NewtonVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.cable.cable_object_cfg import CableAttachmentCfg, CableObjectCfg
from isaaclab_contrib.deformable.newton_manager_cfg import (
    CoupledNewtonCfg,
    NewtonModelCfg,
    VBDSolverCfg,
)

from ..lift_franka_soft import mdp

WATERHOSE_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

# add_rod_graph places each segment's body frame at the edge's start node u (edge
# (u, v), +Z from u->v), so cable_local_pos=(0, 0, 0) welds at u. The anchor sits
# at that start node: cable001's last segment is edge (42, 43) -> u=42; cable002's
# first segment is edge (0, 1) -> u=0.
_FRIDGE_POS = (0.0, 0.0, 0.5)
_CABLE1_TAIL_NODE_42 = (-0.18810473382472992, 0.3453156650066376, -0.25986239314079285)
_CABLE1_TAIL_NODE_43 = (-0.18807558715343475, 0.3453156650066376, -0.2473306804895401)
_CABLE2_HEAD_NODE_0 = (-0.18045400083065033, 0.3453156650066376, -0.24754305183887482)
_CABLE2_HEAD_NODE_1 = (-0.18038532137870789, 0.3453156650066376, -0.25747784972190857)
_CABLE1_ANCHOR_NODE = _CABLE1_TAIL_NODE_42
_CABLE2_ANCHOR_NODE = _CABLE2_HEAD_NODE_1


##
# Scene
##


@configclass
class WaterhoseSceneCfg(InteractiveSceneCfg):
    """Cable + plug with the cable tail pinned to static anchors; sky light and ground."""

    ### Static fridge body
    fridge = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Fridge",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(WATERHOSE_ASSETS_DIR, "fridge", "fridge.usda"),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=_FRIDGE_POS),
    )

    ### Cable 1
    plug1 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Plug1",
        spawn=sim_utils.UsdFileCfg(usd_path=os.path.join(WATERHOSE_ASSETS_DIR, "fridge", "cable", "plug_mesh001.usda")),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.38398558, 0.34585292, 0.5 - 0.36874688),
            rot=(0.0, -0.57096256, 0.0, 0.8209761),
        ),
    )

    cable1 = CableObjectCfg(
        prim_path="/World/envs/env_.*/Cable1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(WATERHOSE_ASSETS_DIR, "fridge", "cable", "cable001.usda"),
            physics_material=NewtonCableMaterialCfg(
                stretch_stiffness=1e6,
                bend_stiffness=2e1,
                stretch_damping=1e-5,
                bend_damping=1e0,
                density=1000.0,
            ),
        ),
        init_state=CableObjectCfg.InitialStateCfg(
            pos=_FRIDGE_POS,
        ),
        attachments=[
            CableAttachmentCfg(
                target_prim_path="/World/envs/env_.*/Plug1",
                cable_anchor=0,
                cable_local_pos=(0.0, 0.0, 0.022),  # the head node is 22mm along +Z from the head body center
            ),
            CableAttachmentCfg(
                target_prim_path="/World/envs/env_.*/Fridge",
                cable_anchor=42,
                target_local_pos=_CABLE1_ANCHOR_NODE,  # fixed node 42 in the fridge's source frame
            ),
        ],
    )

    ### Cable 2
    plug2 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Plug2",
        spawn=sim_utils.UsdFileCfg(usd_path=os.path.join(WATERHOSE_ASSETS_DIR, "fridge", "cable", "plug_mesh001.usda")),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.00921878, 0.34529759, 0.5 - 0.37485825),
            rot=(0.0, 0.52994014, 0.0, 0.84803505),
        ),
    )

    cable2 = CableObjectCfg(
        prim_path="/World/envs/env_.*/Cable2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(WATERHOSE_ASSETS_DIR, "fridge", "cable", "cable002.usda"),
            physics_material=NewtonCableMaterialCfg(
                stretch_stiffness=1e6,
                bend_stiffness=2e1,
                stretch_damping=1e-5,
                bend_damping=1e0,
                density=1000.0,
            ),
        ),
        init_state=CableObjectCfg.InitialStateCfg(
            pos=_FRIDGE_POS,
        ),
        attachments=[
            CableAttachmentCfg(
                target_prim_path="/World/envs/env_.*/Plug2",
                cable_anchor=-1,
                cable_local_pos=(0.0, 0.0, 0.022),  # the head node is 22mm along +Z from the head body center
            ),
            CableAttachmentCfg(
                target_prim_path="/World/envs/env_.*/Fridge",
                cable_anchor=1,
                target_local_pos=_CABLE2_ANCHOR_NODE,  # fixed node 1 in the fridge's source frame
            ),
        ],
    )

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
    """Placeholder per-env zero observation, shape [num_envs, 1]."""
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

    scene: WaterhoseSceneCfg = WaterhoseSceneCfg(num_envs=8, env_spacing=0.5, replicate_physics=True)
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
        # self.sim.gravity = (0.0, 0.0, 0.0)

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
