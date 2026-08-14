# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Grasp a cable at its center, lift it, and launch it outside the table with differential IK.

.. code-block:: bash

    PYTHONPATH="$PWD/source/isaaclab_tasks:$PYTHONPATH" ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION=1 \
        uv run --extra isaacsim --extra video python \
        scripts/environments/state_machine/push_franka_cable.py --viz none
"""

import argparse
import sys
from collections.abc import Sequence

import gymnasium as gym
import torch
import warp as wp
from isaaclab_newton.sim.schemas import MujocoJointCfg

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

TASK_NAME = "Isaac-Lift-Cable-Franka-Camera"
LIFT_HEIGHT = 0.125
DROP_TARGET_Y = 0.55
GRASP_OFFSET_Z = -0.002
CAMERA_EYE = (2.25, 2.0, 0.3)
CAMERA_LOOKAT = (0.0, 0.5, -0.2)

parser = argparse.ArgumentParser(description="Lift a cable and launch it outside a table with a Franka robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--num_steps", type=int, default=305, help="Number of environment steps to run.")
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

wp.init()

RELEASE_FRACTION = wp.constant(0.6)


class GripperState:
    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class PickAndDropSmState:
    REST = wp.constant(0)
    APPROACH_ABOVE_CABLE = wp.constant(1)
    APPROACH_CABLE = wp.constant(2)
    GRASP_CABLE = wp.constant(3)
    LIFT_CABLE = wp.constant(4)
    MOVE_OUTSIDE_TABLE = wp.constant(5)
    HOLD = wp.constant(6)


class PickAndDropSmWaitTime:
    REST = wp.constant(0.0)
    APPROACH_ABOVE_CABLE = wp.constant(0.5)
    APPROACH_CABLE = wp.constant(0.5)
    GRASP_CABLE = wp.constant(0.75)
    LIFT_CABLE = wp.constant(1.5)
    MOVE_OUTSIDE_TABLE = wp.constant(2.0)


@wp.func
def position_reached(current_pose: wp.transform, desired_pose: wp.transform, threshold: float) -> bool:
    return (
        wp.length(wp.transform_get_translation(current_pose) - wp.transform_get_translation(desired_pose)) < threshold
    )


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    grasp_pose: wp.array(dtype=wp.transform),
    lift_pose: wp.array(dtype=wp.transform),
    drop_pose: wp.array(dtype=wp.transform),
    desired_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    approach_offset: wp.array(dtype=wp.transform),
    position_threshold: float,
):
    tid = wp.tid()
    state = sm_state[tid]

    if state == PickAndDropSmState.REST:
        desired_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if sm_wait_time[tid] >= PickAndDropSmWaitTime.REST:
            sm_state[tid] = PickAndDropSmState.APPROACH_ABOVE_CABLE
            sm_wait_time[tid] = 0.0
    elif state == PickAndDropSmState.APPROACH_ABOVE_CABLE:
        desired_ee_pose[tid] = wp.transform_multiply(approach_offset[tid], grasp_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        if position_reached(ee_pose[tid], desired_ee_pose[tid], position_threshold):
            if sm_wait_time[tid] >= PickAndDropSmWaitTime.APPROACH_ABOVE_CABLE:
                sm_state[tid] = PickAndDropSmState.APPROACH_CABLE
                sm_wait_time[tid] = 0.0
    elif state == PickAndDropSmState.APPROACH_CABLE:
        desired_ee_pose[tid] = grasp_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if position_reached(ee_pose[tid], desired_ee_pose[tid], position_threshold):
            if sm_wait_time[tid] >= PickAndDropSmWaitTime.APPROACH_CABLE:
                sm_state[tid] = PickAndDropSmState.GRASP_CABLE
                sm_wait_time[tid] = 0.0
    elif state == PickAndDropSmState.GRASP_CABLE:
        desired_ee_pose[tid] = grasp_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if sm_wait_time[tid] >= PickAndDropSmWaitTime.GRASP_CABLE:
            sm_state[tid] = PickAndDropSmState.LIFT_CABLE
            sm_wait_time[tid] = 0.0
    elif state == PickAndDropSmState.LIFT_CABLE:
        alpha = wp.min(sm_wait_time[tid] / PickAndDropSmWaitTime.LIFT_CABLE, 1.0)
        grasp_position = wp.transform_get_translation(grasp_pose[tid])
        lift_position = wp.transform_get_translation(lift_pose[tid])
        orientation = wp.transform_get_rotation(lift_pose[tid])
        desired_ee_pose[tid] = wp.transform(wp.lerp(grasp_position, lift_position, alpha), orientation)
        gripper_state[tid] = GripperState.CLOSE
        if sm_wait_time[tid] >= PickAndDropSmWaitTime.LIFT_CABLE:
            if position_reached(ee_pose[tid], desired_ee_pose[tid], position_threshold):
                sm_state[tid] = PickAndDropSmState.MOVE_OUTSIDE_TABLE
                sm_wait_time[tid] = 0.0
    elif state == PickAndDropSmState.MOVE_OUTSIDE_TABLE:
        alpha = wp.min(sm_wait_time[tid] / PickAndDropSmWaitTime.MOVE_OUTSIDE_TABLE, 1.0)
        lift_position = wp.transform_get_translation(lift_pose[tid])
        drop_position = wp.transform_get_translation(drop_pose[tid])
        orientation = wp.transform_get_rotation(drop_pose[tid])
        desired_ee_pose[tid] = wp.transform(wp.lerp(lift_position, drop_position, alpha), orientation)
        if alpha < RELEASE_FRACTION:
            gripper_state[tid] = GripperState.CLOSE
        else:
            gripper_state[tid] = GripperState.OPEN
        if sm_wait_time[tid] >= PickAndDropSmWaitTime.MOVE_OUTSIDE_TABLE:
            sm_state[tid] = PickAndDropSmState.HOLD
            sm_wait_time[tid] = 0.0
    elif state == PickAndDropSmState.HOLD:
        desired_ee_pose[tid] = drop_pose[tid]
        gripper_state[tid] = GripperState.OPEN

    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class PickAndDropCableSm:
    """Task-space state machine for picking up and launching a cable."""

    def __init__(self, dt: float, num_envs: int, device: torch.device | str, position_threshold: float = 0.01):
        self.num_envs = num_envs
        self.device = device
        self.position_threshold = position_threshold
        self.sm_dt = torch.full((num_envs,), dt, device=device)
        self.sm_state = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.sm_wait_time = torch.zeros(num_envs, device=device)
        self.desired_ee_pose = torch.zeros((num_envs, 7), device=device)
        self.gripper_state = torch.zeros(num_envs, device=device)
        self.approach_offset = torch.zeros((num_envs, 7), device=device)
        self.approach_offset[:, 2] = 0.05
        self.approach_offset[:, -1] = 1.0

        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.desired_ee_pose_wp = wp.from_torch(self.desired_ee_pose, wp.transform)
        self.gripper_state_wp = wp.from_torch(self.gripper_state, wp.float32)
        self.approach_offset_wp = wp.from_torch(self.approach_offset, wp.transform)

    def reset_idx(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = PickAndDropSmState.REST
        self.sm_wait_time[env_ids] = 0.0

    def compute(
        self, ee_pose: torch.Tensor, grasp_pose: torch.Tensor, lift_pose: torch.Tensor, drop_pose: torch.Tensor
    ) -> torch.Tensor:
        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                wp.from_torch(ee_pose.contiguous(), wp.transform),
                wp.from_torch(grasp_pose.contiguous(), wp.transform),
                wp.from_torch(lift_pose.contiguous(), wp.transform),
                wp.from_torch(drop_pose.contiguous(), wp.transform),
                self.desired_ee_pose_wp,
                self.gripper_state_wp,
                self.approach_offset_wp,
                self.position_threshold,
            ],
            device=self.device,
        )
        return torch.cat([self.desired_ee_pose, self.gripper_state.unsqueeze(-1)], dim=-1)


def main() -> None:
    env_cfg, _ = resolve_task_config(TASK_NAME, "")
    env_cfg.curriculum.gravity = None
    env_cfg.events.variable_gravity = None
    env_cfg.sim.gravity = (0.0, 0.0, -0.4)
    env_cfg.scene.cable.spawn.physics_material.thickness = 0.015
    env_cfg.sim.physics.num_substeps *= 2
    env_cfg.scene.robot.spawn.joint_drive_props = [MujocoJointCfg(actuatorgravcomp=True)]
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.episode_length_s = 15.0
    for term_name in list(vars(env_cfg.terminations)):
        if term_name != "time_out":
            setattr(env_cfg.terminations, term_name, None)
    env_cfg.actions = type(env_cfg)().actions.ik
    env_cfg.scene.ee_frame.target_frames[0].offset.pos = [0.0, 0.0, 0.107]
    env_cfg.actions.arm_action.controller.ik_params = {"lambda_val": 0.01}
    env_cfg.scene.base_camera.width = 1600
    env_cfg.scene.base_camera.height = 1600
    env_cfg.scene.base_camera.spawn.clipping_range = (0.01, 10.0)
    camera_rotation = create_rotation_matrix_from_view(
        torch.tensor([CAMERA_EYE]), torch.tensor([CAMERA_LOOKAT]), up_axis="Z"
    )
    env_cfg.scene.base_camera.offset.pos = CAMERA_EYE
    env_cfg.scene.base_camera.offset.rot = tuple(quat_from_matrix(camera_rotation)[0].tolist())
    env_cfg.video_recorders = [
        VideoRecorderCfg(
            source="sensor:base_camera:rgb",
            output_dir="videos/push_franka_cable",
            output_filename_prefix="per_env_camera_partition_10s",
            video_length=300,
            fps=30,
            step_offset=5,
        )
    ]

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(TASK_NAME, cfg=env_cfg)
        env.reset()

        actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
        actions[:, 3] = 1.0
        segment_index = env_cfg.commands.cable_pose.segment_index
        cable_position = (
            env.unwrapped.scene["cable"].data.segment_pose_w.torch[:, segment_index, :3]
            - env.unwrapped.scene.env_origins
        )
        grasp_position = cable_position.clone()
        grasp_position[:, 2] += GRASP_OFFSET_Z
        lift_position = grasp_position.clone()
        lift_position[:, 2] = LIFT_HEIGHT
        drop_position = lift_position.clone()
        drop_position[:, 0] = 0.0
        drop_position[:, 1] = DROP_TARGET_Y
        top_down_orientation = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
        top_down_orientation[:, 0] = 1.0
        pick_and_drop_sm = PickAndDropCableSm(
            env_cfg.sim.dt * env_cfg.decimation, env.unwrapped.num_envs, env.unwrapped.device
        )

        for _ in range(args_cli.num_steps):
            with torch.inference_mode():
                _, _, terminated, time_outs, _ = env.step(actions)
                dones = terminated | time_outs
                if dones.any():
                    reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                    pick_and_drop_sm.reset_idx(reset_ids)
                    cable_position = (
                        env.unwrapped.scene["cable"].data.segment_pose_w.torch[reset_ids, segment_index, :3]
                        - env.unwrapped.scene.env_origins[reset_ids]
                    )
                    grasp_position[reset_ids] = cable_position
                    grasp_position[reset_ids, 2] += GRASP_OFFSET_Z
                    lift_position[reset_ids] = grasp_position[reset_ids]
                    lift_position[reset_ids, 2] = LIFT_HEIGHT
                    drop_position[reset_ids] = lift_position[reset_ids]
                    drop_position[reset_ids, 0] = 0.0
                    drop_position[reset_ids, 1] = DROP_TARGET_Y

                ee_frame = env.unwrapped.scene["ee_frame"]
                ee_position = ee_frame.data.target_pos_w.torch[..., 0, :] - env.unwrapped.scene.env_origins
                ee_orientation = ee_frame.data.target_quat_w.torch[..., 0, :]

                actions = pick_and_drop_sm.compute(
                    torch.cat([ee_position, ee_orientation], dim=-1),
                    torch.cat([grasp_position, top_down_orientation], dim=-1),
                    torch.cat([lift_position, top_down_orientation], dim=-1),
                    torch.cat([drop_position, top_down_orientation], dim=-1),
                )

        env.close()


if __name__ == "__main__":
    main()
