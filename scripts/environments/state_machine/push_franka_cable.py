# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Close the Franka gripper and shove a cable off the table with differential IK.

.. code-block:: bash

    uv run --extra isaacsim python scripts/environments/state_machine/push_franka_cable.py --viz kit
"""

import argparse
import sys
from collections.abc import Sequence

import gymnasium as gym
import torch
import warp as wp

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.visualizers import VisualizerCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

TASK_NAME = "Isaac-Lift-Cable-Franka"
TABLE_HEIGHT = 0.0
TCP_CLEARANCE = 0.002
START_OFFSET_Y = -0.08
SHOVE_TARGET_Y = 0.55

parser = argparse.ArgumentParser(description="Shove a cable off a table with a Franka robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--num_steps", type=int, default=1000, help="Number of environment steps to run.")
add_launcher_args(parser)
parser.set_defaults(visualizer=["newton"])
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

wp.init()


class GripperState:
    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class ShoveSmState:
    REST = wp.constant(0)
    APPROACH_ABOVE_START = wp.constant(1)
    APPROACH_START = wp.constant(2)
    CLOSE_GRIPPER = wp.constant(3)
    SHOVE_CABLE = wp.constant(4)
    HOLD = wp.constant(5)


class ShoveSmWaitTime:
    REST = wp.constant(0.2)
    APPROACH_ABOVE_START = wp.constant(1.0)
    APPROACH_START = wp.constant(1.0)
    CLOSE_GRIPPER = wp.constant(0.5)
    SHOVE_CABLE = wp.constant(2.0)


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
    start_pose: wp.array(dtype=wp.transform),
    shove_pose: wp.array(dtype=wp.transform),
    desired_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    approach_offset: wp.array(dtype=wp.transform),
    position_threshold: float,
):
    tid = wp.tid()
    state = sm_state[tid]

    if state == ShoveSmState.REST:
        desired_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if sm_wait_time[tid] >= ShoveSmWaitTime.REST:
            sm_state[tid] = ShoveSmState.APPROACH_ABOVE_START
            sm_wait_time[tid] = 0.0
    elif state == ShoveSmState.APPROACH_ABOVE_START:
        desired_ee_pose[tid] = wp.transform_multiply(approach_offset[tid], start_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        if position_reached(ee_pose[tid], desired_ee_pose[tid], position_threshold):
            if sm_wait_time[tid] >= ShoveSmWaitTime.APPROACH_ABOVE_START:
                sm_state[tid] = ShoveSmState.APPROACH_START
                sm_wait_time[tid] = 0.0
    elif state == ShoveSmState.APPROACH_START:
        desired_ee_pose[tid] = start_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if position_reached(ee_pose[tid], desired_ee_pose[tid], position_threshold):
            if sm_wait_time[tid] >= ShoveSmWaitTime.APPROACH_START:
                sm_state[tid] = ShoveSmState.CLOSE_GRIPPER
                sm_wait_time[tid] = 0.0
    elif state == ShoveSmState.CLOSE_GRIPPER:
        desired_ee_pose[tid] = start_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if sm_wait_time[tid] >= ShoveSmWaitTime.CLOSE_GRIPPER:
            sm_state[tid] = ShoveSmState.SHOVE_CABLE
            sm_wait_time[tid] = 0.0
    elif state == ShoveSmState.SHOVE_CABLE:
        alpha = wp.min(sm_wait_time[tid] / ShoveSmWaitTime.SHOVE_CABLE, 1.0)
        start_position = wp.transform_get_translation(start_pose[tid])
        shove_position = wp.transform_get_translation(shove_pose[tid])
        shove_orientation = wp.transform_get_rotation(shove_pose[tid])
        desired_ee_pose[tid] = wp.transform(wp.lerp(start_position, shove_position, alpha), shove_orientation)
        gripper_state[tid] = GripperState.CLOSE
        if sm_wait_time[tid] >= ShoveSmWaitTime.SHOVE_CABLE:
            if position_reached(ee_pose[tid], desired_ee_pose[tid], position_threshold):
                sm_state[tid] = ShoveSmState.HOLD
                sm_wait_time[tid] = 0.0
    elif state == ShoveSmState.HOLD:
        desired_ee_pose[tid] = shove_pose[tid]
        gripper_state[tid] = GripperState.CLOSE

    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class ShoveCableSm:
    """Task-space state machine for a horizontal cable shove."""

    def __init__(self, dt: float, num_envs: int, device: torch.device | str, position_threshold: float = 0.005):
        self.num_envs = num_envs
        self.device = device
        self.position_threshold = position_threshold
        self.sm_dt = torch.full((num_envs,), dt, device=device)
        self.sm_state = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.sm_wait_time = torch.zeros(num_envs, device=device)
        self.desired_ee_pose = torch.zeros((num_envs, 7), device=device)
        self.gripper_state = torch.zeros(num_envs, device=device)
        self.approach_offset = torch.zeros((num_envs, 7), device=device)
        self.approach_offset[:, 2] = 0.15
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
        self.sm_state[env_ids] = ShoveSmState.REST
        self.sm_wait_time[env_ids] = 0.0

    def compute(self, ee_pose: torch.Tensor, start_pose: torch.Tensor, shove_pose: torch.Tensor) -> torch.Tensor:
        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                wp.from_torch(ee_pose.contiguous(), wp.transform),
                wp.from_torch(start_pose.contiguous(), wp.transform),
                wp.from_torch(shove_pose.contiguous(), wp.transform),
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
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.episode_length_s = 12.0
    for term_name in list(vars(env_cfg.terminations)):
        if term_name != "time_out":
            setattr(env_cfg.terminations, term_name, None)
    env_cfg.actions = type(env_cfg)().actions.ik
    env_cfg.scene.ee_frame.target_frames[0].offset.pos = [0.0, 0.0, 0.107]
    env_cfg.viewer.eye = (1.2, 0.8, 0.45)
    env_cfg.viewer.lookat = (0.5, 0.1, 0.0)
    env_cfg.sim.default_visualizer_cfg = VisualizerCfg(eye=env_cfg.viewer.eye, lookat=env_cfg.viewer.lookat)

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
        start_position = cable_position.clone()
        start_position[:, 1] += START_OFFSET_Y
        start_position[:, 2] = TABLE_HEIGHT + TCP_CLEARANCE
        shove_position = start_position.clone()
        shove_position[:, 1] = SHOVE_TARGET_Y
        top_down_orientation = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
        top_down_orientation[:, 0] = 1.0
        shove_sm = ShoveCableSm(env_cfg.sim.dt * env_cfg.decimation, env.unwrapped.num_envs, env.unwrapped.device)

        for _ in range(args_cli.num_steps):
            with torch.inference_mode():
                _, _, terminated, time_outs, _ = env.step(actions)
                dones = terminated | time_outs
                if dones.any():
                    reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                    shove_sm.reset_idx(reset_ids)
                    cable_position = (
                        env.unwrapped.scene["cable"].data.segment_pose_w.torch[reset_ids, segment_index, :3]
                        - env.unwrapped.scene.env_origins[reset_ids]
                    )
                    start_position[reset_ids] = cable_position
                    start_position[reset_ids, 1] += START_OFFSET_Y
                    start_position[reset_ids, 2] = TABLE_HEIGHT + TCP_CLEARANCE
                    shove_position[reset_ids] = start_position[reset_ids]
                    shove_position[reset_ids, 1] = SHOVE_TARGET_Y

                ee_frame = env.unwrapped.scene["ee_frame"]
                ee_position = ee_frame.data.target_pos_w.torch[..., 0, :] - env.unwrapped.scene.env_origins
                ee_orientation = ee_frame.data.target_quat_w.torch[..., 0, :]

                actions = shove_sm.compute(
                    torch.cat([ee_position, ee_orientation], dim=-1),
                    torch.cat([start_position, top_down_orientation], dim=-1),
                    torch.cat([shove_position, top_down_orientation], dim=-1),
                )

        env.close()


if __name__ == "__main__":
    main()
