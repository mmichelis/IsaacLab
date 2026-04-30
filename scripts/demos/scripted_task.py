# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run the Softbody-Franka demo with scripted keyframe motion.

Usage::

    ./isaaclab.sh -p scripts/demos/scripted_task.py --num_envs 1 --visualizer newton presets=newton

"""

import argparse
import sys

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config

TASK = "Isaac-Softbody-Franka-Direct-v0"

parser = argparse.ArgumentParser(description="Scripted Softbody-Franka demo.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--task", type=str, default=TASK, help="Task name.")
parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    help="Maximum number of environment steps to run. Defaults to one episode without a visualizer.",
)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args


def main():
    torch.manual_seed(42)

    env_cfg, _ = resolve_task_config(args_cli.task, "")

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        env = gym.make(args_cli.task, cfg=env_cfg)

        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")
        print("[INFO]: The Gym action tensor is a placeholder; scripted IK targets are applied by the environment.")
        env.reset()

        sim = env.unwrapped.sim
        # DirectRLEnv requires an action tensor for every step. SoftbodyFrankaEnv ignores
        # this external action and advances its keyframe controller in _apply_action().
        dummy_actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        max_steps = args_cli.max_steps
        if not sim.visualizers and max_steps is None:
            max_steps = env.unwrapped.max_episode_length

        step_count = 0
        while True:
            if max_steps is not None and step_count >= max_steps:
                break
            if sim.visualizers:
                if not any(v.is_running() and not v.is_closed for v in sim.visualizers):
                    break
            with torch.inference_mode():
                env.step(dummy_actions)
            step_count += 1

        env.close()


if __name__ == "__main__":
    main()
