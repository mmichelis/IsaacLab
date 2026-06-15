# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure the settled height of the G1 ankle_roll_link origin (stock flat env)."""

import argparse
import sys

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, fold_preset_tokens, launch_simulation, resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-G1-v0")
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + fold_preset_tokens(hydra_args)


def main():
    env_cfg, _ = resolve_task_config(args_cli.task, "")
    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env = gym.make(args_cli.task, cfg=env_cfg)
        u = env.unwrapped
        env.reset()
        robot = u.scene["robot"]
        fid = robot.find_bodies("left_ankle_roll_link", preserve_order=True)[0][0]
        actions = torch.zeros(env.action_space.shape, device=u.device)
        for _ in range(60):
            with torch.inference_mode():
                env.step(actions)
        bz = robot.data.root_pos_w.torch[:, 2].mean().item()
        fz = robot.data.body_link_pos_w.torch[:, fid, 2].mean().item()
        all_z = robot.data.body_link_pos_w.torch[:, :, 2]
        print(f"[measure] settled base_z={bz:.4f}  ankle_roll_link_z={fz:.4f}  min_body_z={all_z.min().item():.4f}")
        env.close()


if __name__ == "__main__":
    main()
