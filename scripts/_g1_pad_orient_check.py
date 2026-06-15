# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Diagnose soft-pad orientation vs the foot's frame."""

import argparse
import sys

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, fold_preset_tokens, launch_simulation, resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-G1-SoftPads-Play-v0")
parser.add_argument("--settle", type=int, default=15)
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + fold_preset_tokens(hydra_args)


def main():
    from isaaclab.utils.math import quat_apply

    env_cfg, _ = resolve_task_config(args_cli.task, "")
    env_cfg.scene.num_envs = 1
    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg)
        u = env.unwrapped
        env.reset()
        robot = u.scene["robot"]
        pad = u.scene["left_foot_pad"]
        fid = robot.find_bodies("left_ankle_roll_link", preserve_order=True)[0][0]
        actions = torch.zeros(env.action_space.shape, device=u.device)
        for _ in range(args_cli.settle):
            with torch.inference_mode():
                env.step(actions)

        root_q = robot.data.root_quat_w.torch[0]  # xyzw
        fq = robot.data.body_link_quat_w.torch[0, fid]  # xyzw
        fp = robot.data.body_link_pos_w.torch[0, fid]
        # robot + foot local axes expressed in world
        ex = torch.tensor([1.0, 0, 0], device=u.device)
        ey = torch.tensor([0, 1.0, 0], device=u.device)
        ez = torch.tensor([0, 0, 1.0], device=u.device)
        print(f"[orient] robot fwd (root x in world): {quat_apply(root_q, ex).tolist()}")
        print(f"[orient] foot local x in world: {quat_apply(fq, ex).tolist()}")
        print(f"[orient] foot local y in world: {quat_apply(fq, ey).tolist()}")
        print(f"[orient] foot local z in world: {quat_apply(fq, ez).tolist()}")

        p = pad.data.nodal_pos_w.torch[0]  # (P,3) world
        pc = p - p.mean(0)
        cov = pc.t() @ pc
        _, eigvecs = torch.linalg.eigh(cov.double())
        long_axis = eigvecs[:, -1].float()  # principal (longest) axis of the pad, world frame
        foot_fwd = quat_apply(fq, ex)
        align = torch.abs(torch.dot(long_axis, foot_fwd)).item()
        print(f"[orient] pad long axis (world): {long_axis.tolist()}")
        print(f"[orient] |pad_long . foot_fwd| = {align:.3f}  (1.0 = pad aligned with heel->toe)")
        env.close()


def _round(env_cfg):
    try:
        return tuple(round(float(s), 3) for s in env_cfg.scene.left_foot_pad.spawn.size)
    except Exception:
        return "?"


if __name__ == "__main__":
    main()
