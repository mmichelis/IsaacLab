# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke test for the G1 soft-pad locomotion env.

Steps the env with zero (default-offset) actions and reports whether the robot
is supported on the pads, whether the pads track the feet, and whether the
simulation stays finite.
"""

import argparse
import sys

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import (
    add_launcher_args,
    fold_preset_tokens,
    launch_simulation,
    resolve_task_config,
    setup_preset_cli,
)

parser = argparse.ArgumentParser(description="G1 soft-pad smoke test.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-G1-SoftPads-Play-v0")
parser.add_argument("--steps", type=int, default=120)
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + fold_preset_tokens(hydra_args)


def main():
    torch.manual_seed(0)
    env_cfg, _ = resolve_task_config(args_cli.task, "")

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device

        env = gym.make(args_cli.task, cfg=env_cfg)
        u = env.unwrapped
        print(f"[smoke] obs space: {env.observation_space}")
        print(f"[smoke] act space: {env.action_space}")

        env.reset()
        robot = u.scene["robot"]
        pads = [u.scene["left_foot_pad"], u.scene["right_foot_pad"]]
        foot_ids = [robot.find_bodies(n, preserve_order=True)[0][0] for n in
                    ("left_ankle_roll_link", "right_ankle_roll_link")]

        actions = torch.zeros(env.action_space.shape, device=u.device)
        worst_base_z = 1e9
        for i in range(args_cli.steps):
            with torch.inference_mode():
                env.step(actions)

            base_z = robot.data.root_pos_w.torch[:, 2]
            foot_z = robot.data.body_link_pos_w.torch[:, foot_ids[0], 2]
            # pad-follow error: distance from each pad centroid to its foot.
            errs = []
            pad_bottoms = []
            pad_tops = []
            for pad, fid in zip(pads, foot_ids):
                npos = pad.data.nodal_pos_w.torch  # (E,P,3)
                if not torch.isfinite(npos).all():
                    print(f"[smoke] step {i}: NON-FINITE pad particles!")
                    env.close()
                    return
                centroid = npos.mean(dim=1)
                fp = robot.data.body_link_pos_w.torch[:, fid, :]
                errs.append((centroid[:, :2] - fp[:, :2]).norm(dim=-1).mean().item())
                pad_bottoms.append(npos[..., 2].amin(dim=1).mean().item())
                pad_tops.append(npos[..., 2].amax(dim=1).mean().item())
            worst_base_z = min(worst_base_z, base_z.mean().item())

            if i % 20 == 0 or i == args_cli.steps - 1:
                # foot sole estimate = foot origin - 0.061; pad_top should be >= sole (contact),
                # pad_bottom should be ~0 (resting on ground, not clipping below).
                sole = foot_z.mean().item() - 0.061
                print(
                    f"[smoke] step {i:3d} | base_z {base_z.mean().item():.3f} "
                    f"foot_z {foot_z.mean().item():.3f} sole~{sole:.3f} | "
                    f"pad_top L {pad_tops[0]:.3f} R {pad_tops[1]:.3f} | "
                    f"pad_bottom L {pad_bottoms[0]:.3f} R {pad_bottoms[1]:.3f} | "
                    f"xy_err L {errs[0]*100:.1f} R {errs[1]*100:.1f}cm"
                )

        print(f"[smoke] DONE. min mean base_z over run: {worst_base_z:.3f} "
              f"(spawn ~{0.74 + 0.06:.2f}; collapse if << 0.5)")
        env.close()


if __name__ == "__main__":
    main()
