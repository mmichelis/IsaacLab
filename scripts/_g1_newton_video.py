# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render a Newton-viewer video of a JIT policy driving a G1 task (optionally following the robot).

Used for matching-style videos: rigid G1 running and G1-on-soft-pads, both via the Newton viewer.
"""

import argparse
import sys

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, fold_preset_tokens, launch_simulation, resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--jit", type=str, default="logs/rsl_rl/g1_flat/2026-06-12_15-21-01/exported/policy.pt")
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--out", type=str, default="/tmp/g1_newton_video.mp4")
parser.add_argument("--follow", action="store_true", help="camera tracks the robot base each frame")
parser.add_argument("--eye_off", type=float, nargs=3, default=[1.6, -1.4, 0.5])
parser.add_argument("--target_off", type=float, nargs=3, default=[0.0, 0.0, 0.35])
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + fold_preset_tokens(hydra_args)


def main():
    env_cfg, _ = resolve_task_config(args_cli.task, "")
    env_cfg.scene.num_envs = 1
    if hasattr(env_cfg.commands, "base_velocity"):
        env_cfg.commands.base_velocity.debug_vis = False

    with launch_simulation(env_cfg, args_cli):
        from newton.viewer import ViewerGL
        from pyglet.math import Vec3 as PygletVec3

        from isaaclab_newton.physics import NewtonManager

        env = gym.make(args_cli.task, cfg=env_cfg)
        u = env.unwrapped
        device = u.device
        policy = torch.jit.load(args_cli.jit, map_location=device).eval()
        robot = u.scene["robot"]

        viewer = ViewerGL(width=1280, height=720, headless=True)
        viewer.set_model(NewtonManager.get_model())
        viewer.set_world_offsets((0.0, 0.0, 0.0))
        viewer.up_axis = 2
        viewer.renderer.draw_sky = True
        viewer.renderer.draw_shadows = True
        eye_off = np.asarray(args_cli.eye_off)
        tgt_off = np.asarray(args_cli.target_off)

        def place_camera():
            base = robot.data.root_pos_w.torch[0].detach().cpu().numpy()
            viewer.camera.pos = PygletVec3(*(base + eye_off))
            viewer.camera.look_at(tuple(base + tgt_off))

        obs_dict, _ = env.reset()
        place_camera()
        frames = []

        def grab():
            if args_cli.follow:
                place_camera()
            viewer.begin_frame(0.0)
            viewer.log_state(NewtonManager.get_state())
            viewer.end_frame()
            arr = viewer.get_frame().numpy()
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return np.ascontiguousarray(arr[..., :3])

        grab()  # warm-up
        for _ in range(args_cli.steps):
            with torch.inference_mode():
                actions = policy(obs_dict["policy"])
                obs_dict, _, _, _, _ = env.step(actions)
            frames.append(grab())

        print(f"[video] captured {len(frames)} frames {frames[0].shape}")
        import imageio

        imageio.mimsave(args_cli.out, frames, fps=30, quality=8)
        print(f"[video] saved -> {args_cli.out}")
        env.close()


if __name__ == "__main__":
    main()
