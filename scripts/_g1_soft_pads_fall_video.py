# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record a video of the G1 on soft pads (driven by the rigid-trained policy) falling over.

Frames are captured with Newton's native headless viewer so the deformable pads are visible.
"""

import argparse
import sys

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import (
    add_launcher_args,
    fold_preset_tokens,
    launch_simulation,
    load_cfg_from_registry,
    resolve_task_config,
    setup_preset_cli,
)

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-G1-SoftPads-Play-v0")
parser.add_argument("--jit", type=str, default="logs/rsl_rl/g1_flat/2026-06-12_15-21-01/exported/policy.pt")
parser.add_argument("--steps", type=int, default=150)
parser.add_argument("--out", type=str, default="/tmp/g1_soft_pads_falling.mp4")
parser.add_argument("--eye", type=float, nargs=3, default=[1.9, -1.8, 0.95])
parser.add_argument("--target", type=float, nargs=3, default=[0.0, 0.0, 0.4])
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + fold_preset_tokens(hydra_args)


def main():
    env_cfg, _ = resolve_task_config(args_cli.task, "")
    env_cfg.scene.num_envs = 1
    env_cfg.commands.base_velocity.debug_vis = False

    with launch_simulation(env_cfg, args_cli):
        from newton.viewer import ViewerGL
        from pyglet.math import Vec3 as PygletVec3

        from isaaclab_newton.physics import NewtonManager

        env = gym.make(args_cli.task, cfg=env_cfg)
        device = env.unwrapped.device
        policy = torch.jit.load(args_cli.jit, map_location=device).eval()

        model = NewtonManager.get_model()
        viewer = ViewerGL(width=1280, height=720, headless=True)
        viewer.set_model(model)
        viewer.set_world_offsets((0.0, 0.0, 0.0))
        viewer.up_axis = 2
        viewer.renderer.draw_sky = True
        viewer.renderer.draw_shadows = True
        viewer.camera.pos = PygletVec3(*args_cli.eye)
        viewer.camera.look_at(tuple(args_cli.target))

        obs_dict, _ = env.reset()
        frames = []

        def grab():
            state = NewtonManager.get_state()
            viewer.begin_frame(0.0)
            viewer.log_state(state)
            viewer.end_frame()
            arr = viewer.get_frame().numpy()
            if arr.dtype != np.uint8:
                arr = np.clip(arr * (255.0 if arr.max() <= 1.0 + 1e-3 else 1.0), 0, 255).astype(np.uint8)
            return np.ascontiguousarray(arr[..., :3])

        grab()  # warm-up frame (first GL frame can be blank)
        for _ in range(args_cli.steps):
            with torch.inference_mode():
                actions = policy(obs_dict["policy"])
                obs_dict, _, _, _, _ = env.step(actions)
            frames.append(grab())

        print(f"[fall-video] captured {len(frames)} frames {frames[0].shape}")
        try:
            import imageio

            imageio.mimsave(args_cli.out, frames, fps=30, quality=8)
        except Exception as exc:
            print(f"[fall-video] mp4 failed ({exc}); writing GIF")
            args_cli.out = args_cli.out.rsplit(".", 1)[0] + ".gif"
            import imageio

            imageio.mimsave(args_cli.out, frames[::2], fps=15)
        print(f"[fall-video] saved -> {args_cli.out}")
        env.close()


if __name__ == "__main__":
    main()
