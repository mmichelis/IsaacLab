# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render a still of the G1 on its soft pads using Newton's native (headless) viewer.

Unlike the RTX camera, Newton's ViewerGL renders the FEM deformable pads directly.
"""

import argparse
import sys

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, fold_preset_tokens, launch_simulation, resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-G1-SoftPads-Play-v0")
parser.add_argument("--settle", type=int, default=10)
parser.add_argument("--out", type=str, default="/tmp/g1_soft_pads_newton.png")
parser.add_argument("--eye", type=float, nargs=3, default=[1.3, -1.1, 0.5])
parser.add_argument("--target", type=float, nargs=3, default=[0.0, 0.0, 0.18])
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + fold_preset_tokens(hydra_args)


def main():
    env_cfg, _ = resolve_task_config(args_cli.task, "")
    env_cfg.scene.num_envs = 1
    env_cfg.commands.base_velocity.debug_vis = False

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg)
        u = env.unwrapped
        env.reset()
        actions = torch.zeros(env.action_space.shape, device=u.device)
        for _ in range(args_cli.settle):
            with torch.inference_mode():
                env.step(actions)

        # Import the GL viewer only after the app has launched (importing pyglet/newton.viewer
        # before SimulationApp corrupts Kit startup).
        from newton.viewer import ViewerGL
        from pyglet.math import Vec3 as PygletVec3

        from isaaclab_newton.physics import NewtonManager

        model = NewtonManager.get_model()
        state = NewtonManager.get_state()

        viewer = ViewerGL(width=1440, height=900, headless=True)
        viewer.set_model(model)
        viewer.set_world_offsets((0.0, 0.0, 0.0))
        viewer.up_axis = 2
        viewer.renderer.draw_sky = True
        viewer.renderer.draw_shadows = True
        viewer.camera.pos = PygletVec3(*args_cli.eye)
        viewer.camera.look_at(tuple(args_cli.target))

        # Render a few frames so the framebuffer + camera settle.
        for _ in range(4):
            viewer.begin_frame(0.0)
            viewer.log_state(state)
            viewer.end_frame()

        frame = viewer.get_frame()
        arr = frame.numpy() if hasattr(frame, "numpy") else np.asarray(frame)
        print(f"[newton-render] raw frame shape={arr.shape} dtype={arr.dtype} "
              f"min={arr.min():.3f} max={arr.max():.3f}")

        # Normalize to HxWx3 uint8.
        if arr.dtype != np.uint8:
            arr = np.clip(arr * (255.0 if arr.max() <= 1.0 + 1e-3 else 1.0), 0, 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[..., :3]
        arr = np.ascontiguousarray(arr)

        try:
            import imageio.v3 as iio

            iio.imwrite(args_cli.out, arr)
        except Exception:
            from PIL import Image

            Image.fromarray(arr).save(args_cli.out)
        print(f"[newton-render] saved {arr.shape} -> {args_cli.out}")
        env.close()


if __name__ == "__main__":
    main()
