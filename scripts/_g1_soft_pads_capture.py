# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render a still image of the G1 standing on its soft foot pads."""

import argparse
import sys

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, fold_preset_tokens, launch_simulation, resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-G1-SoftPads-Play-v0")
parser.add_argument("--settle", type=int, default=40)
parser.add_argument("--out", type=str, default="/tmp/g1_soft_pads.png")
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + fold_preset_tokens(hydra_args)


def _look_at_quat_world(eye, target, world_up=(0.0, 0.0, 1.0)):
    """Quaternion (x,y,z,w) for the 'world' camera convention (forward +X, up +Z)."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    f = target - eye
    f /= np.linalg.norm(f)
    up = np.asarray(world_up, dtype=np.float64)
    up = up - np.dot(up, f) * f
    up /= np.linalg.norm(up)
    y = np.cross(up, f)  # local +Y
    R = np.column_stack([f, y, up])  # columns: world coords of local +X,+Y,+Z
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y_ = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y_ = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y_ = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y_ = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    return (float(x), float(y_), float(z), float(w))


def main():
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    env_cfg, _ = resolve_task_config(args_cli.task, "")
    env_cfg.scene.num_envs = 1

    # Dynamic deformable meshes don't stream to the RTX camera (fabric/RTX limitation), so show
    # the pads via their particle markers instead. Turn off the velocity-command arrow.
    env_cfg.commands.base_velocity.debug_vis = False
    for _name in ("left_foot_pad", "right_foot_pad"):
        getattr(env_cfg.scene, _name).debug_vis = True

    # Fixed look-at vantage (env-0 origin is at the world origin): full robot, biased to the feet.
    eye = (1.3, -1.1, 0.5)
    target = (0.0, 0.0, 0.18)
    cam_quat = _look_at_quat_world(eye, target)

    env_cfg.scene.capture_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/CaptureCam",
        update_period=0.0,
        height=900,
        width=1440,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 100.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=eye, rot=cam_quat, convention="world"),
    )

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg)
        u = env.unwrapped
        env.reset()

        actions = torch.zeros(env.action_space.shape, device=u.device)
        for _ in range(args_cli.settle):
            with torch.inference_mode():
                env.step(actions)

        cam = u.scene["capture_cam"]
        origin = u.scene.env_origins[0].detach().cpu().numpy()
        # Low front-side vantage so the soft pads under the feet are visible.
        eye = torch.tensor([[origin[0] + 1.3, origin[1] - 1.1, origin[2] + 0.45]], device=u.device)
        target = torch.tensor([[origin[0] + 0.0, origin[1] + 0.0, origin[2] + 0.20]], device=u.device)
        cam.set_world_poses_from_view(eye, target)

        # A few more steps so the renderer updates to the new camera pose.
        for _ in range(6):
            with torch.inference_mode():
                env.step(actions)

        rgb = cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
        try:
            import imageio.v3 as iio

            iio.imwrite(args_cli.out, rgb)
        except Exception:
            from PIL import Image

            Image.fromarray(rgb).save(args_cli.out)
        print(f"[capture] saved image {rgb.shape} -> {args_cli.out}")
        env.close()


if __name__ == "__main__":
    main()
