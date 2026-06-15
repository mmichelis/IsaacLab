# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone investigation of soft-pad <-> ground contact & friction.

Frees the pads from the foot-pinning, drops them from a height, and tracks their
height over time to see whether they collide with the ground (rest at z>=0) or
fall through it. ``--ground box`` adds a collidable box floor to compare against
the env's terrain plane. ``--vx`` gives the pads an initial sideways velocity to
probe friction (does the pad slide forever, or stop?).
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
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--drop_z", type=float, default=0.25)
parser.add_argument("--ground", choices=["plane", "box"], default="plane")
parser.add_argument("--vx", type=float, default=0.0, help="initial sideways pad velocity [m/s] (friction probe)")
parser.add_argument("--render", action="store_true")
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + fold_preset_tokens(hydra_args)


def main():
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg

    env_cfg, _ = resolve_task_config(args_cli.task, "")
    env_cfg.scene.num_envs = 1
    env_cfg.commands.base_velocity.debug_vis = False

    # Free the pads: plain joint action (no foot-pinning), so the pads are free deformables.
    env_cfg.actions.joint_pos = JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
    )
    # Drop the pads from a height, out from under the robot, onto open ground.
    env_cfg.scene.left_foot_pad.init_state.pos = (0.6, 0.15, args_cli.drop_z)
    env_cfg.scene.right_foot_pad.init_state.pos = (0.6, -0.15, args_cli.drop_z)

    if args_cli.ground == "box":
        env_cfg.scene.test_floor = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/TestFloor",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.6, 0.0, -0.05)),
            spawn=sim_utils.CuboidCfg(
                size=(1.0, 1.0, 0.1),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
            ),
        )

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg)
        u = env.unwrapped
        env.reset()
        left = u.scene["left_foot_pad"]
        right = u.scene["right_foot_pad"]
        actions = torch.zeros(env.action_space.shape, device=u.device)

        if args_cli.vx != 0.0:
            for pad in (left, right):
                vel = torch.zeros((1, pad.data.nodal_pos_w.torch.shape[1], 3), device=u.device)
                vel[..., 0] = args_cli.vx
                pad.write_nodal_velocity_to_sim_index(vel)

        print(f"[drop] ground={args_cli.ground} drop_z={args_cli.drop_z} vx={args_cli.vx}")
        for i in range(args_cli.steps):
            with torch.inference_mode():
                env.step(actions)
            lp = left.data.nodal_pos_w.torch[0]
            if not torch.isfinite(lp).all():
                print(f"[drop] step {i}: NON-FINITE")
                break
            if i % 10 == 0 or i == args_cli.steps - 1:
                lmin = lp[:, 2].amin().item()
                lcom = lp[:, 2].mean().item()
                lx = lp[:, 0].mean().item()
                print(f"[drop] step {i:3d} | pad_min_z {lmin:+.4f} pad_com_z {lcom:+.4f} pad_com_x {lx:+.4f}")

        if args_cli.render:
            from newton.viewer import ViewerGL
            from pyglet.math import Vec3 as PygletVec3

            from isaaclab_newton.physics import NewtonManager

            viewer = ViewerGL(width=1280, height=720, headless=True)
            viewer.set_model(NewtonManager.get_model())
            viewer.set_world_offsets((0.0, 0.0, 0.0))
            viewer.up_axis = 2
            viewer.camera.pos = PygletVec3(1.4, -0.9, 0.4)
            viewer.camera.look_at((0.6, 0.0, 0.05))
            for _ in range(4):
                viewer.begin_frame(0.0)
                viewer.log_state(NewtonManager.get_state())
                viewer.end_frame()
            arr = viewer.get_frame().numpy()
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            import imageio.v3 as iio

            iio.imwrite(f"/tmp/pad_drop_{args_cli.ground}.png", np.ascontiguousarray(arr[..., :3]))
            print(f"[drop] saved /tmp/pad_drop_{args_cli.ground}.png")
        env.close()


if __name__ == "__main__":
    main()
