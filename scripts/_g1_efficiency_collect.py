# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect joint-power / velocity data for a trained G1 policy across a forward-command sweep.

One launch sweeps several fixed forward commands (resetting between them) and logs per-step,
per-env joint applied torque, joint velocity, base velocity, base height/orientation, and the
episode-length buffer (to segment resets). Saved to an .npz for offline efficiency analysis.
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
parser.add_argument("--jit", type=str, required=True)
parser.add_argument("--commands", type=str, default="0.5,1.0,1.5,2.0,2.5,3.0")
parser.add_argument("--steps_per_cmd", type=int, default=600)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--log_pad", action="store_true", help="also log soft-pad kinetic energy / COM vel")
parser.add_argument("--out", type=str, required=True)
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + fold_preset_tokens(hydra_args)


def main():
    cmds = [float(x) for x in args_cli.commands.split(",")]
    env_cfg, _ = resolve_task_config(args_cli.task, "")
    env_cfg.scene.num_envs = args_cli.num_envs

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg)
        u = env.unwrapped
        dev = u.device
        policy = torch.jit.load(args_cli.jit, map_location=dev).eval()
        robot = u.scene["robot"]

        # Keep heading control ON (straight running, the trained regime); command is set per
        # block via the cfg ranges and held by resampling to the same fixed value.
        term = u.command_manager.get_term("base_velocity")

        mass = robot.data.body_mass.torch.sum(dim=1).detach().cpu().numpy()  # (E,) kg
        joint_names = list(robot.data.joint_names)
        nE = args_cli.num_envs
        nJ = len(joint_names)
        S = args_cli.steps_per_cmd
        nC = len(cmds)

        # Pre-allocate logs: (nC, S, nE, ...)
        log = {
            "tau": np.zeros((nC, S, nE, nJ), np.float32),
            "omega": np.zeros((nC, S, nE, nJ), np.float32),
            "vx_w": np.zeros((nC, S, nE), np.float32),
            "vx_b": np.zeros((nC, S, nE), np.float32),
            "vy_w": np.zeros((nC, S, nE), np.float32),
            "vz_w": np.zeros((nC, S, nE), np.float32),
            "x_w": np.zeros((nC, S, nE), np.float32),
            "z_w": np.zeros((nC, S, nE), np.float32),
            "projg_z": np.zeros((nC, S, nE), np.float32),  # projected gravity z (≈ -1 upright)
            "ep_len": np.zeros((nC, S, nE), np.int32),  # steps since last reset (reset detector)
        }

        # Optional soft-pad energy proxy: pad kinetic energy and COM vertical velocity.
        pads = []
        if args_cli.log_pad:
            for nm in ("left_foot_pad", "right_foot_pad"):
                if nm in u.scene.keys():
                    pads.append(u.scene[nm])
            if pads:
                # Uniform per-particle mass from configured density * volume / N.
                pcfg = pads[0].cfg.spawn
                vol = float(pcfg.size[0] * pcfg.size[1] * pcfg.size[2])
                npart = pads[0].data.nodal_pos_w.torch.shape[1]
                m_part = pcfg.physics_material.density * vol / npart
                log["pad_ke"] = np.zeros((nC, S, nE), np.float32)  # J, summed over both pads
                log["pad_com_vz"] = np.zeros((nC, S, nE), np.float32)  # m/s, mean of both pads

        for ci, v in enumerate(cmds):
            term.cfg.ranges.lin_vel_x = (v, v)
            term.cfg.ranges.lin_vel_y = (0.0, 0.0)
            term.cfg.ranges.ang_vel_z = (0.0, 0.0)
            term.cfg.ranges.heading = (0.0, 0.0)  # run straight along +x
            obs_dict, _ = env.reset()
            print(f"[collect] command vx={v:.2f} m/s ...", flush=True)
            for t in range(S):
                with torch.inference_mode():
                    action = policy(obs_dict["policy"])
                    obs_dict, _, _, _, _ = env.step(action)
                d = robot.data
                log["tau"][ci, t] = d.applied_torque.torch.detach().cpu().numpy()
                log["omega"][ci, t] = d.joint_vel.torch.detach().cpu().numpy()
                vw = d.root_lin_vel_w.torch
                log["vx_w"][ci, t] = vw[:, 0].detach().cpu().numpy()
                log["vy_w"][ci, t] = vw[:, 1].detach().cpu().numpy()
                log["vz_w"][ci, t] = vw[:, 2].detach().cpu().numpy()
                log["vx_b"][ci, t] = d.root_lin_vel_b.torch[:, 0].detach().cpu().numpy()
                pw = d.root_pos_w.torch
                log["x_w"][ci, t] = pw[:, 0].detach().cpu().numpy()
                log["z_w"][ci, t] = pw[:, 2].detach().cpu().numpy()
                log["projg_z"][ci, t] = d.projected_gravity_b.torch[:, 2].detach().cpu().numpy()
                log["ep_len"][ci, t] = u.episode_length_buf.detach().cpu().numpy()
                if pads:
                    ke = torch.zeros(nE, device=dev)
                    vz = torch.zeros(nE, device=dev)
                    for pad in pads:
                        nv = pad.data.nodal_vel_w.torch  # (E, P, 3)
                        ke = ke + 0.5 * m_part * (nv**2).sum(dim=(1, 2))
                        vz = vz + nv[..., 2].mean(dim=1)
                    log["pad_ke"][ci, t] = ke.detach().cpu().numpy()
                    log["pad_com_vz"][ci, t] = (vz / len(pads)).detach().cpu().numpy()

        np.savez_compressed(
            args_cli.out,
            commands=np.array(cmds, np.float32),
            mass=mass,
            joint_names=np.array(joint_names),
            dt=float(u.step_dt),
            task=args_cli.task,
            **log,
        )
        print(f"[collect] saved {args_cli.out}  (nC={nC}, S={S}, nE={nE}, nJ={nJ})", flush=True)
        env.close()


if __name__ == "__main__":
    main()
