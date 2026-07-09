# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime smoke test for the FR3 sysid env (validation gate #1).

Creates Isaac-Sysid-Franka-FR3-v0 on the default (newton_mjwarp) backend,
writes DIFFERENT per-env implicit gains, drives a constant position offset and
asserts (a) the env steps, (b) a stiff env tracks better than a soft env — i.e.
per-env gain writes actually reach the solver, (c) identical-gain envs produce
identical trajectories.

Run:
    python scripts/sysid/smoke_test.py --num_envs 4 --visualizer none
"""

# flake8: noqa: E402

import argparse
import sys

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser(description="FR3 sysid env smoke test.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--task", type=str, default="Isaac-Sysid-Franka-FR3-v0")
parser.add_argument("--steps", type=int, default=200, help="env steps to run (200 Hz).")
parser.add_argument(
    "--no_cuda_graph",
    action="store_true",
    default=False,
    help="Disable CUDA graph capture (gain-write regression must pass in both modes).",
)
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401


def main() -> None:
    env_cfg, _ = resolve_task_config(args_cli.task, "")
    env_cfg.scene.num_envs = args_cli.num_envs
    # resolve_task_config may hand back the PresetCfg or the already-resolved
    # NewtonCfg — target whichever actually carries use_cuda_graph.
    phys = getattr(env_cfg.sim.physics, "newton_mjwarp", env_cfg.sim.physics)
    if args_cli.no_cuda_graph:
        if not hasattr(phys, "use_cuda_graph"):
            raise RuntimeError(f"--no_cuda_graph: resolved physics cfg {type(phys).__name__} has no use_cuda_graph")
        phys.use_cuda_graph = False
    effective_graph = getattr(phys, "use_cuda_graph", None)
    print(f"[SMOKE] physics={type(phys).__name__}, use_cuda_graph={effective_graph}")
    # PhysX presets carry no use_cuda_graph; the flag check only applies to Newton.
    if effective_graph is not None:
        assert effective_graph == (not args_cli.no_cuda_graph), "use_cuda_graph flag did not take effect"

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg)
        env.reset()
        art = env.unwrapped.scene["robot"]
        device = env.unwrapped.device
        n = art.num_joints
        print(f"[SMOKE] backend physics: {type(env_cfg.sim.physics).__name__}")
        print(f"[SMOKE] joints ({n}): {art.joint_names}")
        # The fitted joints (task's sysid joint_order, mapped to articulation
        # names) must all exist; extra articulation joints (e.g. a gripper) are
        # allowed and held at their default target.
        name_map = dict(getattr(env_cfg.sysid, "sim_joint_name_map", None) or {})
        fit_joints = [name_map.get(j, j) for j in env_cfg.sysid.joint_order]
        missing = [j for j in fit_joints if j not in art.joint_names]
        assert not missing, f"fitted joints missing from articulation: {missing}"
        assert env.unwrapped.action_manager.total_action_dim == n
        # 4th fitted joint: ready pose -2.356, well inside limits on both robots.
        jidx = art.joint_names.index(fit_joints[3])

        num_envs = env.unwrapped.num_envs
        assert num_envs >= 4, "smoke test needs at least 4 envs"
        # env0 soft, env1 stiff, env2/env3 identical medium gains.
        stiffness = torch.full((num_envs, n), 600.0, device=device)
        damping = torch.full((num_envs, n), 30.0, device=device)
        stiffness[0], damping[0] = 30.0, 2.0
        stiffness[1], damping[1] = 3000.0, 60.0
        art.write_joint_stiffness_to_sim_index(stiffness=stiffness)
        art.write_joint_damping_to_sim_index(damping=damping)

        default = art.data.default_joint_pos.torch.clone()
        target = default.clone()
        target[:, jidx] += 0.2
        actions = target - default

        with torch.inference_mode():
            for _ in range(args_cli.steps):
                env.step(actions)

        q = art.data.joint_pos.torch[:, jidx].detach().cpu()
        err = (target[:, jidx].cpu() - q).abs()
        print(f"[SMOKE] q_j4 per env after {args_cli.steps} steps: {q.tolist()}")
        print(f"[SMOKE] |target - q| per env: {err.tolist()}")
        assert err[1] < err[0], "stiff env must track better than soft env — per-env gain write ineffective?"
        assert abs(q[2] - q[3]) < 1e-6, "identical-gain envs must match exactly"
        assert err[1] < 0.05, f"stiff env should be near target, err={err[1]:.4f}"
        print("[SMOKE] PASS: env steps on the Newton/mjwarp backend and per-env gain writes take effect.")
        env.close()


if __name__ == "__main__":
    main()
