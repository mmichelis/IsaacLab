# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Before/after overlay plot for droid sysid fits.

For every per-joint chirp run under --data_root, replays the recorded commands
through Isaac-Sysid-Droid-v0 twice in parallel (env0 = asset-default gains,
env1 = fitted gains where a fit artifact is given) and plots, per joint:
measured vs commanded vs sim-default vs sim-fitted position with RMSE in the
title. Mirrors the FR3 joint_fit_before_after plot layout.

Usage:
    python scripts/sysid/plot_droid_fit.py \\
        --data_root logs/sysid/prepared_datasets/franka_high/franka_fr3/sim \\
        --fitted fr3_joint1=logs/sysid/droid_franka_high/<stamp>/best_candidate.pt \\
        --out logs/sysid/droid_fit_before_after.png
"""

# flake8: noqa: E402

import argparse
import sys
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser(description="Droid sysid before/after overlay plot.")
parser.add_argument("--task", type=str, default="Isaac-Sysid-Droid-v0")
parser.add_argument("--data_root", type=str, required=True, help="Dir containing per-joint run dirs.")
parser.add_argument(
    "--fitted",
    nargs="*",
    default=[],
    help="Per-joint fit artifacts as <joint>=<best_candidate.pt>; joints without one plot default-only.",
)
parser.add_argument(
    "--fitted_root",
    type=str,
    default=None,
    help=(
        "Fit log root (e.g. logs/sysid/droid_franka_high): picks the freshest "
        "*/best_candidate.pt per joint. Works mid-fit — candidates update every "
        "generation. Ignored when --fitted is given."
    ),
)
parser.add_argument("--out", type=str, required=True, help="Output PNG path.")
parser.add_argument(
    "--default_gains",
    choices=["asset", "recorded"],
    default="asset",
    help=(
        "Gains for the baseline sim line: 'asset' = the articulation defaults "
        "(DROID_CFG 400/80), 'recorded' = the rig's kp_used/kd_used stamped in "
        "each dataset (sim configured like the real controller)."
    ),
)
parser.add_argument(
    "--mode",
    choices=["replay", "cached"],
    default="replay",
    help=(
        "replay: sim both gain sets and cache each run's default-gains trajectory next to the "
        "dataset. cached: no sim at all — default line from that cache (run replay once first), "
        "fitted line from the fit's own best_trajectory.pt. Instant, works mid-fit."
    ),
)
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401  (registers tasks)

sys.path.insert(0, str(Path(__file__).parent))
from data_contract import load_dataset, validate_contract  # noqa: E402

# Line styling matches the FR3 joint_fit_before_after reference figure. Identity
# is double-encoded (color + line style) so the overlay reads under CVD/print.
STYLE = {
    "measured": dict(color="tab:blue", linestyle="-", linewidth=1.6),
    "commanded": dict(color="tab:purple", linestyle="-.", linewidth=1.3),
    "default": dict(color="tab:green", linestyle=":", linewidth=1.6),
    "fitted": dict(color="tab:red", linestyle="--", linewidth=1.6),
}


def _cache_name() -> str:
    """Baseline rollout cache filename, one per (task, baseline-gains) pair."""
    task_slug = args_cli.task.lower().replace("isaac-sysid-", "").replace("-", "_")
    return f"{task_slug}_{args_cli.default_gains}_replay.pt"


def _resolve_fitted_specs() -> list[str]:
    """Explicit --fitted pairs, else the freshest per-joint artifact under --fitted_root."""
    if args_cli.fitted or not args_cli.fitted_root:
        return args_cli.fitted
    latest: dict[str, Path] = {}
    for artifact in Path(args_cli.fitted_root).glob("*/best_candidate.pt"):
        blob = torch.load(artifact, map_location="cpu", weights_only=False)
        if not isinstance(blob, dict) or len(blob.get("joint_order", [])) != 1:
            continue
        joint = blob["joint_order"][0]
        if joint not in latest or artifact.stat().st_mtime > latest[joint].stat().st_mtime:
            latest[joint] = artifact
    return [f"{joint}={path}" for joint, path in sorted(latest.items())]


def _load_fitted_map(specs: list[str], device) -> dict[str, torch.Tensor]:
    """Parse <joint>=<artifact.pt> pairs into {joint: (kp, kd) tensor}."""
    fitted: dict[str, torch.Tensor] = {}
    for spec in specs:
        joint, _, path = spec.partition("=")
        blob = torch.load(path, map_location=device, weights_only=False)
        params = blob["sim_params"] if isinstance(blob, dict) else blob
        params = params.reshape(-1)
        if params.shape[0] != 2:
            raise ValueError(f"{path}: expected a single-joint (2,) artifact, got {tuple(params.shape)}")
        stored = list(blob.get("joint_order", [joint])) if isinstance(blob, dict) else [joint]
        if stored != [joint]:
            raise ValueError(f"{path} was fitted on {stored}, not {joint}")
        fitted[joint] = params
        print(f"[INFO] fitted gains for {joint}: kp={params[0].item():.1f}, kd={params[1].item():.1f}")
    return fitted


def _collect_cached(run_dirs: list[Path], specs: list[str]) -> list[dict]:
    """Build plot rows without any sim: cached default replays + fit best_trajectory rollouts."""
    artifact_by_joint = {spec.partition("=")[0]: Path(spec.partition("=")[2]) for spec in specs}
    results = []
    for run_dir in run_dirs:
        ds = validate_contract(load_dataset(str(run_dir / "chirp_data_prepared.pt")))
        joint = ds.active_joint_names[0]
        col = ds.joint_names.index(joint)
        cache_path = run_dir / _cache_name()
        if not cache_path.exists():
            print(f"[WARN] skipping {run_dir.name}: no {_cache_name()} yet (run --mode replay to build it).")
            continue
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cache["joint"] != joint or cache["task"] != args_cli.task:
            raise SystemExit(f"{cache_path} is for {cache['joint']}/{cache['task']}, expected {joint}/{args_cli.task}")
        meas = ds.dof_pos[:, col].numpy()
        default_traj = np.asarray(cache["default_traj"])

        fitted_traj, fitted_gains = None, None
        artifact = artifact_by_joint.get(joint)
        if artifact is not None and (artifact.parent / "best_trajectory.pt").exists():
            blob = torch.load(artifact, map_location="cpu", weights_only=False)
            traj = torch.load(artifact.parent / "best_trajectory.pt", map_location="cpu", weights_only=False)
            fitted_traj = traj.reshape(len(meas), -1)[:, 0].numpy()
            params = blob["sim_params"].reshape(-1)
            fitted_gains = (float(params[0]), float(params[1]))

        rmse = [float(np.sqrt(np.mean((default_traj - meas) ** 2)) * 1e3)]
        if fitted_traj is not None:
            rmse.append(float(np.sqrt(np.mean((fitted_traj - meas) ** 2)) * 1e3))
        print(f"[INFO] {run_dir.name} {joint}: RMSE default={rmse[0]:.2f} mrad" +
              (f", fitted={rmse[1]:.2f} mrad" if len(rmse) > 1 else " (no fit artifact)"))
        results.append(
            dict(
                joint=joint,
                time=ds.time.numpy(),
                meas=meas,
                cmd=ds.des_dof_pos[:, col].numpy(),
                sim_default=default_traj,
                sim_fitted=fitted_traj,
                rmse=np.array(rmse),
                default_gains=tuple(cache["default_gains"]),
                fitted_gains=fitted_gains,
            )
        )
    return results


def main() -> None:
    run_dirs = sorted(p.parent for p in Path(args_cli.data_root).glob("*/chirp_data_prepared.pt"))
    if not run_dirs:
        raise SystemExit(f"no */chirp_data_prepared.pt under {args_cli.data_root}")

    if args_cli.mode == "cached":
        _plot(_collect_cached(run_dirs, _resolve_fitted_specs()), Path(args_cli.out))
        return

    env_cfg, _ = resolve_task_config(args_cli.task, "")
    env_cfg.scene.num_envs = 2  # env0 = default gains, env1 = fitted gains
    env_cfg.sim.dt = 0.001
    env_cfg.decimation = 5  # 200 Hz commands zero-order-held at 1 kHz physics
    env_cfg.sim.render_interval = env_cfg.decimation
    env_cfg.episode_length_s = 120.0

    name_map = dict(getattr(env_cfg.sysid, "sim_joint_name_map", None) or {})

    results = []  # one entry per run: dict with joint, time, meas, cmd, sim_default, sim_fitted
    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg)
        device = env.unwrapped.device
        art = env.unwrapped.scene["robot"]
        default_joint_pos = art.data.default_joint_pos.torch
        # Asset-default gains, captured BEFORE the first replay: each run
        # writes kp_used over every joint, so a live read-back after run one
        # returns the previous dataset's gains, not the asset's.
        asset_kp_full = art.data.joint_stiffness.torch[0].detach().clone()
        asset_kd_full = art.data.joint_damping.torch[0].detach().clone()
        fitted_map = _load_fitted_map(_resolve_fitted_specs(), device)

        for run_dir in run_dirs:
            ds = validate_contract(load_dataset(str(run_dir / "chirp_data_prepared.pt")))
            assert len(ds.active_joint_names) == 1, f"{run_dir}: expected a single active joint"
            joint = ds.active_joint_names[0]
            col = ds.joint_names.index(joint)
            # int32: the isaaclab_physx index-write kernels reject int64.
            sim_full_ids = torch.tensor(
                [art.joint_names.index(name_map.get(n, n)) for n in ds.joint_names],
                device=device,
                dtype=torch.int32,
            )
            active_id = sim_full_ids[col : col + 1]

            env.reset()
            # Both envs replay from the measured initial state with the dataset's
            # rig gains on the held joints; only the active joint's gains differ.
            kp = ds.kp_used.to(device).unsqueeze(0).expand(2, -1).contiguous()
            kd = ds.kd_used.to(device).unsqueeze(0).expand(2, -1).contiguous()
            # Held joints always run the rig's recorded gains; the baseline env's
            # ACTIVE joint runs either the asset default or the recorded gain.
            if args_cli.default_gains == "asset":
                kp[0, col] = asset_kp_full[active_id[0]]
                kd[0, col] = asset_kd_full[active_id[0]]
            default_kp = float(kp[0, col])
            default_kd = float(kd[0, col])
            has_fit = joint in fitted_map
            if has_fit:
                kp[1, col], kd[1, col] = fitted_map[joint][0], fitted_map[joint][1]
            art.write_joint_stiffness_to_sim_index(stiffness=kp, joint_ids=sim_full_ids)
            art.write_joint_damping_to_sim_index(damping=kd, joint_ids=sim_full_ids)
            art.write_joint_position_to_sim_index(
                position=ds.dof_pos.to(device)[0].unsqueeze(0).expand(2, -1), joint_ids=sim_full_ids
            )
            art.write_joint_velocity_to_sim_index(
                velocity=ds.dof_vel.to(device)[0].unsqueeze(0).expand(2, -1), joint_ids=sim_full_ids
            )

            des = ds.des_dof_pos.to(device)
            steps = des.shape[0]
            sim_traj = torch.zeros(steps, 2, device=device)
            actions = torch.zeros(2, art.num_joints, device=device)
            with torch.inference_mode():
                for i in range(steps):
                    sim_traj[i] = art.data.joint_pos.torch[:, active_id[0]]
                    actions[:, sim_full_ids] = des[i] - default_joint_pos[:, sim_full_ids]
                    env.step(actions)

            meas = ds.dof_pos[:, col].numpy()
            sim_np = sim_traj.cpu().numpy()
            rmse_mrad = np.sqrt(np.mean((sim_np - meas[:, None]) ** 2, axis=0)) * 1e3
            # Cache the baseline rollout: it only depends on the dataset and the
            # chosen baseline gains, so later --mode cached plots skip the sim.
            torch.save(
                {
                    "joint": joint,
                    "task": args_cli.task,
                    "default_gains": (default_kp, default_kd),
                    "default_traj": sim_np[:, 0],
                },
                run_dir / _cache_name(),
            )
            print(
                f"[INFO] {run_dir.name} {joint}: RMSE default={rmse_mrad[0]:.2f} mrad"
                + (f", fitted={rmse_mrad[1]:.2f} mrad" if has_fit else " (no fit artifact)")
            )
            results.append(
                dict(
                    joint=joint,
                    time=ds.time.numpy(),
                    meas=meas,
                    cmd=ds.des_dof_pos[:, col].numpy(),
                    sim_default=sim_np[:, 0],
                    sim_fitted=sim_np[:, 1] if has_fit else None,
                    rmse=rmse_mrad,
                    default_gains=(default_kp, default_kd),
                    fitted_gains=(
                        (float(fitted_map[joint][0]), float(fitted_map[joint][1])) if has_fit else None
                    ),
                )
            )
        env.close()
        # Plot before the sim context exits: SimulationApp teardown can end the
        # process, so nothing after the with block is guaranteed to run.
        _plot(results, Path(args_cli.out))


def _plot(results: list[dict], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num = len(results)
    ncols = 2
    nrows = (num + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.2 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    # Baseline legend label: exact gains when uniform across joints, a generic
    # tag otherwise (recorded rig gains differ per joint; titles carry them).
    baseline_gains = {r["default_gains"] for r in results}
    if len(baseline_gains) == 1:
        dg = results[0]["default_gains"]
        baseline_label = f"sim {dg[0]:.0f}/{dg[1]:.0f}"
    else:
        baseline_label = "sim (recorded rig gains)"
    for ax, r in zip(axes, results):
        ax.plot(r["time"], r["meas"], label="measured", **STYLE["measured"])
        ax.plot(r["time"], r["cmd"], label="commanded", **STYLE["commanded"])
        ax.plot(r["time"], r["sim_default"], label=baseline_label, **STYLE["default"])
        dg = r["default_gains"]
        title = f"{r['joint']}  RMSE sim({dg[0]:.0f}/{dg[1]:.1f})={r['rmse'][0]:.1f} mrad"
        if r["sim_fitted"] is not None:
            fg = r["fitted_gains"]
            ax.plot(r["time"], r["sim_fitted"], label="fitted", **STYLE["fitted"])
            title += f", fitted({fg[0]:.0f}/{fg[1]:.1f})={r['rmse'][1]:.2f} mrad"
        ax.set_title(title)
        ax.set_ylabel("position [rad]")
        ax.grid(True, linestyle=":", alpha=0.6)
        # The chirp is symmetric about the command mean: show only the upper
        # half so amplitude/phase differences are readable.
        center = float(np.mean(r["cmd"]))
        tops = [r["meas"].max(), r["cmd"].max(), r["sim_default"].max()]
        if r["sim_fitted"] is not None:
            tops.append(r["sim_fitted"].max())
        top = float(max(tops))
        ax.set_ylim(center, top + 0.12 * (top - center))
        # Second half of the sweep only: the high-frequency band is where
        # the controllers/plants actually differ.
        ax.set_xlim(r["time"][len(r["time"]) // 2], r["time"][-1])
    for ax in axes[num:]:
        ax.set_visible(False)
    for ax in axes[max(0, num - ncols) : num]:
        ax.set_xlabel("time [s]")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Droid sysid before/after: measured vs default gains vs fitted (sim replay)", y=1.0)
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 0.97), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"[INFO] wrote {out_path}")


if __name__ == "__main__":
    main()
