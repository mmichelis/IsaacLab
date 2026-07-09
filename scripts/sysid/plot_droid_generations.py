# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plot the CMA-ES optimization evolution across generations. No simulation.

Reads the per-generation best-candidate artifacts (``generations/gen_*.pt``,
written by cma_es.py every generation) of the freshest fit run per joint and
renders two figures:

- ``<prefix>_gains.png``: per joint, best-of-generation stiffness, damping and
  score vs generation, with the rig's recorded gains as a reference line.
- ``<prefix>_trajectories.png``: per joint, the measured trajectory overlaid
  with best-of-generation rollouts, colored light (early) to dark (late).

Usage:
    python scripts/sysid/plot_droid_generations.py \\
        --data_root logs/sysid/prepared_datasets/franka_high/franka_fr3/sim \\
        --fitted_root logs/sysid/droid_franka_high \\
        --out_prefix sysid_plots/droid_generations
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from data_contract import load_dataset, validate_contract  # noqa: E402

# At most this many generation rollouts per trajectory subplot (first, last and
# evenly spaced in between) — more just overplots.
MAX_TRAJ_GENS = 8


def _runs_by_joint(fitted_root: Path) -> dict[str, Path]:
    """Freshest run dir per joint among runs that have per-generation artifacts."""
    latest: dict[str, Path] = {}
    for gen_dir in fitted_root.glob("*/generations"):
        gen_files = sorted(gen_dir.glob("gen_*.pt"))
        if not gen_files:
            continue
        joint = torch.load(gen_files[0], map_location="cpu", weights_only=False)["joint_order"][0]
        if joint not in latest or gen_dir.parent.stat().st_mtime > latest[joint].stat().st_mtime:
            latest[joint] = gen_dir.parent
    return latest


def _load_generations(run_dir: Path) -> list[dict]:
    return [
        torch.load(p, map_location="cpu", weights_only=False)
        for p in sorted((run_dir / "generations").glob("gen_*.pt"))
    ]


def _datasets_by_joint(data_root: Path) -> dict[str, Path]:
    mapping = {}
    for pt in data_root.glob("*/chirp_data_prepared.pt"):
        ds = validate_contract(load_dataset(str(pt)))
        mapping[ds.active_joint_names[0]] = pt
    return mapping


def _plot_gains(joints: list[str], gens_by_joint: dict, rig_gains: dict, out_path: Path) -> None:
    """Rows = joints; columns = stiffness / damping / score vs generation."""
    nrows = len(joints)
    fig, axes = plt.subplots(nrows, 3, figsize=(15, 2.2 * nrows), sharex=True, squeeze=False)
    for row, joint in enumerate(joints):
        gens = gens_by_joint[joint]
        it = [g["iteration"] for g in gens]
        kp = [float(g["sim_params"][0]) for g in gens]
        kd = [float(g["sim_params"][1]) for g in gens]
        sc = [g["score"] for g in gens]
        specs = [
            (kp, f"{joint} stiffness", rig_gains[joint][0]),
            (kd, f"{joint} damping", rig_gains[joint][1]),
            (sc, f"{joint} score [rad2]", None),
        ]
        for ax, (values, title, ref) in zip(axes[row], specs):
            ax.plot(it, values, color="tab:blue", linewidth=1.6, marker="o", markersize=3)
            if ref is not None:
                ax.axhline(ref, color="dimgray", linestyle="--", linewidth=1.0)
                ax.annotate(f"rig {ref:g}", (it[0], ref), color="dimgray", fontsize=8, va="bottom")
            ax.set_title(title, fontsize=10)
            ax.grid(True, linestyle=":", alpha=0.6)
            if title.endswith("[rad2]"):
                ax.set_yscale("log")
    for ax in axes[-1]:
        ax.set_xlabel("generation")
    fig.suptitle("CMA-ES evolution: best-of-generation gains and score per joint", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] wrote {out_path}")


def _plot_trajectories(joints: list[str], gens_by_joint: dict, data_by_joint: dict, out_path: Path) -> None:
    """Measured trajectory + best-of-generation rollouts, light (early) to dark (late)."""
    ncols = 2
    nrows = (len(joints) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.0 * nrows), sharex=True, squeeze=False)
    flat = axes.ravel()
    cmap = plt.get_cmap("Blues")
    for ax, joint in zip(flat, joints):
        ds = validate_contract(load_dataset(str(data_by_joint[joint])))
        col = ds.joint_names.index(joint)
        time = ds.time.numpy()
        meas = ds.dof_pos[:, col].numpy()

        gens = gens_by_joint[joint]
        if len(gens) > MAX_TRAJ_GENS:
            idx = np.unique(np.linspace(0, len(gens) - 1, MAX_TRAJ_GENS).astype(int))
            gens = [gens[i] for i in idx]
        for k, g in enumerate(gens):
            traj = g["trajectory"].reshape(len(time), -1)[:, 0].numpy()
            # Sequential ramp: generation is a magnitude, one hue light -> dark.
            color = cmap(0.3 + 0.7 * k / max(len(gens) - 1, 1))
            ax.plot(time, traj, color=color, linewidth=1.0, label=f"gen {g['iteration']}")
        ax.plot(time, meas, color="black", linewidth=1.4, label="measured")
        ax.set_title(f"{joint}  (gen {gens[0]['iteration']} -> {gens[-1]['iteration']})", fontsize=10)
        ax.set_ylabel("position [rad]")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=7, ncol=2, loc="upper right")
    for ax in flat[len(joints) :]:
        ax.set_visible(False)
    for ax in flat[max(0, len(joints) - ncols) : len(joints)]:
        ax.set_xlabel("time [s]")
    fig.suptitle("Best-of-generation rollouts vs measured (light = early, dark = late)", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", type=str, required=True, help="Dir containing per-joint run dirs.")
    parser.add_argument("--fitted_root", type=str, required=True, help="Fit log root with */generations/.")
    parser.add_argument("--out_prefix", type=str, required=True, help="Output prefix for the two PNGs.")
    args = parser.parse_args()

    runs = _runs_by_joint(Path(args.fitted_root))
    if not runs:
        raise SystemExit(f"no */generations/gen_*.pt under {args.fitted_root} — restart fits with the updated cma_es.py")
    data_by_joint = _datasets_by_joint(Path(args.data_root))
    joints = sorted(runs, key=lambda j: data_by_joint.get(j, Path(j)).parent.name)
    gens_by_joint = {j: _load_generations(runs[j]) for j in joints}
    rig_gains = {}
    for joint in joints:
        ds = validate_contract(load_dataset(str(data_by_joint[joint])))
        col = ds.joint_names.index(joint)
        rig_gains[joint] = (float(ds.kp_used[col]), float(ds.kd_used[col]))

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    _plot_gains(joints, gens_by_joint, rig_gains, out_prefix.parent / f"{out_prefix.name}_gains.png")
    _plot_trajectories(joints, gens_by_joint, data_by_joint, out_prefix.parent / f"{out_prefix.name}_trajectories.png")


if __name__ == "__main__":
    main()
