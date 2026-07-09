# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Three-way overlay: real robot vs two sim plants, same PD gains. No simulation.

Reads the baseline-rollout caches that plot_droid_fit.py --mode replay leaves
next to each dataset and overlays, per joint: measured (real robot), commanded,
and the two cached sim rollouts (e.g. as-is vs corrected asset), all driven by
the same recorded controller gains. RMSEs in the titles.

Usage:
    python scripts/sysid/plot_asset_compare.py \\
        --data_root logs/sysid/prepared_datasets/franka_high/franka_fr3/sim \\
        --cache_a droid_v0_recorded_replay.pt --label_a "as-is asset" \\
        --cache_b droid_corrected_v0_recorded_replay.pt --label_b "corrected asset" \\
        --out sysid_plots/droid_asset_compare.png
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

STYLE = {
    "measured": dict(color="tab:blue", linestyle="-", linewidth=1.6),
    "commanded": dict(color="tab:purple", linestyle="-.", linewidth=1.3),
    "a": dict(color="tab:green", linestyle=":", linewidth=1.6),
    "b": dict(color="tab:red", linestyle="--", linewidth=1.6),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--cache_a", type=str, required=True, help="Cache filename of the first sim plant.")
    parser.add_argument("--cache_b", type=str, required=True, help="Cache filename of the second sim plant.")
    parser.add_argument("--label_a", type=str, default="sim A")
    parser.add_argument("--label_b", type=str, default="sim B")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    rows = []
    for pt in sorted(Path(args.data_root).glob("*/chirp_data_prepared.pt")):
        run_dir = pt.parent
        caches = {}
        for key, name in (("a", args.cache_a), ("b", args.cache_b)):
            path = run_dir / name
            if path.exists():
                caches[key] = torch.load(path, map_location="cpu", weights_only=False)
        if len(caches) < 2:
            print(f"[WARN] skipping {run_dir.name}: missing {[n for k, n in (('a', args.cache_a), ('b', args.cache_b)) if k not in caches]}")
            continue
        ds = validate_contract(load_dataset(str(pt)))
        joint = ds.active_joint_names[0]
        col = ds.joint_names.index(joint)
        gains_a, gains_b = tuple(caches["a"]["default_gains"]), tuple(caches["b"]["default_gains"])
        if caches["a"]["joint"] != joint or caches["b"]["joint"] != joint or gains_a != gains_b:
            raise SystemExit(f"{run_dir.name}: cache mismatch (joints/gains differ) — regenerate the caches.")
        meas = ds.dof_pos[:, col].numpy()
        traj_a = np.asarray(caches["a"]["default_traj"])
        traj_b = np.asarray(caches["b"]["default_traj"])
        rows.append(
            dict(
                joint=joint,
                gains=gains_a,
                time=ds.time.numpy(),
                meas=meas,
                cmd=ds.des_dof_pos[:, col].numpy(),
                traj_a=traj_a,
                traj_b=traj_b,
                rmse_a=float(np.sqrt(np.mean((traj_a - meas) ** 2)) * 1e3),
                rmse_b=float(np.sqrt(np.mean((traj_b - meas) ** 2)) * 1e3),
            )
        )
        print(f"[INFO] {joint}: {args.label_a}={rows[-1]['rmse_a']:.2f} mrad, {args.label_b}={rows[-1]['rmse_b']:.2f} mrad")
    if not rows:
        raise SystemExit("no runs with both caches")

    ncols = 2
    nrows = (len(rows) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.2 * nrows), sharex=True, squeeze=False)
    flat = axes.ravel()
    for ax, r in zip(flat, rows):
        ax.plot(r["time"], r["meas"], label="real robot (measured)", **STYLE["measured"])
        ax.plot(r["time"], r["cmd"], label="commanded", **STYLE["commanded"])
        ax.plot(r["time"], r["traj_a"], label=args.label_a, **STYLE["a"])
        ax.plot(r["time"], r["traj_b"], label=args.label_b, **STYLE["b"])
        kp, kd = r["gains"]
        ax.set_title(
            f"{r['joint']} @ kp={kp:g}/kd={kd:g}  RMSE {args.label_a}={r['rmse_a']:.1f}, "
            f"{args.label_b}={r['rmse_b']:.2f} mrad",
            fontsize=10,
        )
        ax.set_ylabel("position [rad]")
        ax.grid(True, linestyle=":", alpha=0.6)
        # Symmetric chirp: zoom to the upper half for readability.
        center = float(np.mean(r["cmd"]))
        top = float(max(r["meas"].max(), r["cmd"].max(), r["traj_a"].max(), r["traj_b"].max()))
        ax.set_ylim(center, top + 0.12 * (top - center))
        # Second half of the sweep: the high-frequency band is where plants differ.
        ax.set_xlim(r["time"][len(r["time"]) // 2], r["time"][-1])
    for ax in flat[len(rows) :]:
        ax.set_visible(False)
    for ax in flat[max(0, len(rows) - ncols) : len(rows)]:
        ax.set_xlabel("time [s]")
    handles, labels = flat[0].get_legend_handles_labels()
    fig.suptitle("Real robot vs sim plants, identical recorded PD gains", y=1.0)
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.97), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"[INFO] wrote {out}")


if __name__ == "__main__":
    main()
