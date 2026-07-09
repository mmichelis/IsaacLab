# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trajectory overlay for the controller-tracking investigation.

Per joint: the (identical) commanded chirp plus the REAL measured response of
each PD preset, and the sim-estimated response of IsaacLab's 400/80 (from the
corrected-plant replay cache). Commands are byte-identical across the preset
collections, so overlaying responses from different recordings is exact.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys

sys.path.insert(0, str(Path(__file__).parent))
from data_contract import load_dataset, validate_contract  # noqa: E402

ROOT = Path("logs/sysid/prepared_datasets")
PRESETS = [
    ("franka_ros", "franka_ros (600.../30...)", "tab:blue", "-"),
    ("franka_high", "franka_robot_high (400.../20...)", "tab:green", "-"),
    ("franka_low", "franka_robot_low = IsaacLab default (80/4)", "tab:orange", "-"),
]
SIM_CACHE = "droid_fr3_v0_asset_replay.pt"  # IsaacLab 400/80 on the corrected plant


def runs_by_joint(preset: str) -> dict:
    out = {}
    for pt in (ROOT / preset / "franka_fr3/sim").glob("*/chirp_data_prepared.pt"):
        ds = validate_contract(load_dataset(str(pt)))
        out[ds.active_joint_names[0]] = (pt, ds)
    return out


def main() -> None:
    per_preset = {p: runs_by_joint(p) for p, _, _, _ in PRESETS}
    joints = sorted(per_preset["franka_high"])
    fig, axes = plt.subplots(4, 2, figsize=(16, 13), sharex=True)
    flat = axes.ravel()
    short = {"franka_ros": "ros", "franka_high": "high", "franka_low": "low"}
    for ax, joint in zip(flat, joints):
        pt_high, ds_high = per_preset["franka_high"][joint]
        col = ds_high.joint_names.index(joint)
        time = ds_high.time.numpy()
        cmd = ds_high.des_dof_pos[:, col].numpy()
        ax.plot(time, cmd, color="tab:purple", linestyle="-.", linewidth=1.3, label="commanded")
        tops = [cmd.max()]
        stats = []
        for preset, label, color, style in PRESETS:
            _, ds = per_preset[preset][joint]
            c = ds.joint_names.index(joint)
            meas = ds.dof_pos[:, c].numpy()
            ax.plot(time, meas, color=color, linestyle=style, linewidth=1.3, label=label)
            tops.append(meas.max())
            rmse = float(np.sqrt(np.mean((meas - cmd) ** 2)) * 1e3)
            stats.append(f"{short[preset]} {ds.kp_used[c]:g}/{ds.kd_used[c]:g}: {rmse:.0f}")
        cache = torch.load(pt_high.parent / SIM_CACHE, map_location="cpu", weights_only=False)
        sim = np.asarray(cache["default_traj"])
        ax.plot(time, sim, color="tab:red", linestyle="--", linewidth=1.3,
                label="IsaacLab high-PD 400/80 (sim estimate)")
        tops.append(sim.max())
        rmse_sim = float(np.sqrt(np.mean((sim - cmd) ** 2)) * 1e3)
        stats.append(f"IL 400/80: {rmse_sim:.0f}")
        ax.set_title(f"{joint}   RMSE [mrad]  " + " | ".join(stats), fontsize=9.5)
        ax.set_ylabel("position [rad]")
        ax.grid(True, linestyle=":", alpha=0.6)
        # Symmetric chirp: zoom to the upper half for readability.
        center = float(np.mean(cmd))
        top = float(max(tops))
        ax.set_ylim(center, top + 0.12 * (top - center))
        # Second half of the sweep: the high-frequency band separates the controllers.
        ax.set_xlim(time[len(time) // 2], time[-1])
    for ax in flat[len(joints):]:
        ax.set_visible(False)
    for ax in flat[len(joints) - 2: len(joints)]:
        ax.set_xlabel("time [s]")
    handles, labels = flat[0].get_legend_handles_labels()
    fig.suptitle("Which PD gains track best: real responses per preset, identical commands", y=0.995, fontsize=14)
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.975))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path("sysid_plots/summary_1_controller_tracking_traj.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
