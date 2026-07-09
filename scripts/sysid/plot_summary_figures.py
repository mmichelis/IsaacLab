# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compact summary figures for the three investigations (values from the
tables in ROBOT_CONTROLLER_TRACKING.md and SIM2REAL_FINDINGS.md)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("sysid_plots")
JOINTS = [f"j{i}" for i in range(1, 8)]
X = np.arange(7)


def bars(ax, series, width=0.19):
    for k, (label, values, color) in enumerate(series):
        ax.bar(X + (k - (len(series) - 1) / 2) * width, values, width=width * 0.92, label=label, color=color)
    ax.set_xticks(X, JOINTS)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- inv 1
# Real-robot command tracking per controller (raw RMSE and lag).
rmse = {
    "franka_ros (600.../30...)": [24.3, 14.0, 20.1, 21.7, 31.3, 23.6, 68.0],
    "franka_robot_high (400.../20...)": [27.0, 17.4, 23.2, 24.8, 31.4, 25.2, 71.6],
    "franka_robot_low = IsaacLab default (80/4)": [55.8, 40.8, 63.4, 62.3, 45.6, 24.0, 43.1],
    "IsaacLab high-PD 400/80 (sim estimate)": [51.5, 26.6, 42.8, 48.3, 81.3, 38.2, 84.1],
}
lag = {
    "franka_ros (600.../30...)": [54, 64, 57, 57, 43, 73, 97],
    "franka_robot_high (400.../20...)": [59, 76, 63, 63, 43, 78, 103],
    "franka_robot_low = IsaacLab default (80/4)": [128, 226, 175, 153, 63, 73, 57],
    "IsaacLab high-PD 400/80 (sim estimate)": [140, 156, 149, 153, 135, 137, 131],
}
colors = ["tab:blue", "tab:green", "tab:orange", "tab:red"]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
bars(axes[0], [(k, v, c) for (k, v), c in zip(rmse.items(), colors)])
axes[0].set_ylabel("tracking RMSE vs command [mrad]")
axes[0].set_title("command tracking error (real robot; 400/80 sim-estimated)")
bars(axes[1], [(k, v, c) for (k, v), c in zip(lag.items(), colors)])
axes[1].set_ylabel("response lag [ms]")
axes[1].set_title("response lag behind command")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.02))
fig.suptitle("Investigation 1: which PD gains track best on the real robot", y=1.12, fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "summary_1_controller_tracking.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- inv 2
# Sim2real gap: as-is vs corrected asset, identical real gains, no fitting.
asis = [14.90, 7.02, 10.22, 25.60, 73.11, 21.06, 29.30]
corrected = [2.00, 3.18, 2.53, 2.34, 4.46, 5.12, 11.57]
fig, ax = plt.subplots(figsize=(8, 4.2))
bars(ax, [("as-is asset (phantom 7 kg, no armature/friction)", asis, "tab:red"),
          ("corrected asset (mass/armature/friction/FR3)", corrected, "tab:blue")], width=0.36)
ax.set_ylabel("sim-vs-real RMSE [mrad]")
ax.set_title("Investigation 2: sim2real gap, same real PD gains, no fitting\n(mean 25.9 -> 4.5 mrad, wrist up to 16x)")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "summary_2_asset_gap.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- inv 3
# PD identification on the corrected asset closes the remaining gap.
rig = [2.00, 3.18, 2.53, 2.34, 4.46, 5.12, 11.57]
fitted = [1.30, 2.07, 2.09, 1.95, 2.86, 2.13, 3.83]
fig, ax = plt.subplots(figsize=(8, 4.2))
bars(ax, [("corrected asset @ real gains", rig, "tab:blue"),
          ("corrected asset @ identified gains", fitted, "tab:green")], width=0.36)
ax.set_ylabel("sim-vs-real RMSE [mrad]")
ax.set_title("Investigation 3: PD identification on the corrected asset\n(mean 4.5 -> 2.3 mrad; j1-j4 identify to the real gains +-10%)")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "summary_3_pd_identification.png", dpi=150)
plt.close(fig)

print("wrote", *[f"sysid_plots/summary_{i}_*.png" for i in (1, 2, 3)])
