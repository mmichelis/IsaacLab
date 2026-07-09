# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Real-robot command-tracking quality per PD preset. No simulation.

For every prepared chirp run, compares the MEASURED joint trajectory against
the COMMANDED one and reports, per joint and preset:

- rmse: raw tracking error
- lag: time shift maximizing cross-correlation (parabolic sub-sample refine)
- rmse_shifted: tracking error after removing the pure delay
- amp low/mid/high: measured/commanded amplitude ratio (std) in the first,
  middle and last third of the run - for a log chirp these are the low, mid
  and high frequency bands, so >1 late in the run means resonant overshoot
  and <1 means the controller can no longer follow.
- overshoot: peak measured excursion beyond the peak commanded excursion
  (percent, around the command mean)

Usage:
    python scripts/sysid/analyze_robot_tracking.py \\
        --data_root logs/sysid/prepared_datasets
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from data_contract import load_dataset, validate_contract  # noqa: E402


def _lag_ms(meas: np.ndarray, cmd: np.ndarray, dt: float) -> float:
    """Delay of meas behind cmd via cross-correlation, sub-sample refined."""
    a = meas - meas.mean()
    b = cmd - cmd.mean()
    corr = np.correlate(a, b, mode="full")
    k = int(np.argmax(corr))
    # Parabolic interpolation around the peak for sub-sample resolution.
    if 0 < k < len(corr) - 1:
        y0, y1, y2 = corr[k - 1], corr[k], corr[k + 1]
        denom = y0 - 2 * y1 + y2
        k = k + (0.5 * (y0 - y2) / denom if denom != 0 else 0.0)
    return (k - (len(b) - 1)) * dt * 1e3


def analyze_run(pt_path: Path, sim_cache: str | None = None) -> dict:
    ds = validate_contract(load_dataset(str(pt_path)))
    joint = ds.active_joint_names[0]
    col = ds.joint_names.index(joint)
    cmd = ds.des_dof_pos[:, col].numpy().astype(float)
    if sim_cache is None:
        # Real robot response.
        meas = ds.dof_pos[:, col].numpy().astype(float)
        kp = float(ds.kp_used[col])
        kd = float(ds.kd_used[col])
    else:
        # Simulated response from a plot_droid_fit.py baseline-replay cache -
        # same commands, sim plant instead of the real arm.
        cache = torch.load(pt_path.parent / sim_cache, map_location="cpu", weights_only=False)
        if cache["joint"] != joint:
            raise SystemExit(f"{pt_path.parent / sim_cache} is for {cache['joint']}, expected {joint}")
        meas = np.asarray(cache["default_traj"], dtype=float)
        kp, kd = (float(v) for v in cache["default_gains"])
    dt = ds.dt
    n = len(cmd)

    rmse = float(np.sqrt(np.mean((meas - cmd) ** 2)) * 1e3)
    lag = _lag_ms(meas, cmd, dt)
    shift = int(round(lag / 1e3 / dt))
    if 0 < shift < n:
        rmse_shift = float(np.sqrt(np.mean((meas[shift:] - cmd[: n - shift]) ** 2)) * 1e3)
    elif -n < shift <= 0:
        s = -shift
        rmse_shift = float(np.sqrt(np.mean((meas[: n - s] - cmd[s:]) ** 2)) * 1e3)
    else:
        rmse_shift = rmse

    thirds = np.array_split(np.arange(n), 3)
    amp = [float(np.std(meas[idx]) / np.std(cmd[idx])) for idx in thirds]
    center = cmd.mean()
    overshoot = float((np.abs(meas - center).max() / np.abs(cmd - center).max() - 1.0) * 100.0)
    return dict(
        joint=joint, kp=kp, kd=kd, rmse=rmse, lag=lag, rmse_shift=rmse_shift, amp=amp, overshoot=overshoot
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", type=str, required=True, help="Root containing <preset>/franka_fr3/sim/.")
    parser.add_argument(
        "--sim_cache",
        type=str,
        default=None,
        help=(
            "Analyze a SIMULATED response instead of the real one: filename of a "
            "plot_droid_fit.py baseline-replay cache next to each dataset (e.g. "
            "droid_fr3_v0_asset_replay.pt). Estimates tracking for gains never run on the robot."
        ),
    )
    args = parser.parse_args()

    for preset_dir in sorted(Path(args.data_root).iterdir()):
        runs = sorted(preset_dir.glob("franka_fr3/sim/*/chirp_data_prepared.pt"))
        if args.sim_cache is not None:
            runs = [p for p in runs if (p.parent / args.sim_cache).exists()]
        if not runs:
            continue
        source = f" [SIM: {args.sim_cache}]" if args.sim_cache else ""
        print(f"\n== {preset_dir.name}{source}")
        print(
            f"{'joint':12s} {'kp/kd':>10s} {'rmse':>7s} {'lag':>7s} {'rmse-lag':>9s} "
            f"{'amp lo':>7s} {'amp mid':>8s} {'amp hi':>7s} {'overshoot':>10s}"
        )
        rows = sorted((analyze_run(p, args.sim_cache) for p in runs), key=lambda r: r["joint"])
        for r in rows:
            print(
                f"{r['joint']:12s} {r['kp']:.0f}/{r['kd']:g}".ljust(24)
                + f"{r['rmse']:7.2f} {r['lag']:6.1f}ms {r['rmse_shift']:9.2f} "
                f"{r['amp'][0]:7.3f} {r['amp'][1]:8.3f} {r['amp'][2]:7.3f} {r['overshoot']:9.1f}%"
            )


if __name__ == "__main__":
    main()
