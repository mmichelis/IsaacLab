# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parse TensorBoard event files for franka_deformable RL runs and summarize scalars."""

import os

from tensorboard.backend.event_processing import event_accumulator

BASE = "/home/mmichelis/Documents/IsaacLab-Origin/logs/rsl_rl/franka_deformable"
RUNS = [
    "2026-06-12_17-02-47",
    "2026-06-12_16-04-44",
    "2026-06-09_15-15-14",
    "2026-06-09_14-44-05",
    "2026-06-09_13-17-54",
    "2026-06-09_09-45-48",
]


def summarize(events):
    vals = [e.value for e in events]
    steps = [e.step for e in events]
    return {
        "n": len(vals),
        "first_step": steps[0],
        "last_step": steps[-1],
        "start": vals[0],
        "end": vals[-1],
        "min": min(vals),
        "max": max(vals),
    }


for run in RUNS:
    run_dir = os.path.join(BASE, run)
    print("=" * 90)
    print(f"RUN: {run}")
    print("=" * 90)
    # event file lives directly in run dir
    ea = event_accumulator.EventAccumulator(run_dir, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    tags = sorted(ea.Tags()["scalars"])
    if not tags:
        print("  (no scalar tags found)")
        continue
    # max step across all tags = iterations reached
    max_step = 0
    for t in tags:
        evs = ea.Scalars(t)
        if evs:
            max_step = max(max_step, evs[-1].step)
    print(f"  Iterations reached (max step): {max_step}")
    print(f"  Num scalar tags: {len(tags)}")
    print()
    for t in tags:
        evs = ea.Scalars(t)
        s = summarize(evs)
        trend = s["end"] - s["start"]
        arrow = "UP" if trend > 0 else ("DOWN" if trend < 0 else "flat")
        print(
            f"  {t:42s} n={s['n']:4d} step[{s['first_step']}-{s['last_step']}] "
            f"start={s['start']:+.5g} end={s['end']:+.5g} "
            f"min={s['min']:+.5g} max={s['max']:+.5g} d={trend:+.5g} {arrow}"
        )
    print()
