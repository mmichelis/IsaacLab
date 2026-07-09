#!/bin/bash
# Overnight PD identification on the corrected FR3 droid plant
# (Isaac-Sysid-Droid-FR3-v0): for each real-robot gain preset (franka_high,
# franka_low, franka_ros), fit per-joint {stiffness, damping} on all 7 chirp
# runs in parallel, then render the before/after overlay (baseline = that
# preset's recorded rig gains) and the CMA-ES generation plots.
#
# Outputs per preset:
#   logs/sysid/droid_fr3_fit_<preset>/            fit runs (per-joint dirs)
#   sysid_plots/droid_fr3_fit_<preset>_before_after.png
#   sysid_plots/droid_fr3_fit_<preset>_generations_{gains,trajectories}.png
set -u
cd /workspace/isaaclab
source env_isaaclab/bin/activate
export PYTHONUNBUFFERED=1

TASK=Isaac-Sysid-Droid-FR3-v0

for preset in franka_high franka_low franka_ros; do
  DATA_ROOT=logs/sysid/prepared_datasets/$preset/franka_fr3/sim
  FIT_DIR=logs/sysid/droid_fr3_fit_$preset
  mkdir -p "$FIT_DIR"
  touch "$FIT_DIR/.batch_start"
  echo "[overnight] ===== preset $preset: launching fits ====="

  pids=()
  for run in "$DATA_ROOT"/*/; do
    name=$(basename "$run")
    python scripts/sysid/fit.py --task "$TASK" --num_envs 64 \
      --data "$run/chirp_data_prepared.pt" \
      --log_dir "$FIT_DIR" \
      --max_iterations 12 \
      --warmstart_from_data \
      --kit_args="--/crashreporter/enabled=false" \
      --visualizer none > "logs/fit_${preset}_$name.log" 2>&1 &
    pids+=($!)
    # Stagger kit boots: concurrent cold starts race on the extension cache.
    sleep 45
  done
  fail=0
  for pid in "${pids[@]}"; do
    wait "$pid" || fail=$((fail + 1))
  done
  echo "[overnight] preset $preset fits done ($fail failed)"

  FITTED_ARGS=$(FIT_DIR=$FIT_DIR python - <<'EOF'
import json
import os
from pathlib import Path

fit_dir = Path(os.environ["FIT_DIR"])
stamp = (fit_dir / ".batch_start").stat().st_mtime
latest = {}
for result in fit_dir.glob("*/fit_result.json"):
    if result.stat().st_mtime < stamp:
        continue
    joint = json.loads(result.read_text())["joint_order"][0]
    if joint not in latest or result.stat().st_mtime > latest[joint].stat().st_mtime:
        latest[joint] = result
print(" ".join(f"{j}={r.parent / 'best_candidate.pt'}" for j, r in sorted(latest.items())))
EOF
)
  echo "[overnight] $preset plotting with: $FITTED_ARGS"
  python scripts/sysid/plot_droid_fit.py \
    --task "$TASK" \
    --data_root "$DATA_ROOT" \
    --fitted $FITTED_ARGS \
    --default_gains recorded \
    --out "sysid_plots/droid_fr3_fit_${preset}_before_after.png" \
    --kit_args="--/crashreporter/enabled=false" --visualizer none \
    > "logs/plot_fit_$preset.log" 2>&1
  python scripts/sysid/plot_droid_generations.py \
    --data_root "$DATA_ROOT" \
    --fitted_root "$FIT_DIR" \
    --out_prefix "sysid_plots/droid_fr3_fit_${preset}_generations" \
    > "logs/plot_generations_$preset.log" 2>&1
  chown -R 1776605604:748400513 /workspace/isaaclab/sysid_plots /workspace/isaaclab/logs 2>/dev/null
  echo "[overnight] ===== preset $preset COMPLETE ====="
done
echo "[overnight] ALL PRESETS DONE"
