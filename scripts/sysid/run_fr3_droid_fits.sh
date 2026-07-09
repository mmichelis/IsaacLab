#!/bin/bash
# Residual fits on the corrected FR3 droid plant (Isaac-Sysid-Droid-FR3-v0):
# all franka_high per-joint chirp runs in parallel, then the before/after and
# generation plots. Outputs are suffixed _fr3 so earlier (as-is plant) plots
# and fit dirs are preserved.
set -u
cd /workspace/isaaclab
source env_isaaclab/bin/activate
export PYTHONUNBUFFERED=1

TASK=Isaac-Sysid-Droid-FR3-v0
DATA_ROOT=logs/sysid/prepared_datasets/franka_high/franka_fr3/sim
FIT_DIR=logs/sysid/droid_fr3_franka_high
STAMP_FILE=$FIT_DIR/.batch_start
mkdir -p "$FIT_DIR"
touch "$STAMP_FILE"

pids=()
for run in "$DATA_ROOT"/*/; do
  name=$(basename "$run")
  echo "[driver] launching fr3 fit for $name"
  python scripts/sysid/fit.py --task "$TASK" --num_envs 64 \
    --data "$run/chirp_data_prepared.pt" \
    --log_dir "$FIT_DIR" \
    --max_iterations 60 \
    --warmstart_from_data \
    --kit_args="--/crashreporter/enabled=false" \
    --visualizer none > "logs/fit_fr3_$name.log" 2>&1 &
  pids+=($!)
  # Stagger kit boots: concurrent cold starts race on the shared extension cache.
  sleep 45
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=$((fail + 1))
done
echo "[driver] fr3 fits done ($fail failed)"

FITTED_ARGS=$(python - <<'EOF'
import json
from pathlib import Path

fit_dir = Path("logs/sysid/droid_fr3_franka_high")
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
echo "[driver] plotting with: $FITTED_ARGS"
python scripts/sysid/plot_droid_fit.py \
  --task "$TASK" \
  --data_root "$DATA_ROOT" \
  --fitted $FITTED_ARGS \
  --default_gains recorded \
  --out sysid_plots/droid_fr3_fit_before_after.png \
  --kit_args="--/crashreporter/enabled=false" --visualizer none \
  > logs/plot_fr3_fit.log 2>&1
python scripts/sysid/plot_droid_generations.py \
  --data_root "$DATA_ROOT" \
  --fitted_root "$FIT_DIR" \
  --out_prefix sysid_plots/droid_fr3_generations \
  > logs/plot_fr3_generations.log 2>&1
chown -R 1776605604:748400513 /workspace/isaaclab/sysid_plots /workspace/isaaclab/logs 2>/dev/null
echo "[driver] DONE -> sysid_plots/droid_fr3_fit_before_after.png + droid_fr3_generations_*.png"
