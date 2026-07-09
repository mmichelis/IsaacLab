#!/bin/bash
# Fits all franka_high per-joint chirp runs on Isaac-Sysid-Droid-v0 in parallel
# (one kit instance per joint, staggered boots), then renders the before/after
# overlay plot from the freshest fit artifact per joint.
#
# Run inside sysid_container:
#   bash scripts/sysid/run_franka_high_droid_fits.sh
set -u
cd /workspace/isaaclab
source env_isaaclab/bin/activate
export PYTHONUNBUFFERED=1

DATA_ROOT=logs/sysid/prepared_datasets/franka_high/franka_fr3/sim
FIT_DIR=logs/sysid/droid_franka_high
STAMP_FILE=$FIT_DIR/.batch_start
mkdir -p "$FIT_DIR"
touch "$STAMP_FILE"

pids=()
for run in "$DATA_ROOT"/*/; do
  name=$(basename "$run")
  echo "[driver] launching fit for $name"
  # warmstart_from_data: seed the CMA mean at the rig's recorded kp/kd instead
  # of the bounds midpoint (~2500 stiffness), so early generations explore
  # around the physically-motivated point.
  python scripts/sysid/fit.py --task Isaac-Sysid-Droid-v0 --num_envs 64 \
    --data "$run/chirp_data_prepared.pt" \
    --log_dir "$FIT_DIR" \
    --max_iterations 60 \
    --warmstart_from_data \
    --kit_args="--/crashreporter/enabled=false" \
    --visualizer none > "logs/fit_droid_$name.log" 2>&1 &
  pids+=($!)
  # Stagger kit boots: concurrent cold starts race on the shared extension cache.
  sleep 45
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=$((fail + 1))
done
echo "[driver] fits done ($fail failed)"

# Freshest artifact per joint from this batch -> plot args.
FITTED_ARGS=$(python - <<'EOF'
import json
from pathlib import Path

fit_dir = Path("logs/sysid/droid_franka_high")
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
# sysid_plots/ is outside the gitignored logs/ tree so the PNGs show up in editors.
# Baseline = the rig's recorded gains (sim configured like the real controller).
python scripts/sysid/plot_droid_fit.py \
  --data_root "$DATA_ROOT" \
  --fitted $FITTED_ARGS \
  --default_gains recorded \
  --out sysid_plots/droid_fit_before_after.png \
  --kit_args="--/crashreporter/enabled=false" --visualizer none \
  > logs/plot_droid_fit.log 2>&1
python scripts/sysid/plot_droid_generations.py \
  --data_root "$DATA_ROOT" \
  --fitted_root "$FIT_DIR" \
  --out_prefix sysid_plots/droid_generations \
  > logs/plot_generations.log 2>&1
echo "[driver] DONE -> sysid_plots/droid_fit_before_after.png + droid_generations_*.png"
