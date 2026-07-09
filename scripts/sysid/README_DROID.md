# Droid sysID: fitting DROID_CFG arm gains from robot-control chirp data

Fits per-joint implicit-actuator `{stiffness, damping}` for the **droid articulation
used by dextrah training** (fabrics-sim `droid_robotiq.usd`, PhysX) by replaying
chirp recordings collected with the dex/robot-control stack. Everything below was
set up and tested on this machine (July 2026); all changes are local and
uncommitted — see "Local changes vs upstream" at the bottom.

The upstream pipeline is `scripts/sysid/` from `mmichelis/IsaacLab@franka-sysid`
(see `README.md` next to this file). This document covers the droid-specific
additions plus the exact runbook for this machine.

## What exists

| Piece | Where |
| --- | --- |
| Checkout (branch tip `1df78c9` + local changes) | `~/workspaces/IsaacLab-sysid` |
| Container | `sysid_container` (from `nvcr.io/nvidia/isaac-sim:5.1.0`, used only for its GPU/GL system libs — the actual runtime is a **python 3.12 venv + pip Isaac Sim 6.0.1** at `env_isaaclab/` in the checkout) |
| New task | `Isaac-Sysid-Droid-v0` (`source/.../contrib/sysid/config/droid/`) — droid USD, PhysX-only, zero-g, fixed base, `fr3_jointN -> panda_jointN` name map, Robotiq joints held at home with DROID_CFG gains (200/10) |
| Dataset prep | `scripts/sysid/prepare_robot_control_dataset.py` (robot-control collections predate the fail-closed contract — this fills `shaper_type`, `kp_used/kd_used`, `safety_controller` from stamped ground truth) |
| Prepared data | `logs/sysid/prepared_datasets/<preset>/franka_fr3/sim/<run>/chirp_data_prepared.pt` (21 runs: franka_high / franka_low / franka_ros x j1-j7) |
| Control task | `Isaac-Sysid-Franka-FR3-v0` (upstream FR3 replay env; run it on the same data with `physics=physx` as the plant-matched control) |

## Runbook

### 0. Container (once per boot / recreation)

```bash
xhost +local:    # on the host, allows the container to open windows on your display

docker run -d --name sysid_container \
  --gpus all --network host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ACCEPT_EULA=YES \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v ~/workspaces/IsaacLab-sysid:/workspace/isaaclab \
  -v ~/workspaces/nvblox_next/submodules/fabrics-sim:/workspace/fabrics-sim:ro \
  -v ~/datasets/chirp_data_robot_control:/datasets/chirp_data_robot_control:ro \
  --entrypoint bash nvcr.io/nvidia/isaac-sim:5.1.0 -c "sleep infinity"
```

**Gotcha — after every container recreation** the venv's python interpreter is gone
(it lives under container-local `/root/.local/share/uv/`; the venv on the mount
only symlinks it). Restore it with:

```bash
docker exec -u 0 sysid_container bash -c \
  'curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; \
   /root/.local/bin/uv python install cpython-3.12.13'
```

Everything else (venv packages, isaac sim, generated assets, prepared datasets,
fit results) lives on the mounted checkout and survives.

Get a shell for all following steps:

```bash
docker exec -u 0 -it sysid_container bash
cd /workspace/isaaclab && source env_isaaclab/bin/activate
```

Preset naming: the collection-tree directories `franka_high` / `franka_low`
are the dex/robot-control FrankaDriver gain presets - in prose we call them
**franka_robot_high** / **franka_robot_low** to avoid confusion with
IsaacLab's FRANKA_PANDA_*_CFG (robot-control's yaml labels the high preset
"IsaacLab FRANKA_PANDA_HIGH_PD_CFG", but the numbers differ). Paths in the
commands below keep the original directory names.

### 1. Prepare datasets (already done for the current 21 runs)

```bash
python scripts/sysid/prepare_robot_control_dataset.py \
  --input_root /datasets/chirp_data_robot_control \
  --output_root logs/sysid/prepared_datasets
```

Re-run only when new collections land. Originals are never modified. The script
hard-fails if the stamped PD yaml is ambiguous.

### 2. Fit — headless (production)

One fit per per-joint run (each run excites one joint; `fit.py` reads
`active_joint_names` and fits only that joint's 2 params, holding the other six
at the dataset's `kp_used/kd_used`):

```bash
python scripts/sysid/fit.py --task Isaac-Sysid-Droid-v0 --num_envs 64 \
  --data logs/sysid/prepared_datasets/franka_high/franka_fr3/sim/2026-07-07_17-47/chirp_data_prepared.pt \
  --log_dir logs/sysid/droid_franka_high \
  --kit_args="--/crashreporter/enabled=false" \
  --visualizer none
```

All seven joints of a preset:

```bash
for run in logs/sysid/prepared_datasets/franka_high/franka_fr3/sim/*/; do
  python scripts/sysid/fit.py --task Isaac-Sysid-Droid-v0 --num_envs 64 \
    --data $run/chirp_data_prepared.pt \
    --log_dir logs/sysid/droid_franka_high \
    --kit_args="--/crashreporter/enabled=false" \
    --visualizer none
done
```

Timing on the RTX A6000: ~90 s per CMA-ES generation at 64 envs; a 2-param fit
plateaus well before the 200-generation cap (budget roughly 1-2 h per joint run,
sequential). Results land in `logs/sysid/droid_franka_high/<stamp>/`:

- `fitted_parameters.txt` — human-readable per-joint gains (CMA mean, rerolled +
  best candidate)
- `fit_result.json` — machine-readable summary + provenance
- `best_candidate.pt`, `mean_*.pt` — parameter artifacts for `--eval_params`
- tensorboard event files (`tensorboard --logdir logs/sysid/droid_franka_high`)

### 3. Fit — with the viewer (watch the robot)

Same command, swap the visualizer flag:

```bash
python scripts/sysid/fit.py --task Isaac-Sysid-Droid-v0 --num_envs 8 \
  --data logs/sysid/prepared_datasets/franka_high/franka_fr3/sim/2026-07-07_17-47/chirp_data_prepared.pt \
  --log_dir logs/sysid/droid_viz \
  --kit_args="--/crashreporter/enabled=false" \
  --visualizer kit
```

A full Isaac Sim viewport opens on your display showing the parallel droids
replaying the chirp (each env runs a different gain candidate — visible as
slightly different arm phases). Use a small `--num_envs` so the window stays
responsive; treat viewer runs as inspection, not production fits.

Quick 30-second sanity check with the viewer (no fitting):

```bash
python scripts/sysid/smoke_test.py --task Isaac-Sysid-Droid-v0 --num_envs 4 \
  --kit_args="--/crashreporter/enabled=false" --visualizer kit
```

### 3b. Plotting before/after overlays

`scripts/sysid/plot_droid_fit.py` renders the per-joint overlay grid (measured vs
commanded vs baseline sim vs fitted sim, RMSE per joint — same layout as the FR3
`joint_fit_before_after` figure). PNGs go to `sysid_plots/` in the repo root:
that directory is NOT gitignored (unlike `logs/`), so the images are visible in
VS Code. Nothing in this checkout gets committed either way.

```bash
# One sim pass: replays every run with the baseline gains AND the fitted
# candidates, caches each run's baseline rollout next to the dataset
# (droid_<baseline>_replay.pt). ~10 min on an idle GPU.
python scripts/sysid/plot_droid_fit.py \
  --data_root logs/sysid/prepared_datasets/franka_high/franka_fr3/sim \
  --fitted_root logs/sysid/droid_franka_high \
  --default_gains recorded \
  --out sysid_plots/droid_fit_before_after.png \
  --kit_args="--/crashreporter/enabled=false" --visualizer none

# Instant re-plot, no sim (works mid-fit; fitted line = the optimizer's own
# best_trajectory.pt, baseline from the cache; skips runs with no cache yet):
python scripts/sysid/plot_droid_fit.py \
  --data_root logs/sysid/prepared_datasets/franka_high/franka_fr3/sim \
  --fitted_root logs/sysid/droid_franka_high \
  --default_gains recorded \
  --out sysid_plots/droid_fit_current.png \
  --mode cached
```

- `--default_gains recorded` = baseline sim runs the rig's stamped kp/kd (sim
  configured like the real controller). `asset` = DROID_CFG's 400/80.
- `--fitted_root` picks the freshest `best_candidate.pt` per joint; drop it (and
  `--fitted`) for a 3-line plot without the fitted overlay.
- `scripts/sysid/run_franka_high_droid_fits.sh` runs all 7 fits in parallel and
  renders `sysid_plots/droid_fit_before_after.png` automatically at the end.

### 3c. Plant variants

Three registered droid tasks (see SIM2REAL_FINDINGS.md for the evidence):

- `Isaac-Sysid-Droid-v0` — the fabrics USD as-is (dextrah training plant today,
  including the ~7 kg phantom-frame runtime fallback mass).
- `Isaac-Sysid-Droid-Corrected-v0` — overlay fixes the phantom masses; actuator
  cfg adds menagerie FR3 armature + per-joint friction. At the REAL recorded
  gains this plant tracks the real robot at 2-5 mrad (vs 7-73 mrad as-is).
- `Isaac-Sysid-Droid-FR3-v0` — corrected plant + Panda->FR3 conversion (FR3
  link masses and FR3 joint limits, incl. the j6 window shift).

Swap `--task` in any fit/plot command. Baseline caches are task-qualified, so
per-plant replays coexist; `plot_asset_compare.py` overlays two cached plants
against the real data:

```bash
python scripts/sysid/plot_asset_compare.py \
  --data_root logs/sysid/prepared_datasets/franka_high/franka_fr3/sim \
  --cache_a droid_v0_recorded_replay.pt --label_a "as-is asset" \
  --cache_b droid_corrected_v0_recorded_replay.pt --label_b "corrected asset" \
  --out sysid_plots/droid_asset_compare.png
```

### 4. Interpreting the fitted gains

The data is the REAL FR3 (with the Robotiq mounted) under robot-control's
FrankaDriver: libfranka torque mode plus a 1 kHz host PD at the stamped
kp_used/kd_used, no command shaping. libfranka gravity-compensates internally
in torque mode, which is what the env's zero-gravity setting models.

The fitted gains are therefore EFFECTIVE sim gains: they absorb everything the
PhysX droid articulation gets wrong about the real plant (armature, joint
friction, link inertias, motor dynamics). Do not expect them to match the real
controller's gains — matching the real trajectory is the objective, matching
the real gain numbers is not. Empirically the droid asset needs several times
stiffer gains than the rig ran, strongest at the wrist.

A/B on the bare-FR3 URDF plant (different asset, same data) is available via
the upstream task; differences between its fit and the droid fit isolate how
much of the gain inflation comes from the droid USD specifically:

```bash
python scripts/sysid/fit.py --task Isaac-Sysid-Franka-FR3-v0 --num_envs 64 \
  --data logs/sysid/prepared_datasets/franka_high/franka_fr3/sim/2026-07-07_17-47/chirp_data_prepared.pt \
  --log_dir logs/sysid/fr3_urdf_physx \
  --kit_args="--/crashreporter/enabled=false" \
  --visualizer none physics=physx
```

The FR3 asset is already generated; regenerate after a clean checkout with
`python scripts/sysid/prepare_fr3_asset.py` + the `convert_urdf.py` command it
prints.

### 5. Feeding results back into dextrah

Collect the per-joint `fitted_parameters.txt` rows across the seven runs of the
preset the real robot will use, then update the arm `ImplicitActuatorCfg` groups
in `dextrah_surfel_extensions/assets/droid/droid.py` (currently uniform
stiffness=400 / damping=80). Per-joint values need per-joint actuator groups or
dict-valued `stiffness=`/`damping=` expressions. fit.py's
"asset_default_gains" baseline in eval mode measures exactly how far the current
training gains are from the recorded plant.

## Verified so far (2026-07-08)

- `pytest scripts/sysid/tests/`: 116 passed, 1 pre-existing failure
  (`test_stale_override_capped_at_pm_ceiling` — upstream commit `3ca1cfc`
  loosened `MAX_ALLOW_STALE_FRACTION` to 1.0 without updating the test).
- All 21 prepared runs pass the data contract (0% stale rows, exact 200 Hz).
- FR3 smoke test (Newton) and droid smoke test (PhysX, headless AND `--visualizer
  kit`) pass: per-env gain writes reach the solver, identical-gain envs match.
- FR3 and droid fits run generations on PhysX (results under
  `logs/sysid/droid_franka_high/`; plots under `sysid_plots/`).

## Local changes vs upstream (all uncommitted, `git diff` to review)

- `scripts/sysid/fit.py`
  - joint-id tensors created as **int32** — the `isaaclab_physx`
    `write_*_to_sim_index` warp kernels reject int64; the upstream
    `physics=physx` path had never been exercised.
  - `cma_es` import deferred until after `launch_simulation()` — tensorboard's
    grpc/protobuf C extensions intermittently segfault Kit when loaded first.
  - optional `sysid.sim_joint_name_map` applied when resolving sim joint indices
    (dataset `fr3_joint*` -> articulation `panda_joint*`).
- `scripts/sysid/smoke_test.py` — works for any sysid task (joint lookup by
  name via the map, no hard-coded 7-joint assumption, PhysX presets without
  `use_cuda_graph` allowed).
- `source/.../contrib/sysid/sysid_env_cfg.py` — `sim_joint_name_map` field.
- `source/.../contrib/sysid/config/droid/` — the new task (see module docstring).
- `apps/isaaclab.python.kit` — `isaacsim.util.debug_draw` made optional (not
  shipped in pip isaacsim 6.0.1; hard dep aborted every GUI launch).
- `scripts/sysid/prepare_robot_control_dataset.py`, `scripts/sysid/README_DROID.md` — new.

## Known warts

- Always pass `--kit_args="--/crashreporter/enabled=false"` (use `=`, argparse
  chokes on the space form). Crash dumps queued in the kit data dir otherwise
  re-crash subsequent boots; if boots start segfaulting, delete
  `env_isaaclab/.../isaacsim/kit/data/Kit/IsaacLab/3.0/*.dmp*`.
- `ruckig` is pinned to 0.12.2 (newest sdist fails to build against current
  scikit-build-core). Only needed by the franka_fr3 shaper path/tests — the
  robot-control datasets are `shaper_type: none` and never touch it.
- The container runs as root, so files it creates in the checkout (logs,
  caches, `__pycache__`) end up root-owned on the host:
  `sudo chown -R $USER ~/workspaces/IsaacLab-sysid` when it gets annoying.
