# Cable-Plug RL Tuning — Progress Summary

Task: `Isaac-Lift-CablePlug-Franka-v0` (Franka grasps a cable-tethered plug, transports it to a
target pose, bonus = insert into socket). Command:
`./isaaclab.sh train --rl_library rsl_rl --task Isaac-Lift-CablePlug-Franka-v0 --num_envs 4096 --headless`

Where to look:
- This file = consolidated summary.
- Task list (`/tasks`) = per-iteration notes.
- `tmp/runs/iterN.log` = raw training stdout per run.
- Checkpoints/tensorboard: `logs/rsl_rl/franka_deformable/<timestamp>/`.

## CURRENT STATUS (iter 14, running, ~iter 1250/1478)
- **Position transport SOLVED** (narrowed goal range): deterministic policy gets the plug within
  15 cm of target 63% of the time, within 10 cm 32%, median 12.9 cm (iter 13 eval). Was stuck
  ~30 cm for the first 11 iterations.
- **Orientation now WORKING** (iter 14): reformulated to long-axis alignment (correct for a
  cylinder); axis-alignment reward climbed from flat ~0.005 to ~4.4 — the plug axis aligns to the
  bore. Position stays precise (~0.135) while it does this.
- **Insertion**: starting to occur occasionally (rare); downstream of precise pose.
- Training robust: solver-NaN crash + Isaac-Sim startup segfault both fixed.
- VIDEO: headless Newton-GL capture only renders an auto-framed plug view (no camera control / not
  full scene) — a viewer limitation. For a real clip, run play WITHOUT --headless on a display
  machine. Per user: dropped video, focus on training.

## BEST CHECKPOINTS
- **BEST overall (smoother arm + pose + fewer drops): `logs/rsl_rl/franka_deformable/2026-06-15_13-45-30/model_1826.pt` (iter 15)**
- Most complete pre-smoothing: `logs/rsl_rl/franka_deformable/2026-06-15_08-43-56/model_1477.pt` (iter 14)
- Precise position only: `logs/rsl_rl/franka_deformable/2026-06-15_00-37-05/model_1078.pt` (iter 13)

## ITER 15 (slower/smoother arm, user request) — DONE
- Raised motion penalties (joint_vel -1e-3, action_rate -8e-4, joint_acc -5e-5), entropy 0.0002.
- Position IMPROVED: det median 0.127->0.110, within 10cm 31%->40%, within 15cm 63%->72%.
- Drop rate HALVED (0.10 -> 0.054). std 0.59 -> 0.49. No regression / do-nothing.
- iter-15 arm joint speed: mean 0.66 rad/s, median 0.32, p90 1.56 (baseline iter-14 comparison pending).

## ITER 14 FINAL (done, crash-free)
- position precise: POSERR ~0.13 (min 0.116), near_goal ~1.8, drop ~0.10, std down to 0.59.
- orientation axis-alignment reward ~4.8 (flat ~0.005 for 13 iters before the reformulation).
- insertion firing occasionally (max ~0.13).

## ITERATION LOG (change -> outcome)
| # | Key change | Outcome |
|---|---|---|
| 1 | warm-start + dense insertion shaping | REWARD HACKING (farmed ungated axis term); aborted |
| 2 | remove gripper-close penalty; gate goal on "held"; fresh | reach + grasp SOLVED; plug held, not transported |
| 3 | dual-kernel goal (coarse+fine) | transport started but std inflated (entropy too high) -> regressed |
| 4-5 | chase std via entropy | wrong lever; deterministic eval showed std wasn't the blocker; LR was floored |
| 6 | desired_kl 0.016 (un-throttle LR) | LR fixed, but transport still flat -> behavioral local optimum |
| 7 | action scale 0.5->0.2 + heavy penalties (fresh) | DO-NOTHING collapse (over-aggressive on a fresh run) |
| 8 | scale 0.2, restore drivers | std settled healthy 0.84 (fix!) but scale 0.2 too slow for reach |
| 9 | scale 0.5 + entropy 0.005 (fresh) | reach+grasp fast, std 0.84; crashed at iter 238 (solver NaN) |
| 10-11 | log-std (crash-robust) | still NaN-crashed ~iter 280 (NaN std, not negative) |
| 12 | NARROW goals + fixed-per-episode + crash fix | TRANSPORT BREAKTHROUGH: plug reaches goal (median 0.185) |
| 13 | entropy 0.005->0.0005 (precision) | PRECISE: median 0.129, within 15cm 63%, within 10cm 32% |
| 14 | orientation reward -> axis-alignment (running) | orientation reward now climbing (0.04 -> 3.5) |

## CURRENT TUNED CONFIG (vs original)
Rewards: reaching 8, grasp_plug 6->4, goal_tracking coarse 16->12, goal_tracking_fine (NEW) 18,
goal_orientation (NEW, axis-align) 12, near_goal (NEW, <5cm) 5, plug_inserted 10;
removed lifting_plug and gripper_close penalty; action_rate/joint_vel -2e-4.
Goal ranges NARROWED (pos_y ±0.1, pitch ±20°, yaw ±45°), fixed per episode.
PPO: entropy_coef 0.0005, noise_std_type "log", desired_kl 0.016, num_learning_epochs 3.
Env: velocity_divergence termination tightened to 25/10; obs+reward NaN guards in wrapper.

## REMAINING / NEXT
- Confirm orientation alignment converges (iter 14 deterministic eval).
- Then widen goal ranges back toward the original (full) distribution (iter 15).
- Push insertion (dense seating term, gated on aligned+near) once pose is precise.

## NOTES
- Code changes: env_cfg, rsl_rl_ppo_cfg, mdp/rewards.py (+grasp_plug, object_goal_orientation,
  object_near_goal), mdp/__init__.pyi, franka_cable_plug_env.py (obs guard). NOT committed (per your rule).
- `scripts/.../play_rsl_rl.py` has a temporary `--eval_steps` instrumentation for deterministic eval — revert before any PR.
