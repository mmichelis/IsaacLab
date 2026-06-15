# G1 Running on Soft Foot Pads — Progress Log

Goal: a training-ready RL environment for the Unitree G1 humanoid running forward at
maximal velocity, with a soft deformable "shoe sole" pad attached beneath each foot
(stiff material, E ≈ 1e6–1e7 Pa, 5–10 cm thick), then a trained running policy.

Branch: `mym/coupled_cable` (Newton coupled MJWarp + VBD solver available).

---

## 1. Design decisions (autonomous, minimal-change)

**Base env.** Built on the existing `G1FlatEnvCfg`
(`.../locomotion/velocity/config/g1/flat_env_cfg.py`). New package:
`.../locomotion/velocity/config/g1_soft_pads/`. Gym ids:
- `Isaac-Velocity-Flat-G1-SoftPads-v0` (train)
- `Isaac-Velocity-Flat-G1-SoftPads-Play-v0` (eval)

**Solver.** Replaced the rigid Newton solver with the coupled **MJWarp (robot) + VBD
(pads)** solver (`CoupledMJWarpVBDSolverCfg`, `coupling_mode="two_way"`), mirroring the
Franka cloth/soft envs. This is the only path on this branch that simulates FEM
deformables coupled to a rigid articulation.

**Pads.** Two volume-deformable cuboids (`MeshCuboidCfg` + `NewtonDeformableBodyMaterialCfg`),
one per `*_ankle_roll_link` foot. Size `(0.18, 0.09, 0.06)` m (6 cm thick — within the
requested 5–10 cm). Material started at the **low end** of the requested range, E = 1e6 Pa,
ν = 0.3, ρ = 200 kg/m³, for VBD stability; converted to Lamé `k_mu`/`k_lambda`. Will raise
toward 1e7 once stable.

**Attachment (the crux).** Newton has *no* particle↔rigid-body weld. So the pad is bound
to the foot by **kinematic targets**: every sim-step the pad's top 25% of particles are
re-pinned to the foot's current pose (`write_nodal_kinematic_target_to_sim_index`), while
the free lower particles deform against the ground. Re-pinning happens in
`apply_actions()` (runs before every `sim.step()`), so the pad tracks the foot through
swing without per-env-step lag. On reset the whole pad is teleported back under the foot.
Implemented as `JointPositionActionWithSoftPads` (subclass of the joint-position action, so
`action_dim` stays = joint count). Binding geometry is captured lazily from the spawned
(undeformed) pad shape + foot rest pose, so spawn alignment need not be exact.

**Support / force feedback.** Because kinematically-pinned (massless) particles cannot
transmit ground reaction back to the foot, the robot is held up by **two-way soft contact**
between the foot collision shape and the pad's free particles (the coupled solver's
`_apply_reactions` path). Whether this actually carries the robot's weight is the first
thing the smoke test must confirm.

**Known tradeoff — contact sensing.** With a pad between foot and ground, the MJWarp foot
no longer touches the ground, so the foot-contact reward terms (`feet_air_time`,
`feet_slide`) read ~0. They are left in place (low impact); the base-contact termination
(torso) still works. If gait quality suffers, a height-based contact proxy will be added.

**Cadence.** `sim.dt = 0.005` (200 Hz), `decimation = 4` (50 Hz policy), `num_substeps = 4`.
Robot spawn raised by the pad thickness so the soles clear the ground at reset.

**Command.** Forward only: `lin_vel_x ∈ [0, 1.5]`, `lin_vel_y = 0`, small yaw — to drive
fast forward running.

---

## 2. Status

- [x] Investigated G1 velocity envs, coupled solver, deformable API, attachment options.
- [x] Wrote env package (scene, coupled solver, pads, pad-follow action, PPO cfg, gym ids).
- [ ] Smoke test: env builds; robot is supported on pads; pads follow feet; physics stable.
- [ ] Train (start 1000 iters); inspect logdir; iterate.
- [ ] Final running policy.

(Updated as work proceeds.)

---

## 3. Run log

### Smoke-test bring-up (debugging)

1. **`configclass` import bug** — used `from isaaclab.utils import configclass` (the submodule)
   instead of `from isaaclab.utils.configclass import configclass`. Fixed.
2. **App-startup segfault (the big one).** Importing the action *implementation*
   (`JointPositionAction` → `Articulation`) at config-module load time pulled `pxr` in
   **before** `SimulationApp` launched, corrupting Kit's plugin loader → segfault during
   `_start_app`. Diagnosed by confirming stock G1 + franka-soft launch fine, then bisecting
   which import loaded `pxr`. Fix: split the action term into an import-light **cfg**
   (`soft_pads.py`) and a lazily-imported **impl** (`soft_pads_impl.py`) referenced by a string
   `class_type` — the same pattern the cable-plug env uses for `UniformPoseCommand`.
3. **Contact sensor unsupported by coupled solver.** The coupled MJWarp+VBD manager's dummy
   solver raises `NotImplementedError` on `update_contacts`. Removed the `contact_forces`
   sensor; nulled `feet_air_time`/`feet_slide`; replaced `base_contact` termination with
   `bad_orientation` (tilt > ~57°) + `root_height_below_minimum` (root z < 0.5 m).
4. **Pads exploded (non-finite by step 7).** The `ankle_roll_link` origin sits ~6 cm *above*
   the sole (measured 0.061 m on stock G1), so a pad placed 3 cm below the foot origin was
   buried inside the foot collision → huge penetration forces. Fixed placement
   (`pad_z_offset = -(0.06 + thickness/2)`, pad top at the sole) and hardened stability for
   bring-up: E 1e6→3e5, `num_substeps` 4→8, VBD iters 10→15, contact `ke` 2e4/4e4→1e4,
   `pin_fraction` 0.25→0.15.

**Result (100-step zero-action rollout, 4 envs):** stable, no NaNs. Robot stays upright
(base_z ≈ 0.6–0.75, not collapsing) → **pads carry the robot via two-way soft contact**.
Pads track the feet (xy error settles to ~0.4 cm). Known imperfection: pads sink ~8 cm into
the ground under load (contact stiffness modest) — to be stiffened during training.

### Training

**Run 1 — direct (128 envs, soft-pad env).** Pipeline works end-to-end (env builds, PPO
iterates, checkpoints). But **throughput ≈ 67 steps/s, ~45 s/iteration** — the coupled
MJWarp+VBD solver (8 substeps × 4 decimation × FEM contact, ×128 envs) is ~3 orders of
magnitude slower than rigid G1 locomotion. Early metrics: episode length ~20 steps (~0.4 s)
— the random policy falls almost immediately (`base_contact`/tilt termination ~85%). At this
rate, learning humanoid running *from scratch* (needs 1e8–1e9 env steps) is impractical
(1000 iters ≈ 12 h and far too little data). **Stopped.**

**Strategy — warm-start (pretrain rigid → fine-tune soft).** The soft-pad env's observation
(123) and action (37) spaces are **identical** to the stock rigid `Isaac-Velocity-Flat-G1-v0`
(it inherits that obs/action cfg unchanged), and the PPO network is the same shape
([256,128,128]). So:
  1. Pretrain a forward-running policy on the fast rigid `Isaac-Velocity-Flat-G1-v0`
     (PhysX, 4096 envs) — minutes/hours to a competent runner.
  2. Fine-tune that checkpoint on `Isaac-Velocity-Flat-G1-SoftPads-v0` — the policy only has
     to adapt to the compliant soles + slightly taller stance, needing far fewer of the
     expensive coupled-solver iterations.

**Run 2 — pretraining** `Isaac-Velocity-Flat-G1-v0` (rigid, PhysX), 4096 envs, 1000 iters.
~18k steps/s, ~3.4 s/iter. **Result: a competent forward runner** — mean reward 21.0,
episode length ~985/1000, success_rate 1.0, `error_vel_xy` 0.14. Checkpoint:
`logs/rsl_rl/g1_flat/2026-06-12_15-21-01/model_999.pt`.

**Run 3 — soft-pad fine-tune (PhysX warm-start) — abandoned.** Warm-starting the soft-pad
(Newton-coupled) env from the **PhysX** `model_999` fell immediately. Root cause: a large
**PhysX→Newton sim-gap**, not just the soft pads — the same gap makes the PhysX policy topple
when merely *rendered* on the Newton backend.

**Run 4 — proper Newton rigid baseline.** Train on the existing Newton G1 locomotion env
(`Isaac-Velocity-Flat-G1-v0`, `presets=newton_mjwarp` — full reward suite, working Newton
contact sensors), warm-started from the PhysX `model_999`. Fast (~1.1 s/iter, ~80k steps/s).
Recovery is slow-then-fast (PPO): ep-length 44 (iter 1064) → 350 (1314) → 815 (1499),
vel-error 1.1 → 0.42. Being extended to convergence (target ep-length ~950+, vel-error <0.3) to
serve as the **Newton-native warm-start** for the soft-pad fine-tune (Newton→Newton+pads is a far
smaller gap). This is the correct baseline to start the soft-feet policy from.

### Rendering a still of the pads

RTX camera (`scripts/_g1_soft_pads_capture.py`): the G1 renders cleanly, but the **dynamic FEM
pad meshes do not stream to the RTX render delegate** (`[rtx.hydra]
readTransformsFromFabricInRenderDelegate + geometry streaming` warning) — a known Isaac Sim
limitation for deformables through a camera sensor.

**Newton native viewer (`scripts/_g1_soft_pads_newton_render.py`) — works.** A headless
`newton.viewer.ViewerGL(headless=True)` renders the deformables directly: `get_model()` /
`get_state()` from `NewtonManager`, `set_model` + `camera.look_at`, `begin_frame/log_state/
end_frame`, then `get_frame()` → PNG. `/tmp/g1_soft_pads_newton3.png` shows the G1 on two gold
soft soles, flush with the feet and resting on the ground. Gotcha: import `newton.viewer`/
`pyglet` only *after* the app launches (importing earlier corrupts Kit startup, same class of
bug as the `pxr` preload).

### Finalized pad physics (after review)

* **Stiffness raised to E = 1e6 Pa** (Lamé `k_mu`/`k_lambda`), `num_substeps` 8→10, VBD iters
  15→20 for stability of the stiffer FEM.
* **Placement:** pad top overlaps the foot by ~1.5 cm (`_PAD_Z_OFFSET = -(0.061 - 0.015 +
  thickness/2)`) so the sole is in solid contact with the foot — no gap.
* **Contact:** `soft_contact_ke`/`shape_material_ke` raised to 4e4 so the pad rests on the ground
  with minimal penetration. Smoke (spawn pose): `pad_top 0.064 > sole 0.047` (foot contact) and
  `pad_bottom ≈ 0.004` (on the ground). Clipping only appears when the *untrained* zero-action
  robot crouches its feet below ground level — a control artifact, not a placement bug.

### Pad contact investigation (standalone) + fix

Reviewer flagged the pads looking like they fall through the ground with no friction.
Standalone drop test (`scripts/_g1_pad_drop_test.py`, pads freed from foot-pinning):

* **Ground collision works.** A free pad dropped from 0.25 m hits the floor at step ~10
  (`pad_min_z ≈ 0`), bounces, and settles resting on the terrain plane (`pad_min_z ≈ 0.008`,
  never passing through). Rendered frame shows both pads sitting squished on the floor.
* **Friction works.** A pad kicked at 1 m/s slides only ~8 cm and stops — decent grip, not
  frictionless.

So collision/friction are fine. The env's fall-through was the **kinematic pinning dragging
the pad below the floor when the robot's feet drop below ground level** (crouch/stumble).

**Fix:** clamp each pinned kinematic target to `ground + (rest height above pad bottom)`
(`clamp_to_ground` in the action term). The pad can lift with the foot during swing but can
never be forced through the floor. Smoke (zero-action, crouchy): `pad_bottom` now stays ≈ 0.00
(was −0.07), pads lift cleanly during swing, and the robot is better supported (base 0.61–0.74).
Render: `/tmp/g1_soft_pads_fixed.png`.

**Run 5/6 — soft-pad fine-tune (Newton warm-start) — SUCCESS.** Warm-started from the converged
Newton baseline (`model_1800`) on the corrected env (E=1e6, ground clamp, foot-aligned soles).
After the orientation fix the policy converges into a **running soft-sole gait**:

| iter | ep length | vel-error | reward |
|---|---|---|---|
| 1806 | ~90  | 0.63 | -4.4 |
| 1846 | ~270 | 0.34 | -3.6 |
| 1902 | ~577 | 0.28 | -1.1 |
| 2002 | ~735 | 0.25 | +0.2 |
| ~2100 | ~725 (plateau) | ~0.3 | ~0 |

Final policy: `logs/rsl_rl/g1_soft_pads/2026-06-12_20-51-53/model_2100.pt` — the **G1 runs forward
on the compliant soles**. Hero video: `/tmp/g1_soft_pads_running.mp4`. (~180 s/iter while sharing
the GPU with the CablePlug run; the Newton warm-start + correct sole orientation are what made it
trainable in a few hundred iters.)

### Pad orientation bug + fix

Reviewer noticed the soles looked wrongly oriented. Measured it: foot heel→toe (foot local x)
pointed e.g. world `(-0.91,-0.40)` while the **pad's long axis stayed locked to world +x**
(|pad_long·foot_fwd| was small). Cause: the binding rotated the rest layout by
`quat_conjugate(foot_quat_rest)`, leaving the pad **world-axis-aligned** instead of foot-aligned —
so once the robot turned, the sole was rotated relative to the foot. Fix: express the rest layout
directly in the foot frame (`rest_local = z_offset + pad_local`, no world rotation), mapping box
x→foot heel-toe and box z→sole normal. Re-measured: **|pad_long·foot_fwd| = 0.996** (aligned).
Render: `/tmp/g1_soft_pads_oriented.png`. The soft-pad fine-tune was restarted on the corrected env.

### Videos

Both rendered through the **Newton viewer** (headless `ViewerGL`) for a matching look
(`scripts/_g1_newton_video.py`, generic JIT-policy driver, robot-following camera aimed at the
legs so the full body + feet are framed: `--eye_off 2.0 -1.8 0.3 --target_off 0 0 -0.3`).
Both use the **converged Newton baseline** policy (`model_1800`):

* `/tmp/g1_rigid_running_v3.mp4` — rigid G1 (`presets=newton_mjwarp`) running **upright** on the
  Newton backend, full body + feet.
* `/tmp/g1_soft_pads_v3.mp4` — same policy on the soft-sole env; gold soles visible under both
  feet. It stumbles since it isn't yet trained on the compliant soles (that's the in-progress
  soft-pad fine-tune).

Note: a Newton-rigid video from the *PhysX* policy was NOT upright (PhysX→Newton sim-gap) — which
is exactly why the proper Newton baseline was trained.

### Rigid foot-ground penetration + stiff-contact retrain

The stock Newton rigid baseline lets the feet punch ~4–5 cm through the ground at footstrike
(worst-case `ankle_roll` origin z ≈ −0.044 m). This is the upstream MJWarp contact (`num_substeps=1`,
`nconmax=10`), unrelated to the soft-pad work. MJWarp doesn't expose `solref`/`solimp`, so the
levers are contact *resolution*: new task `Isaac-Velocity-Flat-G1-StiffRigid-v0`
(`config/g1_soft_pads/rigid_stiff_env_cfg.py`) — concrete kitless `NewtonCfg`, `num_substeps`
1→4, `nconmax` 10→80, `njmax`→300, + Newton contact sensor. Warm-started retrain from the Newton
baseline `model_1800` (fast, ~1.9 s/iter) converged to a **clean runner: ep-len 998, vel-error
0.13, reward 20.8** (`logs/rsl_rl/g1_flat/2026-06-15_09-00-48/model_2199.pt`). Visually the feet
now plant flat on the ground (`/tmp/g1_rigid_stiff_running.mp4`); the instantaneous worst-case
footstrike spike is similar in magnitude but no longer the sustained visible toe-drag. A fuller fix
would require setting MuJoCo `solref`/`solimp` on the model post-build (no cfg hook in the rigid
Newton manager today).
