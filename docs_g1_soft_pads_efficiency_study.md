# G1 Locomotion: Rigid Feet vs. Soft Foot-Pads — Speed & Mechanical-Efficiency Study

**Question.** For the Unitree G1 humanoid, how does adding compliant FEM "shoe-sole" foot-pads
affect (1) top forward running speed and (2) joint-level mechanical efficiency (Cost of Transport)?

**Policies compared (fair pair).** Both warm-started from the same Newton baseline (`model_1800`),
same forward-command training range (0–1.5 m/s), same robot/observations/actions:
- **Rigid**: `Isaac-Velocity-Flat-G1-StiffRigid` (`model_2199`, stiffer MJWarp contact).
- **Soft**: `Isaac-Velocity-Flat-G1-SoftPads` (`model_2100`, coupled MJWarp+VBD, E=1e6 Pa pads).

**Method.** Per policy, swept fixed forward commands {0.5,1,1.5,2,2.5,3} m/s, 64 envs × 12 s each
(3 s warmup discarded). Logged per-joint applied torque & velocity, base velocity, height,
orientation, reset/fall flags at 50 Hz. Achieved speed = body-frame forward velocity `vx_b`
(world-x averages to ~0 because envs hold different headings; each env runs straight at `vx_b`).
Mechanical power P = Σ_joints τ·ω, reported as absolute Σ|τ·ω| (primary, non-regenerative
actuators) and positive-only Σmax(τ·ω,0). CoT = P̄/(m·g·v̄), m = 32.2 kg (robot body; pads excluded).
Stats over surviving envs; Welch's t-test for rigid-vs-soft. (Scripts: `scripts/_g1_efficiency_collect.py`,
`/tmp/eff_analysis.py`, `/tmp/audit*.py`; data `/tmp/eff_{rigid,soft}.npz`.)

---

## Result 1 — Top speed: the rigid policy is clearly faster

| | Rigid | Soft pads |
|---|---|---|
| Top **sustained** speed (survival ≥ 0.8) | **1.83 m/s** (cmd 2.0) | **1.22 m/s** (cmd 2.5) |
| Max achieved speed (any survival) | 2.30 m/s (cmd 2.5, surv 0.80) | 1.22 m/s |
| Command tracking @ cmd 1.5 | 1.37 m/s (−9%) | **0.97 m/s (−35%, undershoots)** |
| Saturation command | ~2.5–3.0 | **2.0** (flatlines at 1.22 m/s) |
| Both collapse at | cmd 3.0 (survival ~0.06) | cmd 3.0 (survival ~0.09) |

**The rigid policy runs ~50% faster sustained (1.83 vs 1.22 m/s)** and keeps accelerating to 2.3 m/s,
while the soft policy saturates at ~1.22 m/s and undershoots its command badly even in-distribution.
→ **Soft pads (as trained here) reduce top speed.** Note this also reflects the soft policy being
**less converged** (it plateaued at ~725/1000 episode length vs the rigid 998 during training).

## Result 2 — Efficiency: the dramatic "4× win" is an ARTIFACT; no real benefit

Naive all-37-joint CoT looked like a huge soft-pad win:

| matched ~0.9 m/s | Rigid | Soft | Δ |
|---|---|---|---|
| CoT_abs (all joints) | 4.15 | 1.03 | **−75%** (soft "better") |
| CoT_pos (all joints) | 0.41 | 0.55 | **+35%** (soft worse) |

The two conventions **disagree in sign** → dig deeper. The cause: **the rigid policy "buzzes" its 14
dexterous finger joints** — continuously spinning them (~3 rad/s at ~80 N·m), burning **~750–880 W
(65–76 % of its total joint power)** on non-locomotor motion. The soft policy holds its hands still
(~7 W). This is a pure **training artifact** (one policy learned still hands, the other didn't),
unrelated to foot compliance.

**Removing the artifact — legs-only (hips+knees+ankles), matched achieved speed:**

| matched speed | metric | Rigid | Soft | Δ (soft vs rigid) | sig. |
|---|---|---|---|---|---|
| ~0.9 m/s | CoT_abs (legs) | 0.85 | 0.83 | −2.5% | n.s. (p=0.11) |
| ~1.2 m/s | CoT_abs (legs) | 0.83 | 0.85 | +1.6% | tiny (p≈0.05) |
| ~0.9 m/s | CoT_pos (legs) | 0.35 | 0.45 | **+28%** | p<1e-19 |
| ~1.2 m/s | CoT_pos (legs) | 0.39 | 0.46 | **+18%** | p<1e-26 |

→ **At the locomotion joints, soft pads give NO efficiency benefit:** absolute-work CoT is a tie
(within noise), and positive-work CoT is ~18–28% *worse* for soft. **No impact-smoothing** benefit
was found either (base vertical-velocity oscillation not lower for soft). Gait localization: the soft
policy uses less knee/ankle but more hip power — a shorter-stride, hip-dominated gait at its speed
ceiling.

## Critical confounds (why this is "no evidence", not "proven no benefit")

1. **Finger-buzzing artifact** (dominant): inflated rigid all-joint power ~4×. Removed by legs-only
   analysis; would be fixed by an action-rate/finger penalty in training.
2. **Unmeasured pin/solver energy** (important): the soft policy's *net* joint power is implausibly
   small (net/abs ratio 0.03–0.09 vs rigid 0.33–0.87; net ≈ 0.6 W/kg is below the physical
   locomotion floor of ~1–3 W/kg). This strongly suggests the **kinematic pin + VBD pad mediate
   ground-reaction work that never appears in joint torque** → joint-work CoT *under-counts* the
   soft env's true cost. The two envs also use different solvers (MJWarp vs coupled MJWarp+VBD), so
   even the legs-only absolute-power comparison is not perfectly apples-to-apples. (Base height is
   plausible/stable, so the pin isn't grossly levitating the robot — but it likely carries support
   work.)
3. **Policy-competence mismatch**: the soft policy is gentler and slower (RMS joint torque −40%,
   saturates at 1.2 m/s). Lower-effort gait ≠ pad efficiency.

## Verdict

- **Top speed:** rigid wins decisively (~1.8 vs ~1.2 m/s sustained); the soft-pad setup is slower.
- **Mechanical efficiency:** **this study does not support that the soft pads are more efficient.**
  The eye-catching ~4× Cost-of-Transport gap is an artifact of finger-joint buzzing in the rigid
  policy plus a gentler/slower soft gait; at the actual locomotion joints, speed-matched, there is
  no efficiency advantage (and positive-work is slightly worse). A real elastic-return benefit, if
  any, is masked/confounded by uncounted pad+pin energy and the solver difference.

## Follow-up controls (done) — they confirm & strengthen the verdict

New env variants (`followup_env_cfg.py`, registered): `*-SoftPadsStiff-*` (E=4e6 pad, same coupled
solver+pin) and `*-Clean-*` (strong finger-velocity penalty). Pad kinetic energy logging added to
`scripts/_g1_efficiency_collect.py` (`--log_pad`). Data: `/tmp/eff_{soft_padE,stiffpad,rigid_clean}.npz`.

**(A) Pad-stiffness control (same soft policy, E=1e6 vs E=4e6).** Leg-work CoT is **invariant to
pad stiffness**:

| cmd | soft E=1e6: v / CoT_legs / survival | stiff E=4e6: v / CoT_legs / survival |
|---|---|---|
| 1.0 | 0.62 / 0.75 / 1.00 | 0.70 / 0.73 / 1.00 |
| 1.5 | 0.90 / 0.73 / 0.97 | 0.99 / 0.76 / 1.00 |
| 2.0 | 1.10 / 0.84 / 0.84 | 1.27 / 0.83 / 1.00 |
| 2.5 | 1.01 / 1.70 / 0.25 | 1.41 / 0.95 / **0.94** |

→ CoT_legs ≈ 0.73–0.84 regardless of pad stiffness — the **same ~0.8 the rigid feet show**. So the
soft env's leg efficiency is **not** due to pad elasticity. What the stiffer pad *does* buy is
**firmer footing**: the identical policy survives far better and runs faster on E=4e6 (cmd 2.5:
94% vs 25% survival, 1.41 vs 1.01 m/s). Compliance helps *stability ceiling*, not joint efficiency.

**(B) Pad kinetic energy.** Small: 0.16–2.0 J (E=1e6), 0.02–0.15 J (E=4e6); energy flux 8–58 W
(≈5–10% of joint power). The pads are not a large kinetic reservoir. But joint *net* power stays
~5% of absolute (vs ~33–87% for rigid), so the compliant contact still mediates the energy balance
in a way joint-work CoT cannot fully see (unmeasured pad strain/damping + kinematic-pin work).

**(C) Finger penalty.** A clean-reward variant with a finger-velocity penalty is set up and
registered; a first weak-penalty (−0.002) retrain didn't kill the buzzing (penalty bumped to
−0.02 for a future run). The legs-only analysis already removes the artifact, so the conclusion
does not depend on it.

**Bottom-line update:** across rigid feet, soft pads (E=1e6), and stiff pads (E=4e6), **leg-work
Cost-of-Transport is ~0.8 everywhere** — no efficiency advantage from foot compliance. Soft/stiff
pads change the *stability/speed ceiling* and *contact energy routing*, not actuator efficiency.

## What a fully clean follow-up would still require
1. **Equal-competence policies**: retrain both to the same top speed with identical reward +
   an **action-rate / finger penalty** (kill the buzzing), then compare legs-only CoT.
2. **Instrument the pad+pin energy**: log VBD pad strain/viscoelastic dissipation and the
   kinematic-pin constraint power so total mechanical work is conserved and comparable across solvers.
3. **Ultra-stiff-pad control**: same coupled solver + pin but near-rigid pad stiffness; if CoT then
   matches the rigid run, the differences were solver/pin, not elasticity.
4. Compare at each policy's energetically-optimal speed (CoT–speed U-curve), not just matched points.

*Methodology grounded in standard Cost-of-Transport / footwear energy-return literature (see the
design-phase notes). The multi-agent audit was essential: the headline "4× more efficient" would
have been wrong — it was finger buzzing, not soft soles.*
