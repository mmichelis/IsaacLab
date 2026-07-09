# Droid sim2real: findings, plant corrections, and identified PD parameters

This work covers three investigations:

1. **Which PD gains track best on the real robot** -> see
   ROBOT_CONTROLLER_TRACKING.md (summary figure:
   `sysid_plots/summary_1_controller_tracking.png`)
2. **How the corrected asset reduces the sim2real gap** -> sections 2-5
   below (summary figure: `sysid_plots/summary_2_asset_gap.png`)
3. **How PD identification on the corrected asset reduces it further** ->
   section 6 below (trajectory figure, best-tracking preset:
   `sysid_plots/droid_fr3_fit_franka_ros_before_after.png`)

Consolidated report of the 2026-07-08/09 system-identification work: why the
PhysX droid training robot did not match the real robot, what was fixed, how
the fix was validated, and the identified PD parameters for all three real
controller presets.

- Data: real-robot chirp recordings (FR3 + Robotiq 2F-85, dex/robot-control
  FrankaDriver: libfranka torque mode + 1 kHz host PD, 200 Hz targets),
  presets franka_robot_high / franka_robot_low / franka_ros, 7 per-joint
  runs each. Naming: the "robot_" presets are dex/robot-control FrankaDriver
  gain sets. franka_robot_high is NOT IsaacLab's FRANKA_PANDA_HIGH_PD_CFG
  despite robot-control's yaml label (400x4/175/100/35 with kd 20x4/6.5x2/3.5
  vs IsaacLab's uniform 400/80). franka_robot_low DOES equal IsaacLab's
  default FRANKA_PANDA_CFG (80/4). The collection tree and artifact/figure
  paths keep the collector's original directory names franka_high /
  franka_low.
- Pipeline: `mmichelis/IsaacLab@franka-sysid` sysid stack, extended with droid
  task variants, dataset preparation, and plotting (see README_DROID.md).
- All figures in `sysid_plots/`; all fit artifacts under `logs/sysid/`.
- Training-side fix: nvblox_next PR #337 (`remos/droid_sim2real_plant_fix`).

## 1. Executive summary

With the real controller's gains copied into the sim, the training robot
lagged the real arm by 7-73 mrad RMSE per joint. The cause was the robot
model, not the controller or the timing. After correcting the plant, the same
replay tracks the real arm at 2-5 mrad with NO fitting - and per-joint PD
fits then converge to the real controller's gains on the shoulder joints,
confirming the plant is right. A consistent 1.4-2.3x wrist residual remains
and is compensated by the identified wrist gains below.

## 2. Root causes found in the training robot model

1. **Phantom TCP mass (~7 kg) - dominant.** The droid URDF's virtual
   fabric-control frames (`right_gripper`, `right_gripper_{x,y,z,+neg}`) are
   massless by design; the USD authors `physics:mass = 0`, and PhysX replaces
   every zero-mass dynamic body without collision geometry by a **1.0 kg
   runtime fallback**: 7 bodies x 1 kg, ~6 kg of it 25 cm off the gripper.
   The training arm effectively carried an invisible payload (27.1 kg total
   vs ~19 real). Invisible in the file; only a runtime dump shows it.
2. **Zero joint armature.** The real FR3 gearboxes contribute large reflected
   rotor inertia (menagerie: 0.195 kg m2 j1-4, 0.074 j5-7). No asset authors
   it.
3. **Uncalibrated friction.** USD: 0. Training injected a hand-tuned flat
   1.0 Nm on all arm joints (right ballpark j1-4, ~4x high j7). Menagerie
   reference: 1.137 j1-4, 0.763 / 0.44 / 0.248 j5-7.
4. **Sim controller was not the deployed controller.** Training used uniform
   400/80; the real FrankaDriver "high" preset runs per-joint
   400x4/175/100/35 with damping 20x4/6.5/6.5/3.5 (wrist damping 12x apart).
   Note: robot-control's yaml labels this preset "IsaacLab
   FRANKA_PANDA_HIGH_PD_CFG" but the numbers differ from IsaacLab's 400/80.
5. **Panda vs FR3.** Both sim USDs model a Panda; the real arm is an FR3:
   different link masses (17.5-18.1 vs 17.89 kg identified, different
   distribution) and different joint limits - j6 window [-1, 215] deg (Panda)
   vs [31.2, 258.8] deg (FR3), a hard feasibility mismatch for policies.

Ruled out: control/physics rates (1 kHz PD + 200 Hz ZOH reproduced exactly
from dataset stamps) and the fitting machinery (validated end to end).

## 3. Asset audit (all values read from the files / live runtime)

| Property | Real droid (spec/menagerie) | fabrics droid_robotiq.usd | Arena/PolaRiS franka_robotiq_2f_85 | sysid-branch fr3_nomesh |
|---|---|---|---|---|
| Arm masses | FR3 17.89 kg | Panda, 18.1 kg | Panda, 17.5 kg | FR3 exact |
| Gripper | 0.925 kg | ~0.9 kg + 7 kg fallback | ~0.06 kg authored, rest runtime-derived | none (bare arm) |
| Armature | 0.195/0.074 | 0 | 0 | 0 |
| Friction | 1.14...0.25 Nm | 0 (training injected 1.0) | 0 | 0.2 flat (importer default) |
| Joint limits | FR3 | Panda | Panda | FR3 |

Structural fabrics-vs-Arena diff: identical arm kinematics; different gripper
mount orientation (fabrics carries Karl Van Wyk's lab-verified mount; Arena
uses the classic -45 deg hand convention, unverified), Arena leaves the four
inner finger joints unlimited (+-180 deg), has no world-fixing root joint and
no fabric control frames, and authors kp=1e7/kd=1e5 placeholder drives. The
Arena USD is content-identical to the PolaRiS-Hub `nvidia_droid` original.
Decision: the fabrics USD stays the base asset (verified geometry, fabric
integration, = training plant); its dynamics are corrected via an overlay.

## 4. The fix

Applied as a thin USD sublayer over the unchanged fabrics asset + config
values (ported to dextrah training in nvblox_next PR #337):

- author 0.1 g on the 7 virtual frames (suppresses the 1 kg fallbacks),
- FR3 link masses and FR3 joint limits (franka_description identified),
- menagerie FR3 armature on the arm actuators (0.195 / 0.074),
- menagerie per-joint friction in DOF_FRICTION (1.137x4, 0.763, 0.44, 0.248),
- arm PD gains = the deployed FrankaDriver profile.

Sysid task variants: `Isaac-Sysid-Droid-v0` (as-is plant),
`Isaac-Sysid-Droid-Corrected-v0` (mass/armature/friction),
`Isaac-Sysid-Droid-FR3-v0` (+ FR3 masses and limits - the plant matching the
training port). Applied values verified in the live training stack
(19.87 kg total, correct gains/armature/limits, 1.12 mrad home-hold drift).

## 5. Validation: replay at the REAL recorded gains, no fitting

franka_robot_high preset (the deployment controller), RMSE vs the real trajectory:

| joint | rig kp/kd | as-is asset | corrected FR3 asset | factor |
|---|---|---|---|---|
| j1 | 400/20 | 14.90 mrad | 2.00 mrad | 7.4x |
| j2 | 400/20 | 7.02 | 3.18 | 2.2x |
| j3 | 400/20 | 10.22 | 2.53 | 4.0x |
| j4 | 400/20 | 25.60 | 2.34 | 10.9x |
| j5 | 175/6.5 | 73.11 | 4.46 | 16.4x |
| j6 | 100/6.5 | 21.06 | 5.12 | 4.1x |
| j7 | 35/3.5 | 29.30 | 11.57 | 2.5x |

Mean 25.9 -> 4.5 mrad. Improvement scales with proximity to the phantom TCP
mass (wrist worst-affected), confirming the causal story. The Panda->FR3 mass
swap alone is dynamically neutral at these excitations (within noise vs the
corrected-Panda variant); it is kept for the joint-limit feasibility fix.
Figure: `droid_asset_compare_fr3.png` (also `droid_asset_compare.png`).

Generalization at soft gains (franka_robot_low recordings, kp=80/kd=4 - 5-10x
outside the regime the corrections were derived at, no re-tuning):

| joint | as-is | corrected FR3 |
|---|---|---|
| j1 | 43.8 | 15.7 |
| j2 | 27.0 | 22.7 |
| j3 | 50.2 | 20.5 |
| j4 | 72.1 | 18.4 |
| j5 | 57.7 | 10.5 |
| j6 | 19.3 | 7.2 |
| j7 | 16.8 | 5.5 |

Mean 41.0 -> 14.4 mrad: better on every joint. Residuals grow at soft gains
because plant error dominates there; around each joint's resonance the
corrected sim overshoots the real amplitude - the real joints carry
amplitude/velocity-dependent damping that a constant friction torque cannot
capture. Figure: `droid_asset_compare_low.png`.

## 6. Identified PD parameters (overnight fits on the corrected FR3 plant)

Per-joint {kp, kd} CMA-ES fits (12 generations, 64 candidates, warm-started
at each preset's rig gains), one batch per preset. Values = rerolled CMA
mean. RMSE = replay at rig gains -> at fitted gains.

### franka_robot_high (rig 400x4 / 175 / 100 / 35, kd 20x4 / 6.5 / 6.5 / 3.5)

| joint | fitted kp | fitted kd | rig kp/kd | RMSE [mrad] |
|---|---|---|---|---|
| j1 | 413.6 | 22.0 | 400/20 | 2.00 -> 1.30 |
| j2 | 396.9 | 23.3 | 400/20 | 3.18 -> 2.07 |
| j3 | 413.2 | 22.1 | 400/20 | 2.53 -> 2.09 |
| j4 | 423.4 | 22.3 | 400/20 | 2.34 -> 1.95 |
| j5 | 298.9 | 12.4 | 175/6.5 | 4.46 -> 2.86 |
| j6 | 139.7 | 11.5 | 100/6.5 | 5.12 -> 2.13 |
| j7 | 73.4 | 9.0 | 35/3.5 | 11.57 -> 3.83 |

### franka_robot_low (rig 80 / 4 flat)

| joint | fitted kp | fitted kd | RMSE [mrad] |
|---|---|---|---|
| j1 | 81.7 | 6.4 | 15.67 -> 7.74 |
| j2 | 82.1 | 9.8 | 22.71 -> 7.99 |
| j3 | 83.8 | 6.6 | 20.47 -> 10.32 |
| j4 | 86.0 | 6.0 | 18.37 -> 15.11 |
| j5 | 120.5 | 7.6 | 10.54 -> 5.74 |
| j6 | 91.4 | 6.5 | 7.21 -> 3.42 |
| j7 | 135.5 | 7.7 | 5.54 -> 2.51 |

### franka_ros (rig 600x4 / 250 / 150 / 50, kd 30x4 / 10 / 10 / 5)

| joint | fitted kp | fitted kd | RMSE [mrad] |
|---|---|---|---|
| j1 | 664.3 | 34.2 | 1.17 -> 0.87 |
| j2 | 615.4 | 34.3 | 1.89 -> 1.30 |
| j3 | 644.3 | 33.2 | 1.50 -> 1.34 |
| j4 | 656.4 | 33.6 | 1.42 -> 1.26 |
| j5 | 542.1 | 23.0 | 2.92 -> 1.89 |
| j6 | 228.9 | 17.8 | 3.36 -> 1.52 |
| j7 | 116.9 | 13.4 | 8.01 -> 2.76 |

### Cross-preset reading

- **j1-j4: fitted == rig gains within 3-10% on all three presets.** The
  corrected plant is essentially exact for the shoulder; there is nothing
  left for the optimizer to absorb.
- **j5-j7: consistent controller-independent residual.** Fitted gains land
  1.4-2.3x stiffer and 1.8-2.7x more damped than the rig, same direction and
  similar magnitude across all presets: a genuine remaining wrist plant error
  (friction/damping character, gripper inertia detail), not fit noise. The
  identified wrist gains compensate it effectively (j7 at the high preset:
  11.6 -> 3.8 mrad).
- The stiffer the real controller, the smaller the baseline error
  (franka_ros baseline already 0.9-8.0 mrad): stiff PD masks plant error.

Figures: `droid_fr3_fit_<preset>_before_after.png` and
`droid_fr3_fit_<preset>_generations_{gains,trajectories}.png`.

## 7. Recommendations

1. **Training gains (franka_robot_high deployment)**: keep j1-j4 at the real
   400/20 (fitted equals rig within noise); use the identified wrist values
   **j5 = 299/12.4, j6 = 140/11.5, j7 = 73/9.0**. Candidate amendment to
   nvblox_next PR #337.
2. **Upstream the phantom-mass fix to fabrics-sim** - it affects every
   consumer of `droid_robotiq.usd`.
3. Retrain with the corrected plant and center domain randomization on the
   identified values instead of compensating a fixed bias with wide DR.
4. If wrist fidelity beyond ~3 mrad or soft-gain fidelity matters: identify
   velocity-dependent damping (the soft-gain resonance overshoot is the
   evidence), and collect richer excitation (higher frequencies, multiple
   poses, hold segments).
5. Flag to the Arena team: their droid USD (= PolaRiS original) has
   unauthored gripper masses, no armature/friction, placeholder 1e7 drives,
   unlimited finger joints, and an unverified mount.

## 8. Reproduction

Runbook: `README_DROID.md` next to this file (container setup, dataset
preparation, fit/plot commands - all tested). Upstream-relevant pipeline
fixes made along the way (int32 joint ids for the PhysX path, deferred
tensorboard import at kit boot, optional debug_draw GUI dependency,
per-generation artifact saving, joint-name mapping) are local uncommitted
changes on the `franka-sysid` checkout - candidates for Mike's branch.
