# Real-robot command tracking per PD preset (controller quality, not sim2real)

**Investigation 1 of 3** (2 = asset corrections, 3 = PD identification:
see SIM2REAL_FINDINGS.md). Summary figure:
`sysid_plots/summary_1_controller_tracking.png`.

How well does each FrankaDriver PD preset make the REAL arm follow its
commands? Pure recording analysis (measured vs commanded, no simulation),
same per-joint chirp excitation (0.3-2 Hz log sweep, +-0.1 rad, 20 s) for all
presets, so the comparison is fair.

Metrics per joint (see `analyze_robot_tracking.py`):

- **rmse**: raw measured-vs-commanded error [mrad]
- **lag**: cross-correlation delay of the response behind the command [ms]
- **rmse-lag**: error after removing the pure delay - what remains is
  amplitude/shape error
- **amp lo/mid/hi**: measured/commanded amplitude ratio in the first, middle
  and last third of the sweep (low, mid, high frequency band). Above 1 =
  resonant overshoot, below 1 = the controller no longer keeps up.
- **overshoot**: peak measured excursion beyond the peak commanded excursion

## franka_robot_high (kp 400x4/175/100/35, kd 20x4/6.5/6.5/3.5)

| joint | kp/kd | rmse | lag | rmse-lag | amp lo | amp mid | amp hi | overshoot |
|---|---|---|---|---|---|---|---|---|
| j1 | 400/20 | 27.0 | 59 ms | 1.5 | 0.98 | 1.01 | 1.02 | 2.8% |
| j2 | 400/20 | 17.4 | 76 ms | 4.6 | 1.00 | 1.05 | 1.18 | 27.3% |
| j3 | 400/20 | 23.2 | 63 ms | 4.3 | 1.00 | 1.03 | 1.11 | 18.3% |
| j4 | 400/20 | 24.8 | 63 ms | 4.5 | 1.01 | 1.04 | 1.09 | 14.7% |
| j5 | 175/6.5 | 31.4 | 43 ms | 3.8 | 0.97 | 0.97 | 0.97 | -4.1% |
| j6 | 100/6.5 | 25.2 | 78 ms | 5.3 | 0.94 | 0.93 | 0.86 | -5.7% |
| j7 | 35/3.5 | 71.6 | 103 ms | 24.6 | 0.88 | 0.81 | 0.70 | -13.3% |

## franka_robot_low (kp 80, kd 4, flat) = IsaacLab FRANKA_PANDA_CFG

This preset is numerically identical to IsaacLab's default
FRANKA_PANDA_CFG arm gains (franka.py), so this table doubles as a
REAL-robot measurement of the IsaacLab default convention.

| joint | rmse | lag | rmse-lag | amp lo | amp mid | amp hi | overshoot |
|---|---|---|---|---|---|---|---|
| j1 | 55.8 | 128 ms | 12.0 | 0.99 | 1.16 | 1.01 | 24.0% |
| j2 | 40.8 | 226 ms | 12.4 | 1.07 | 1.32 | 0.84 | 42.0% |
| j3 | 63.4 | 175 ms | 24.8 | 1.08 | 1.35 | 1.28 | 67.7% |
| j4 | 62.3 | 153 ms | 20.5 | 1.13 | 1.35 | 1.16 | 48.9% |
| j5 | 45.6 | 63 ms | 7.2 | 0.94 | 0.95 | 0.95 | -6.7% |
| j6 | 24.0 | 73 ms | 3.9 | 0.94 | 0.96 | 0.94 | -7.9% |
| j7 | 43.1 | 57 ms | 6.7 | 0.97 | 0.95 | 0.92 | -4.8% |

## franka_ros (kp 600x4/250/150/50, kd 30x4/10/10/5)

| joint | kp/kd | rmse | lag | rmse-lag | amp lo | amp mid | amp hi | overshoot |
|---|---|---|---|---|---|---|---|---|
| j1 | 600/30 | 24.3 | 54 ms | 1.2 | 0.98 | 1.00 | 0.98 | -1.6% |
| j2 | 600/30 | 14.0 | 64 ms | 2.3 | 1.00 | 1.02 | 1.09 | 12.7% |
| j3 | 600/30 | 20.1 | 57 ms | 2.0 | 0.99 | 1.01 | 1.04 | 6.6% |
| j4 | 600/30 | 21.7 | 57 ms | 1.9 | 1.00 | 1.01 | 1.03 | 5.4% |
| j5 | 250/10 | 31.3 | 43 ms | 4.3 | 0.98 | 0.97 | 0.95 | -3.3% |
| j6 | 150/10 | 23.6 | 73 ms | 5.4 | 0.94 | 0.92 | 0.85 | -5.6% |
| j7 | 50/5 | 68.0 | 97 ms | 23.8 | 0.89 | 0.82 | 0.70 | -11.3% |

## Reading

1. **Ranking for tracking quality: franka_ros > franka_robot_high >>
   franka_robot_low.** The ros preset has the lowest lag, the smallest
   delay-compensated error (1.2-2.3 mrad on j1-j4) and overshoot at most
   12.7%. The high preset is close but resonates more at the 2 Hz end
   (j2 hits 27% overshoot, amp-hi 1.18). The low preset is unusable as a
   tracking controller: up to 226 ms lag and 68% overshoot with a clear
   mid-sweep resonance (amp-mid up to 1.35, underdamped at kd=4).
2. **Raw RMSE is dominated by phase lag, not amplitude error.** Removing the
   delay cuts the error 5-20x (e.g. j1 high: 27.0 -> 1.5 mrad). The lags are
   closed-loop DYNAMIC lag, not transport latency: the corrected simulation
   with the same gains reproduces the measured trajectories to 2-5 mrad
   (SIM2REAL_FINDINGS.md), which a fixed command delay would not allow.
   Lag shrinks with stiffness (j1: 128 -> 59 -> 54 ms across presets) and
   saturates around 43-55 ms at the stiff end.
3. **The wrist is the bottleneck in every preset.** j7 attenuates to ~0.70 of
   the commanded amplitude by 2 Hz with ~100 ms lag at both 35/3.5 and 50/5,
   and j6 to ~0.85. The presets differ mainly in the proximal joints; nobody
   ever stiffened the wrist. The sysid fits point the same direction (the
   identified sim-side wrist gains are 1.4-2.3x the rig values).
4. **Suggested next controller iteration** (if tracking is the objective and
   vibration/compliance limits allow): start from franka_ros for j1-j4, and
   raise the wrist toward j5 ~300/12, j6 ~150-200/12, j7 ~70-100/8 - i.e.
   the region the sysid fits identified as dynamically appropriate for this
   arm+gripper. Validate with the same chirp protocol (collect, then rerun
   `analyze_robot_tracking.py`); watch overshoot at the 2 Hz end and any
   audible vibration as the stop criteria. Keep in mind stiffer tracking
   trades away compliance in contact - for grasping tasks the "best" tracker
   is not automatically the best manipulation controller.

## Reproduction

```bash
python scripts/sysid/analyze_robot_tracking.py \
  --data_root logs/sysid/prepared_datasets
```

Works on any future collection prepared with
`prepare_robot_control_dataset.py` - new presets show up as new tables.

## IsaacLab high-PD 400/80 (SIM ESTIMATE - never run on the robot)

No real recording exists for IsaacLab's FRANKA_PANDA_HIGH_PD_CFG convention
(uniform kp=400, kd=80 - what dextrah training used before the plant fix), so
its tracking is estimated by replaying the same chirp commands through the
corrected FR3 plant.

Estimator calibration: the same sim-based analysis at the RECORDED gains
reproduces the real-robot tables closely in the stiff regime (franka_robot_high
j1: rmse 26.0 sim vs 27.0 real, lag 56 vs 59 ms; j7: 67.6 vs 71.6 mrad, 95 vs
103 ms), with a mild tendency to over-predict resonance (j2 overshoot 34% est
vs 27% real). At soft gains (80/4) it exaggerates resonance strongly (the
constant-friction damping limit) - so soft-gain estimates are qualitative
only. 400/80 is stiff and heavily damped, squarely in the trustworthy regime.

| joint | kp/kd | rmse | lag | rmse-lag | amp lo | amp mid | amp hi | overshoot |
|---|---|---|---|---|---|---|---|---|
| j1 | 400/80 | 51.5 | 140 ms | 25.1 | 0.87 | 0.75 | 0.50 | -8.3% |
| j2 | 400/80 | 26.6 | 156 ms | 10.9 | 0.89 | 0.76 | 0.56 | -8.3% |
| j3 | 400/80 | 42.8 | 149 ms | 19.0 | 0.85 | 0.73 | 0.54 | -10.5% |
| j4 | 400/80 | 48.3 | 153 ms | 20.8 | 0.86 | 0.72 | 0.56 | -11.8% |
| j5 | 400/80 | 81.3 | 135 ms | 42.3 | 0.80 | 0.66 | 0.48 | -15.3% |
| j6 | 400/80 | 38.2 | 137 ms | 19.6 | 0.79 | 0.65 | 0.48 | -17.0% |
| j7 | 400/80 | 84.1 | 131 ms | 45.4 | 0.77 | 0.61 | 0.46 | -18.1% |

Verdict: kd=80 (4-20x any real preset) makes the arm heavily OVERDAMPED.
Zero overshoot, but 131-156 ms lag on every joint (~2.5x the real presets)
and the response collapses to ~0.5x the commanded amplitude by 2 Hz. As a
robot controller it would be the worst tracker of the four by a wide margin
(and this is the optimistic estimate - it assumes the plant corrections).
For training, this is what the policy learned against pre-fix: an arm that
smoothly executes about half of every fast command - nothing the real robot
does under any preset.

Note: producing this table exposed a stateful-gains bug in
`plot_droid_fit.py` (the "asset default" was read back from the live
articulation, which each run overwrites - runs after the first got the
previous dataset's gains). Fixed by capturing the asset defaults before the
first replay. No previously reported result relied on the affected asset-gain
caches (all published comparisons used recorded gains, which come from the
dataset).

## Conclusion: which controller is best

Ranking by command-tracking quality on this arm+gripper (chirps 0.3-2 Hz,
mean values over the 7 joints):

| rank | controller | mean rmse | lag range | character |
|---|---|---|---|---|
| 1 | **franka_ros** (600x4/250/150/50, 30x4/10/10/5) | 29 mrad | 43-97 ms | best proximal tracking, minimal overshoot (<=13%), wrist still soft |
| 2 | franka_robot_high (400x4/175/100/35, 20x4/6.5x2/3.5) | 31 mrad | 43-103 ms | close second, resonates more at 2 Hz (j2 +27%) |
| 3 | franka_robot_low = IsaacLab FRANKA_PANDA_CFG (80/4 flat) | 48 mrad | 57-226 ms | underdamped, up to 68% overshoot - unusable for tracking (real measurement) |
| 4 | IsaacLab high-PD 400/80 (sim estimate) | 53 mrad | 131-156 ms | overdamped, ~0.5x amplitude at 2 Hz - worst of all |

- **Best existing preset: franka_ros.** Lowest lag, lowest residual error,
  no meaningful overshoot on any joint. If switching the deployed controller
  is on the table, this is the drop-in winner - with the usual caveat that
  stiffer gains carry more contact force (grasping compliance).
- **The IsaacLab 400/80 convention ranks LAST as a robot controller** despite
  its name: the damping is 4-20x any real preset, trading a little overshoot
  for 2.5x the lag and half the commanded amplitude at 2 Hz. It should not be
  used on the robot, and (post plant fix) no longer represents anything in
  training either. Together with rank 3 (franka_robot_low IS IsaacLab's
  default FRANKA_PANDA_CFG, measured on the real arm), BOTH IsaacLab
  franka.py conventions occupy the bottom two ranks: the default is
  underdamped, the "high PD" overdamped - neither matches any gain set a
  real Franka driver ships.
- **Every preset, including the winner, under-serves the wrist** (j7 at
  ~0.70x amplitude, ~100 ms lag). The best controller for this robot does
  not exist as a preset yet: franka_ros proximal gains + a stiffened wrist
  (j5 ~300/12, j6 ~150-200/12, j7 ~70-100/8, per the sysid fits). Collect
  chirps with that candidate and rerun this analysis to confirm.
