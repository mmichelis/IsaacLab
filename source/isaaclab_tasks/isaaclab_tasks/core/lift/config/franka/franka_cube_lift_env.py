# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cube-lift environment with a guard against non-finite solver output."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.managers import RewardManager


class _FiniteRewardManager(RewardManager):
    """Reward manager that zeros any non-finite reward-term output.

    A rare diverged solve produces NaN/Inf sim state, which makes the affected terms return
    NaN. The base manager accumulates each raw term into the returned reward, the per-term
    episode-reward logs, and the live-viz step reward, so a single diverged step poisons all of
    them. Sanitizing each term value here -- the only point before it is accumulated -- keeps
    them finite.
    """

    def compute(self, dt: float) -> torch.Tensor:
        # Mirrors RewardManager.compute (keep in sync) with one added nan_to_num on each term.
        self._reward_buf[:] = 0.0
        for term_idx, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            if term_cfg.weight == 0.0:
                self._step_reward[:, term_idx] = 0.0
                continue
            value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
            value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            self._reward_buf += value
            self._episode_sums[name] += value
            self._step_reward[:, term_idx] = value / dt
        return self._reward_buf


class FrankaCubeLiftEnv(ManagerBasedRLEnv):
    """Cube-lift env that keeps NaN/Inf solver blow-ups from reaching the RL algorithm.

    The MJWarp (and coupled) solve can rarely diverge to NaN/Inf during gripper-cube contact. The
    ``body_velocity_out_of_bounds`` termination catches such envs (it tests finiteness of every
    state the observations are built from) and resets them, so their observations are recomputed
    clean. Two further guards keep the rare divergence from reaching the learner:
    :class:`_FiniteRewardManager` zeros non-finite reward terms at the source (covering the reward
    computed from the pre-reset state and the per-term episode-reward logs), and :meth:`step`
    zeros any non-finite observation that slips past the divergence reset.
    """

    def load_managers(self):
        super().load_managers()
        # Swap in the finite-guarded reward manager so a diverged step cannot leak NaN into the
        # returned reward or the per-term episode-reward logs (both written inside step()).
        self.reward_manager = _FiniteRewardManager(self.cfg.rewards, self)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        obs, reward, terminated, time_outs, extras = super().step(action)
        # Reward is sanitized in _FiniteRewardManager; guard observations here as a backstop in
        # case a divergence escapes the reset and would otherwise poison the policy/critic.
        for group in obs.values():
            torch.nan_to_num_(group, nan=0.0, posinf=0.0, neginf=0.0)
        return obs, reward, terminated, time_outs, extras
