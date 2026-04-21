# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class FrankaCableReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 50
    experiment_name = "franka_cable_reach"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        # Running-mean/std normalization — the obs vector mixes joint angles (rad),
        # clamped velocities, unit quaternions, and meter-scale positions, and will
        # swamp first-layer activations without per-dim normalization.
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # Raised from 0.006 — previous run collapsed ``Policy/mean_std`` from 1.0
        # to 0.13 in 1650 iters, locking the policy onto the "reach and hold" local
        # optimum. More entropy pressure keeps exploration alive for the reward
        # reshuffle to take effect.
        entropy_coef=0.015,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
