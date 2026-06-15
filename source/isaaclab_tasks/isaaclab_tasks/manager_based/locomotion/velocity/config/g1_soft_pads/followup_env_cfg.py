# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Follow-up controls for the rigid-vs-soft-pad efficiency study.

The first study found the apparent soft-pad efficiency "win" was dominated by the rigid policy
buzzing its 14 finger joints (~65-76% of its joint power), with residual confounds from the
coupled solver / kinematic pin. These variants enable cleaner controls:

* ``*-Clean-*``  — adds a strong finger-joint velocity penalty so neither policy wastes power on
  the (non-locomotor) fingers; retrain both to compare leg work fairly.
* ``*-SoftStiff-*`` — the soft-pad env with a near-rigid pad (E=1e7) on the SAME coupled solver +
  pin, to test whether the soft env's low joint work is due to pad elasticity or to solver/pin.
"""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .rigid_stiff_env_cfg import G1FlatStiffEnvCfg, G1FlatStiffEnvCfg_PLAY
from .soft_pad_env_cfg import G1SoftPadEnvCfg, G1SoftPadEnvCfg_PLAY

# G1 dexterous-hand finger joints (the ones the rigid policy was buzzing).
_FINGER_JOINTS = [
    ".*_five_joint",
    ".*_three_joint",
    ".*_six_joint",
    ".*_four_joint",
    ".*_zero_joint",
    ".*_one_joint",
    ".*_two_joint",
]


def _add_finger_penalty(cfg) -> None:
    """Penalize finger-joint velocity (and tighten the deviation penalty) to stop the buzzing."""
    fingers = SceneEntityCfg("robot", joint_names=_FINGER_JOINTS)
    # Strong finger-velocity penalty: the buzzing is non-locomotor, so heavily suppress it.
    cfg.rewards.finger_vel_penalty = RewTerm(func=mdp.joint_vel_l2, weight=-2.0e-2, params={"asset_cfg": fingers})
    if getattr(cfg.rewards, "joint_deviation_fingers", None) is not None:
        cfg.rewards.joint_deviation_fingers.weight = -2.0


@configclass
class G1FlatStiffCleanEnvCfg(G1FlatStiffEnvCfg):
    """Rigid stiff env + finger penalty (clean efficiency baseline)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        _add_finger_penalty(self)


@configclass
class G1FlatStiffCleanEnvCfg_PLAY(G1FlatStiffEnvCfg_PLAY):
    def __post_init__(self) -> None:
        super().__post_init__()
        _add_finger_penalty(self)


@configclass
class G1SoftPadCleanEnvCfg(G1SoftPadEnvCfg):
    """Soft-pad env + finger penalty (clean efficiency comparison)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        _add_finger_penalty(self)


@configclass
class G1SoftPadCleanEnvCfg_PLAY(G1SoftPadEnvCfg_PLAY):
    def __post_init__(self) -> None:
        super().__post_init__()
        _add_finger_penalty(self)


# Stiffer-pad control: E = 4e6 Pa (4x the soft pad), same coupled solver + pin. (1e7 needs too many
# VBD substeps to stay stable/tractable; 4e6 keeps it stable at a modest substep bump.)
_STIFF_E = 4.0e6
_STIFF_NU = 0.3
_STIFF_K_MU = _STIFF_E / (2.0 * (1.0 + _STIFF_NU))
_STIFF_K_LAMBDA = _STIFF_E * _STIFF_NU / ((1.0 + _STIFF_NU) * (1.0 - 2.0 * _STIFF_NU))


def _stiffen_pads(cfg) -> None:
    """Raise pad Young's modulus to ~1e7 and add VBD substeps for stability; add finger penalty."""
    for name in ("left_foot_pad", "right_foot_pad"):
        mat = getattr(cfg.scene, name).spawn.physics_material
        mat.k_mu = _STIFF_K_MU
        mat.k_lambda = _STIFF_K_LAMBDA
    # Stiffer FEM needs a modest VBD bump for stability.
    cfg.sim.physics.num_substeps = 12
    cfg.sim.physics.solver_cfg.soft_solver_cfg.iterations = 20
    _add_finger_penalty(cfg)


@configclass
class G1SoftPadStiffEnvCfg(G1SoftPadEnvCfg):
    """Near-rigid pad (E=1e7) on the soft-pad coupled solver — solver/pin control."""

    def __post_init__(self) -> None:
        super().__post_init__()
        _stiffen_pads(self)


@configclass
class G1SoftPadStiffEnvCfg_PLAY(G1SoftPadEnvCfg_PLAY):
    def __post_init__(self) -> None:
        super().__post_init__()
        _stiffen_pads(self)
