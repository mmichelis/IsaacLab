# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 forward-running locomotion with soft deformable pads under the feet.

This mirrors the flat-terrain G1 velocity task (:class:`G1FlatEnvCfg`) but:

* swaps the rigid Newton solver for the coupled MJWarp + VBD solver,
* attaches one volume-deformable cuboid "shoe sole" beneath each ankle-roll foot,
  pinned to the foot by :class:`SoftPadJointPositionActionWithSoftPads`,
* commands a forward velocity so the policy learns to run as fast as it can.

The pads are stiff (Young's modulus ~1e6 Pa) cuboids ~6 cm thick that deform on
ground contact and feed a cushioned reaction back into the feet through two-way
soft contact.
"""

from __future__ import annotations

from isaaclab_newton.physics import MJWarpSolverCfg
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import NewtonDeformableBodyMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.assets.deformable_object import DeformableObjectCfg
from isaaclab.envs.mdp import bad_orientation, root_height_below_minimum
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.deformable.newton_manager_cfg import (
    CoupledMJWarpVBDSolverCfg,
    CoupledNewtonCfg,
    NewtonModelCfg,
    VBDSolverCfg,
)

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import MySceneCfg

from .mdp import SoftPadJointPositionActionCfg

##
# Soft-pad geometry and material.
##

# Pad cuboid size [m]: (length along foot x, width along foot y, thickness along z).
_PAD_SIZE = (0.18, 0.09, 0.06)

# Stiff-but-compliant sole: Young's modulus [Pa] and Poisson's ratio [-], in the requested range.
_PAD_YOUNGS_MODULUS = 1.0e6
_PAD_POISSONS_RATIO = 0.3
_PAD_DENSITY = 200.0  # kg/m^3 (foam/rubber-like)

# Foot-origin (ankle_roll_link) sits ~6.1 cm above the sole (measured on stock G1). Place the pad
# so its top sits slightly INTO the foot (overlap) — this guarantees foot<->pad contact (support)
# and removes the visible gap between shoe and sole.
_FOOT_ORIGIN_TO_SOLE = 0.061
_PAD_FOOT_OVERLAP = 0.015  # pad top this far above the sole, i.e. inside the foot
_PAD_Z_OFFSET = -(_FOOT_ORIGIN_TO_SOLE - _PAD_FOOT_OVERLAP + _PAD_SIZE[2] / 2.0)

# Lame parameters from (E, nu).
_PAD_K_MU = _PAD_YOUNGS_MODULUS / (2.0 * (1.0 + _PAD_POISSONS_RATIO))
_PAD_K_LAMBDA = (
    _PAD_YOUNGS_MODULUS * _PAD_POISSONS_RATIO / ((1.0 + _PAD_POISSONS_RATIO) * (1.0 - 2.0 * _PAD_POISSONS_RATIO))
)


def _foot_pad_cfg(prim_name: str, init_y: float) -> DeformableObjectCfg:
    """A soft cuboid pad spawned near a foot; the action term re-anchors it on reset."""
    return DeformableObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, init_y, _PAD_SIZE[2] / 2.0)),
        spawn=sim_utils.MeshCuboidCfg(
            size=_PAD_SIZE,
            deformable_props=NewtonDeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.12)),
            physics_material=NewtonDeformableBodyMaterialCfg(
                density=_PAD_DENSITY,
                k_mu=_PAD_K_MU,
                k_lambda=_PAD_K_LAMBDA,
                particle_radius=0.01,
            ),
        ),
    )


@configclass
class G1SoftPadSceneCfg(MySceneCfg):
    """Flat-terrain G1 scene with a soft deformable pad under each foot."""

    left_foot_pad: DeformableObjectCfg = _foot_pad_cfg("LeftFootPad", init_y=0.1)
    right_foot_pad: DeformableObjectCfg = _foot_pad_cfg("RightFootPad", init_y=-0.1)


@configclass
class G1SoftPadEnvCfg(G1FlatEnvCfg):
    """G1 running on soft foot pads, coupled MJWarp (robot) + VBD (pads)."""

    scene: G1SoftPadSceneCfg = G1SoftPadSceneCfg(num_envs=128, env_spacing=2.5, replicate_physics=True)

    def __post_init__(self) -> None:
        super().__post_init__()

        # -- Coupled MJWarp + VBD solver (robot rigid in MJWarp, pads soft in VBD, two-way).
        self.sim.physics = CoupledNewtonCfg(
            solver_cfg=CoupledMJWarpVBDSolverCfg(
                rigid_solver_cfg=MJWarpSolverCfg(
                    njmax=120,
                    nconmax=40,
                    ls_iterations=20,
                    cone="pyramidal",
                    impratio=1,
                    ls_parallel=False,
                    integrator="implicitfast",
                ),
                soft_solver_cfg=VBDSolverCfg(
                    iterations=20,
                    integrate_with_external_rigid_solver=True,
                    particle_enable_self_contact=False,
                    particle_collision_detection_interval=-1,
                ),
                coupling_mode="two_way",
            ),
            model_cfg=NewtonModelCfg(
                # Foot <-> pad and pad <-> ground contact: firm enough to carry the robot and keep
                # the pad resting on the ground (minimal clipping), high friction so the soles grip.
                soft_contact_ke=4.0e4,
                soft_contact_kd=1.0e-3,
                soft_contact_mu=2.0,
                shape_material_ke=4.0e4,
                shape_material_kd=1.0e-3,
                shape_material_mu=2.0,
            ),
            num_substeps=10,
        )

        # Locomotion sim cadence: 200 Hz sim, 50 Hz policy (matches the base velocity env).
        self.sim.dt = 0.005
        self.decimation = 4
        self.sim.render_interval = self.decimation

        # Lift the spawn so the added pad thickness clears the ground at reset.
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.74 + _PAD_SIZE[2])

        # The coupled MJWarp+VBD manager does not support contact sensors (its dummy solver
        # raises on ``update_contacts``), and with a pad between foot and ground the foot would
        # not register ground contact anyway. Drop the contact sensor and the contact-based
        # reward/termination terms, replacing fall detection with orientation + root height.
        self.scene.contact_forces = None
        self.rewards.feet_air_time = None
        self.rewards.feet_slide = None
        self.terminations.base_contact = DoneTerm(func=bad_orientation, params={"limit_angle": 1.0})
        self.terminations.root_too_low = DoneTerm(func=root_height_below_minimum, params={"minimum_height": 0.3})

        # -- Actions: joint position control plus per-foot soft-pad pinning.
        self.actions.joint_pos = SoftPadJointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=0.5,
            use_default_offset=True,
            pad_foot_pairs=[
                ("left_foot_pad", "left_ankle_roll_link"),
                ("right_foot_pad", "right_ankle_roll_link"),
            ],
            pad_z_offset=_PAD_Z_OFFSET,
            pin_fraction=0.15,
        )

        # -- Command a forward run; let the policy push toward maximal speed.
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # Reduce reset-time velocity kicks that can destabilize the soft contact at episode start.
        self.events.push_robot = None


@configclass
class G1SoftPadEnvCfg_PLAY(G1SoftPadEnvCfg):
    """Smaller, deterministic config for visualization / evaluation."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 16
        self.episode_length_s = 20.0
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
