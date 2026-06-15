# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 flat-terrain rigid locomotion with a STIFFER Newton ground contact.

The stock Newton G1 preset (``num_substeps=1``, ``nconmax=10``) lets the feet penetrate the
ground by several cm at running footstrike. This variant hardens the rigid contact:

* ``num_substeps`` 1 -> 4 (finer contact resolution per control step), and
* larger contact buffers (``nconmax``/``njmax``) so foot contacts are not dropped.

It is otherwise the stock :class:`G1FlatEnvCfg` (rigid, no pads). Used to retrain a clean Newton
rigid baseline whose feet stay on the ground.
"""

from __future__ import annotations

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.sensors import ContactSensorCfg as NewtonContactSensorCfg

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg


@configclass
class G1FlatStiffEnvCfg(G1FlatEnvCfg):
    """Stock G1 flat env on a stiffer Newton MJWarp contact (kitless)."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Concrete Newton config (kitless, no preset needed). Stiffer/firmer ground contact than
        # the stock newton_mjwarp preset.
        self.sim.physics = NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=300,
                nconmax=80,
                iterations=100,
                ls_iterations=50,
                cone="pyramidal",
                impratio=1,
                integrator="implicitfast",
            ),
            num_substeps=4,
        )

        # Newton contact sensor (the rigid Newton manager supports it, so the foot air-time / slide
        # rewards work here, unlike the coupled soft-pad env).
        self.scene.contact_forces = NewtonContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
        )
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class G1FlatStiffEnvCfg_PLAY(G1FlatStiffEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
