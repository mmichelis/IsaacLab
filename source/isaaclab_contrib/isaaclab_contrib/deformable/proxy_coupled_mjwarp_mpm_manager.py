# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Proxy-coupled MJWarp + implicit MPM Newton manager.

Wraps :class:`newton.solvers.experimental.coupled.SolverCoupledProxy` with
MuJoCo Warp as the rigid sub-solver and implicit MPM as the particle
sub-solver, exposing selected MuJoCo bodies as proxies in the MPM view.
"""

from __future__ import annotations

import logging

from isaaclab_newton.physics.newton_manager import NewtonManager
from newton import CollisionPipeline, Model
from newton.solvers import SolverImplicitMPM, SolverMuJoCo
from newton.solvers.experimental.coupled import SolverCoupledProxy

from isaaclab.physics import PhysicsManager

from ._proxy_partition import partition_model_by_entities, select_proxy_bodies
from .newton_manager_cfg import CoupledNewtonCfg, ProxyCoupledMJWarpMPMSolverCfg

logger = logging.getLogger(__name__)

_CFG_LABEL = "ProxyCoupledMJWarpMPMSolverCfg"


class NewtonProxyCoupledMJWarpMPMManager(NewtonManager):
    """Newton manager wrapping :class:`newton.solvers.experimental.coupled.SolverCoupledProxy` with an MJWarp+MPM split.

    Bodies/joints/shapes are partitioned between the two entries; all particles
    are solved by implicit MPM.
    """

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: ProxyCoupledMJWarpMPMSolverCfg) -> None:
        mjc_kw = cls._filter_solver_kwargs(SolverMuJoCo, solver_cfg.mjwarp_cfg)
        mpm_cfg_kw = cls._filter_solver_kwargs(SolverImplicitMPM.Config, solver_cfg.mpm_cfg)
        mpm_config = SolverImplicitMPM.Config(**mpm_cfg_kw)

        outer_cfg = PhysicsManager._cfg
        scene_cfg = outer_cfg.scene_cfg if isinstance(outer_cfg, CoupledNewtonCfg) else None

        mjc_bodies, mpm_bodies, mjc_joints, mpm_joints, mjc_shapes, mpm_shapes = partition_model_by_entities(
            model,
            solver_cfg.mjwarp_bodies,
            solver_cfg.mpm_bodies,
            scene_cfg,
            cfg_label=_CFG_LABEL,
            entry_a_field="mjwarp_bodies",
            entry_b_field="mpm_bodies",
        )
        mpm_particles = list(range(int(model.particle_count)))

        proxy_body_ids = select_proxy_bodies(
            model, solver_cfg.proxy_bodies, scene_cfg, cfg_label=_CFG_LABEL
        )
        if solver_cfg.proxy_bodies and not proxy_body_ids:
            logger.warning(
                "ProxyCoupledMJWarpMPMSolverCfg.proxy_bodies=%s matched no bodies with COLLIDE_SHAPES. "
                "Rigid bodies will not be visible to MPM.",
                solver_cfg.proxy_bodies,
            )

        entries = [
            SolverCoupledProxy.Entry(
                name="mjc",
                solver=lambda v, _kw=mjc_kw: SolverMuJoCo(model=v, **_kw),
                bodies=mjc_bodies,
                joints=mjc_joints,
                shapes=mjc_shapes,
            ),
            SolverCoupledProxy.Entry(
                name="mpm",
                solver=lambda v, _cfg=mpm_config: SolverImplicitMPM(model=v, config=_cfg),
                bodies=mpm_bodies,
                joints=mpm_joints,
                particles=mpm_particles,
                shapes=mpm_shapes,
            ),
        ]

        proxies: list[SolverCoupledProxy.Proxy] = []
        if proxy_body_ids:
            proxies.append(
                SolverCoupledProxy.Proxy(
                    source="mjc",
                    destination="mpm",
                    bodies=proxy_body_ids,
                    mode=solver_cfg.proxy_mode,
                    mass_scale=float(solver_cfg.proxy_mass_scale),
                    collision_pipeline=lambda destination_model: CollisionPipeline(
                        destination_model,
                        broad_phase="explicit",
                    ),
                    collide_interval=int(solver_cfg.proxy_collide_interval),
                )
            )

        NewtonManager._solver = SolverCoupledProxy(
            model=model,
            entries=entries,
            coupling=SolverCoupledProxy.Config(
                proxies=proxies,
                iterations=int(solver_cfg.proxy_iterations),
            ),
        )
        NewtonManager._use_single_state = False
        # When the MJWarp entry runs without MuJoCo's internal contacts,
        # NewtonManager must populate ``contacts`` from a collision pipeline
        # before stepping so SolverMuJoCo can consume them.
        NewtonManager._needs_collision_pipeline = not bool(solver_cfg.mjwarp_cfg.use_mujoco_contacts)

    @classmethod
    def get_entry_solver(cls, name: str):
        """Return the wrapped sub-solver instance for ``name`` (e.g. ``"mjc"``, ``"mpm"``).

        Args:
            name: Entry name passed to :class:`SolverCoupledProxy.Entry`
                during :meth:`_build_solver` (``"mjc"`` or ``"mpm"``).

        Returns:
            The :class:`newton.solvers.SolverBase` instance for that entry.

        Raises:
            RuntimeError: Active solver is not a :class:`SolverCoupledProxy`.
            KeyError: No entry by that name.
        """
        coupled = NewtonManager._solver
        if not isinstance(coupled, SolverCoupledProxy):
            raise RuntimeError("Active solver is not SolverCoupledProxy.")
        return coupled.solver(name)
