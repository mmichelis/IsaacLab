# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Proxy-coupled MJWarp + VBD Newton manager.

Wraps :class:`newton.solvers.experimental.coupled.SolverCoupledProxy` with
MuJoCo Warp as the rigid sub-solver and VBD as the soft sub-solver, exposing
selected MuJoCo bodies as proxies in the VBD view.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from isaaclab_newton.physics.newton_manager import NewtonManager
from newton import CollisionPipeline, Model
from newton.solvers import SolverMuJoCo, SolverVBD
from newton.solvers.experimental.coupled import SolverCoupledProxy

from isaaclab.managers import SceneEntityCfg
from isaaclab.physics import PhysicsManager

from ._proxy_partition import (
    partition_model_by_entities,
    resolve_entity_to_body_ids,
    select_proxy_bodies,
)
from .newton_manager_cfg import CoupledNewtonCfg, ProxyCoupledMJWarpVBDSolverCfg
from .vbd_manager import NewtonVBDManager

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveSceneCfg

logger = logging.getLogger(__name__)

_CFG_LABEL = "ProxyCoupledMJWarpVBDSolverCfg"


class NewtonProxyCoupledMJWarpVBDManager(NewtonVBDManager):
    """Newton manager wrapping :class:`newton.solvers.experimental.coupled.SolverCoupledProxy` with an MJWarp+VBD split.

    Bodies/joints/shapes are partitioned between the two entries; all particles
    are solved by VBD.
    """

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: ProxyCoupledMJWarpVBDSolverCfg) -> None:
        mjc_kw = cls._filter_solver_kwargs(SolverMuJoCo, solver_cfg.mjwarp_cfg)
        vbd_kw = cls._filter_solver_kwargs(SolverVBD, solver_cfg.vbd_cfg)

        outer_cfg = PhysicsManager._cfg
        scene_cfg = outer_cfg.scene_cfg if isinstance(outer_cfg, CoupledNewtonCfg) else None

        mjc_bodies, vbd_bodies, mjc_joints, vbd_joints, mjc_shapes, vbd_shapes = partition_model_by_entities(
            model,
            solver_cfg.mjwarp_bodies,
            solver_cfg.vbd_bodies,
            scene_cfg,
            cfg_label=_CFG_LABEL,
            entry_a_field="mjwarp_bodies",
            entry_b_field="vbd_bodies",
        )
        vbd_particles = list(range(model.particle_count))

        proxy_body_ids = select_proxy_bodies(model, solver_cfg.proxy_bodies, scene_cfg, cfg_label=_CFG_LABEL)
        if solver_cfg.proxy_bodies and not proxy_body_ids:
            logger.warning(
                "ProxyCoupledMJWarpVBDSolverCfg.proxy_bodies=%s matched no bodies with COLLIDE_SHAPES. "
                "Rigid bodies will not be visible to VBD.",
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
                name="vbd",
                solver=lambda v, _kw=vbd_kw: SolverVBD(model=v, **_kw),
                bodies=vbd_bodies,
                joints=vbd_joints,
                particles=vbd_particles,
                shapes=vbd_shapes,
            ),
        ]

        proxies: list[SolverCoupledProxy.Proxy] = []
        if proxy_body_ids:
            proxies.append(
                SolverCoupledProxy.Proxy(
                    source="mjc",
                    destination="vbd",
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
        NewtonManager._needs_collision_pipeline = False

    # ------------------------------------------------------------------
    # Backwards-compatible shims that delegate to ``_proxy_partition``.
    # Existing tests call these directly.
    # ------------------------------------------------------------------

    @classmethod
    def _resolve_entity_to_body_ids(
        cls,
        model: Model,
        spec: SceneEntityCfg | str,
        scene_cfg: InteractiveSceneCfg | None,
        field: str,
    ) -> list[int]:
        return resolve_entity_to_body_ids(model, spec, scene_cfg, cfg_label=_CFG_LABEL, field=field)

    @classmethod
    def _partition_model_by_entities(
        cls,
        model: Model,
        mjwarp_bodies: list[SceneEntityCfg | str],
        vbd_bodies: list[SceneEntityCfg | str],
        scene_cfg: InteractiveSceneCfg | None,
    ) -> tuple[list[int], list[int], list[int], list[int], list[int], list[int]]:
        return partition_model_by_entities(
            model,
            mjwarp_bodies,
            vbd_bodies,
            scene_cfg,
            cfg_label=_CFG_LABEL,
            entry_a_field="mjwarp_bodies",
            entry_b_field="vbd_bodies",
        )

    @classmethod
    def _select_proxy_bodies(
        cls,
        model: Model,
        proxy_bodies: list[SceneEntityCfg | str],
        scene_cfg: InteractiveSceneCfg | None,
    ) -> list[int]:
        return select_proxy_bodies(model, proxy_bodies, scene_cfg, cfg_label=_CFG_LABEL)
