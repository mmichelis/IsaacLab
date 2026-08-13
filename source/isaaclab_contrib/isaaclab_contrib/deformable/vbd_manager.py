# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""VBD Newton manager."""

from __future__ import annotations

from isaaclab_newton.physics.newton_manager import NewtonManager
from newton import Model, ModelBuilder
from newton.solvers import SolverVBD

from .newton_manager_cfg import NewtonModelSolverCfg, VBDSolverCfg


def _apply_model_cfg(model: Model) -> None:
    """Apply the active solver cfg's :class:`NewtonModelCfg` to the finalized model.

    Sets the model-global ``soft_contact_ke/kd/mu``. Per-shape material defaults
    are applied earlier via ``builder.default_shape_cfg``, not here.
    """
    from isaaclab.physics import PhysicsManager

    solver_cfg = getattr(PhysicsManager._cfg, "solver_cfg", None)
    if not isinstance(solver_cfg, NewtonModelSolverCfg) or solver_cfg.model_cfg is None:
        return

    model_cfg = solver_cfg.model_cfg
    model.soft_contact_ke = float(model_cfg.soft_contact_ke)
    model.soft_contact_kd = float(model_cfg.soft_contact_kd)
    model.soft_contact_mu = float(model_cfg.soft_contact_mu)


class NewtonVBDManager(NewtonManager):
    """:class:`NewtonManager` specialization for the VBD solver.

    Always uses Newton's :class:`CollisionPipeline` for contact handling.
    """

    @classmethod
    def _prepare_builder_for_finalize(cls, builder: ModelBuilder) -> None:
        """Color imported particles before VBD model finalization."""
        super()._prepare_builder_for_finalize(builder)
        builder.color()

    @classmethod
    def start_simulation(cls) -> None:
        """Start simulation by finalizing model and initializing state.

        This function finalizes the model and initializes the simulation state.
        Note: Collision pipeline is initialized later in initialize_solver() after
        we determine whether the solver needs external collision detection.
        """
        super().start_simulation()

        if cls._model is not None:
            _apply_model_cfg(cls._model)

    @classmethod
    def _create_solver(cls, model: Model, solver_cfg: VBDSolverCfg) -> SolverVBD:
        """Construct the configured VBD solver."""
        return SolverVBD(model, **cls._filter_solver_kwargs(SolverVBD, solver_cfg))

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: VBDSolverCfg) -> None:
        """Construct :class:`SolverVBD` and populate the base-class slots.

        VBD always uses Newton's :class:`CollisionPipeline` and steps with
        separate input/output states, so the flags are fixed.
        """
        NewtonManager._solver = cls._create_solver(model, solver_cfg)
        NewtonManager._use_single_state = False
        NewtonManager._needs_collision_pipeline = True
        NewtonManager._supports_rigid_body_force_input = not solver_cfg.integrate_with_external_rigid_solver

    @classmethod
    def _simulate_physics_only(cls) -> None:
        # Rebuild BVH once per step for solvers that require it (e.g. VBD cloth).
        if cls._model.particle_count > 0 and hasattr(cls._solver, "rebuild_bvh"):
            cls._solver.rebuild_bvh(cls._state_0)
        super()._simulate_physics_only()
