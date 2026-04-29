# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kamino Newton manager."""

from __future__ import annotations

import inspect
import logging

import warp as wp

from newton import Model
from newton.solvers import SolverKamino

from .newton_manager import NewtonManager
from .newton_manager_cfg import KaminoSolverCfg

from isaaclab.physics import PhysicsManager
from isaaclab.utils.timer import Timer

logger = logging.getLogger(__name__)


class NewtonKaminoManager(NewtonManager):
    """:class:`NewtonManager` specialization for the Kamino solver.

    Always uses Newton's :class:`CollisionPipeline` for contact handling.
    """

    @classmethod
    def step(cls) -> None:
        """Step the physics simulation."""
        sim = PhysicsManager._sim
        if sim is None or not sim.is_playing():
            return

        # Kamino: run solver.reset() with the accumulated world mask to reinitialise
        # internal state (warm-start containers, constraint multipliers) for reset worlds.
        # Note: runs every step. solver.reset() with an all-False world_mask is a no-op
        # (kernels check mask per-world and skip). The cost of a no-op launch is negligible
        # compared to the complexity of maintaining a separate boolean guard.
        cls._forward_kamino(world_mask=cls._world_reset_mask)
        
        # Continue normal stepping
        super().step()


    @classmethod
    def _build_solver(
        cls, model: Model, solver_cfg: KaminoSolverCfg
    ) -> tuple[SolverKamino, bool, bool]:
        """Construct :class:`SolverKamino` from *solver_cfg*.

        Returns ``(solver, use_single_state=False, needs_collision_pipeline)``
        where the pipeline flag is ``True`` only when
        ``use_collision_detector=False``.
        """
        return SolverKamino(model, solver_cfg.to_solver_config()), False, not solver_cfg.use_collision_detector


    @classmethod
    def _capture_or_defer_cuda_graph(cls) -> None:
        """Capture the physics CUDA graph, or defer if RTX is initializing."""
        cfg = PhysicsManager._cfg
        device = PhysicsManager._device
        use_cuda_graph = cfg is not None and cfg.use_cuda_graph and "cuda" in device  # type: ignore[union-attr]

        with Timer(name="newton_cuda_graph", msg="CUDA graph took:"):
            if not use_cuda_graph:
                NewtonManager._graph = None
                return
            if cls._usdrt_stage is None:
                # No RTX active — use standard Warp capture (cudaStreamCaptureModeGlobal).
                with wp.ScopedCapture() as capture:
                    cls._simulate()
                NewtonManager._graph = capture.graph
                logger.info("Newton CUDA graph captured (standard Warp mode)")

                # TODO: Kamino: StateKamino.from_newton() lazily allocates body_f_total,
                # joint_q_prev, and joint_lambdas via wp.clone/wp.zeros during the
                # first step() inside graph capture. Replay once to pin those
                # memory-pool addresses before any eager solver.reset() call.
                wp.capture_launch(cls._graph)
            else:
                # RTX is active during initialization — cudaImportExternalMemory and other
                # non-capturable RTX ops run on background CUDA streams right now.
                # Defer capture to the first step() call, after RTX is fully initialized
                # and idle between render frames (clean capture window).
                NewtonManager._graph = None
                NewtonManager._graph_capture_pending = True
                logger.info("Newton CUDA graph capture deferred until first step() (RTX active)")