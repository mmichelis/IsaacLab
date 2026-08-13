# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from isaaclab_newton.cloner import PHYSICS_CONTEXT, NewtonReplicateContext
from isaaclab_newton.physics import NewtonManager

from isaaclab_contrib.deformable import DeformableObject, VBDSolverCfg
from isaaclab_contrib.deformable.vbd_manager import NewtonVBDManager


def test_deformable_package_exports_public_symbols():
    """Test that deformable symbols are exported from the package root."""
    assert DeformableObject.__name__ == "DeformableObject"
    assert VBDSolverCfg.__name__ == "VBDSolverCfg"


@pytest.mark.parametrize("external_rigid_solver", [False, True])
def test_vbd_solver_force_input_capability(monkeypatch, external_rigid_solver: bool):
    """VBD consumes rigid forces only when it owns AVBD rigid integration."""
    solver = object()
    monkeypatch.setattr(NewtonVBDManager, "_create_solver", lambda model, cfg: solver)
    monkeypatch.setattr(NewtonManager, "_solver", None)
    monkeypatch.setattr(NewtonManager, "_use_single_state", True)
    monkeypatch.setattr(NewtonManager, "_needs_collision_pipeline", False)
    monkeypatch.setattr(NewtonManager, "_supports_rigid_body_force_input", False)

    NewtonVBDManager._build_solver(object(), VBDSolverCfg(integrate_with_external_rigid_solver=external_rigid_solver))

    assert NewtonManager._solver is solver
    assert NewtonManager._supports_rigid_body_force_input is not external_rigid_solver


def test_newton_physics_context_is_replicate_context():
    """Test that Newton exports its replicate context as the physics context."""
    assert PHYSICS_CONTEXT is NewtonReplicateContext
