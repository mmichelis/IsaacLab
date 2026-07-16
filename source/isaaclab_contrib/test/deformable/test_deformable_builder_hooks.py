# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import pytest
from isaaclab_newton.physics import NewtonManager

from isaaclab.assets.deformable_object.base_deformable_object import BaseDeformableObject
from isaaclab.cloner.replicate_session import REPLICATION_QUEUE, queue_replication

from isaaclab_contrib.deformable import DeformableObject, VBDSolverCfg


def test_deformable_package_exports_public_symbols():
    """Test that deformable symbols are exported from the package root."""
    assert DeformableObject.__name__ == "DeformableObject"
    assert VBDSolverCfg.__name__ == "VBDSolverCfg"


def test_newton_deformable_registers_cfg_for_replication(monkeypatch):
    """Test that constructing a Newton deformable registers its cfg for replication.

    The queue holds the raw cfg (context resolution happens at :func:`replicate` time). The
    fake base init mirrors :meth:`AssetBase.__init__`, which enqueues every asset cfg.
    """
    cfg = SimpleNamespace()

    def fake_base_init(self, cfg):
        # Mirror AssetBase.__init__: register the cfg for cloning.
        queue_replication(cfg)
        self.cfg = cfg
        self._DTYPE_TO_TORCH_TRAILING_DIMS = {}
        self._initialize_handle = None
        self._invalidate_initialize_handle = None
        self._prim_deletion_handle = None
        self._debug_vis_handle = None
        self._physics_ready_handle = None

    monkeypatch.setattr(BaseDeformableObject, "__init__", fake_base_init)
    REPLICATION_QUEUE.clear()

    try:
        DeformableObject(cfg)
        queued = [queued_cfg for queued_cfg in REPLICATION_QUEUE if queued_cfg is cfg]
    finally:
        REPLICATION_QUEUE.clear()

    assert queued == [cfg]


def test_newton_deformable_rejects_missing_world(monkeypatch):
    """Test that imported deformable groups cover every simulation world."""
    groups = [SimpleNamespace(family="cloth", world=0, particle_start=0, particle_end=4)]
    monkeypatch.setattr(
        NewtonManager,
        "_get_deformable_particle_groups",
        classmethod(lambda cls, prim_path: groups),
    )
    monkeypatch.setattr(NewtonManager, "get_num_envs", classmethod(lambda cls: 2))
    deformable = object.__new__(DeformableObject)
    deformable.cfg = SimpleNamespace(prim_path="/World/envs/env_.*/Cloth")
    for name in ("_initialize_handle", "_invalidate_initialize_handle", "_prim_deletion_handle", "_debug_vis_handle"):
        setattr(deformable, name, None)

    with pytest.raises(RuntimeError, match=r"worlds \[0, 1\].*found \[0\]"):
        deformable._initialize_impl()
