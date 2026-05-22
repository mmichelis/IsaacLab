# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Pure-Python unit tests for proxy-coupled MJWarp+MPM partitioning logic.

These tests intentionally do NOT launch :class:`isaaclab.app.AppLauncher`.
:class:`NewtonProxyCoupledMJWarpMPMManager` delegates body/joint/shape
partitioning and proxy-body resolution to module-level helpers that operate on
a Newton :class:`newton.Model` and an :class:`isaaclab.scene.InteractiveSceneCfg`,
so they can be tested against minimal fakes without spinning up Isaac Sim.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
from newton import ShapeFlags

from isaaclab.managers import SceneEntityCfg

from isaaclab_contrib.deformable._proxy_partition import (
    partition_model_by_entities,
    resolve_entity_to_body_ids,
    select_proxy_bodies,
)

_CFG_LABEL = "ProxyCoupledMJWarpMPMSolverCfg"

##
# Fakes
##


@dataclass
class _FakeArray:
    """Minimal stand-in for a Newton/warp array exposing ``.numpy()``."""

    data: np.ndarray

    def numpy(self) -> np.ndarray:
        return self.data


@dataclass
class _FakeModel:
    """Minimal stand-in for :class:`newton.Model`."""

    body_count: int
    body_label: list[str]
    joint_count: int = 0
    joint_child: _FakeArray = field(default_factory=lambda: _FakeArray(np.zeros(0, dtype=np.int32)))
    shape_count: int = 0
    shape_body: _FakeArray = field(default_factory=lambda: _FakeArray(np.zeros(0, dtype=np.int32)))
    shape_flags: _FakeArray = field(default_factory=lambda: _FakeArray(np.zeros(0, dtype=np.int32)))


@dataclass
class _FakeAsset:
    """Stand-in for a scene asset cfg with the ``prim_path`` field consulted by the helpers."""

    prim_path: str


@dataclass
class _FakeSceneCfg:
    """Stand-in for :class:`InteractiveSceneCfg`."""

    robot: _FakeAsset | None = None
    other: _FakeAsset | None = None


##
# Helpers
##


def _model_with_two_bodies(
    *,
    with_shapes: bool = False,
    with_joints: bool = False,
    extra_static_shape: bool = False,
) -> _FakeModel:
    """Build a 2-body Franka-like model under ``/World/envs/env_0/Robot``."""
    body_count = 2
    body_label = [
        "/World/envs/env_0/Robot/panda_link0",
        "/World/envs/env_0/Robot/panda_hand",
    ]

    shape_body = np.zeros(0, dtype=np.int32)
    shape_flags = np.zeros(0, dtype=np.int32)
    shape_count = 0
    if with_shapes:
        owners = [0, 1]
        if extra_static_shape:
            owners.append(-1)
        shape_body = np.asarray(owners, dtype=np.int32)
        shape_flags = np.full(len(owners), int(ShapeFlags.COLLIDE_SHAPES), dtype=np.int32)
        shape_count = len(owners)

    joint_child = np.zeros(0, dtype=np.int32)
    joint_count = 0
    if with_joints:
        joint_child = np.asarray([1], dtype=np.int32)
        joint_count = 1

    return _FakeModel(
        body_count=body_count,
        body_label=body_label,
        joint_count=joint_count,
        joint_child=_FakeArray(joint_child),
        shape_count=shape_count,
        shape_body=_FakeArray(shape_body),
        shape_flags=_FakeArray(shape_flags),
    )


def _robot_scene() -> _FakeSceneCfg:
    return _FakeSceneCfg(robot=_FakeAsset(prim_path="/World/envs/env_.*/Robot"))


##
# resolve_entity_to_body_ids
##


def test_resolve_entity_no_body_names_returns_all_under_asset():
    model = _model_with_two_bodies()
    body_ids = resolve_entity_to_body_ids(
        model, SceneEntityCfg("robot"), _robot_scene(), cfg_label=_CFG_LABEL, field="mjwarp_bodies"
    )
    assert body_ids == [0, 1]


def test_resolve_entity_body_names_filter_by_regex():
    model = _model_with_two_bodies()
    body_ids = resolve_entity_to_body_ids(
        model,
        SceneEntityCfg("robot", body_names=["panda_hand"]),
        _robot_scene(),
        cfg_label=_CFG_LABEL,
        field="proxy_bodies",
    )
    assert body_ids == [1]


def test_resolve_entity_asset_missing_on_scene_cfg_raises():
    model = _model_with_two_bodies()
    with pytest.raises(ValueError, match="not on the attached scene cfg"):
        resolve_entity_to_body_ids(
            model,
            SceneEntityCfg("missing_asset"),
            _robot_scene(),
            cfg_label=_CFG_LABEL,
            field="mjwarp_bodies",
        )


def test_resolve_entity_unmatched_body_names_raises():
    model = _model_with_two_bodies()
    with pytest.raises(ValueError, match="no bodies matching"):
        resolve_entity_to_body_ids(
            model,
            SceneEntityCfg("robot", body_names=["nonexistent_link"]),
            _robot_scene(),
            cfg_label=_CFG_LABEL,
            field="proxy_bodies",
        )


##
# partition_model_by_entities
##


def test_partition_splits_bodies_joints_shapes():
    model = _model_with_two_bodies(with_shapes=True, with_joints=True, extra_static_shape=True)
    scene = _FakeSceneCfg(
        robot=_FakeAsset(prim_path="/World/envs/env_.*/Robot"),
        other=_FakeAsset(prim_path="/World/envs/env_.*/Robot"),
    )

    mjc_b, mpm_b, mjc_j, mpm_j, mjc_s, mpm_s = partition_model_by_entities(
        model,
        entry_a_bodies=[SceneEntityCfg("robot", body_names=["panda_link0"])],
        entry_b_bodies=[SceneEntityCfg("other", body_names=["panda_hand"])],
        scene_cfg=scene,
        cfg_label=_CFG_LABEL,
        entry_a_field="mjwarp_bodies",
        entry_b_field="mpm_bodies",
    )

    assert mjc_b == [0]
    assert mpm_b == [1]
    assert mjc_j == []
    assert mpm_j == [0]
    # Shape 0 -> body 0 (MJC). Shape 1 -> body 1 (MPM). Shape 2 -> body -1 (static, -> MPM).
    assert mjc_s == [0]
    assert mpm_s == [1, 2]


def test_partition_overlapping_bodies_raises():
    model = _model_with_two_bodies()
    scene = _FakeSceneCfg(
        robot=_FakeAsset(prim_path="/World/envs/env_.*/Robot"),
        other=_FakeAsset(prim_path="/World/envs/env_.*/Robot"),
    )
    with pytest.raises(ValueError, match="match both"):
        partition_model_by_entities(
            model,
            entry_a_bodies=[SceneEntityCfg("robot")],
            entry_b_bodies=[SceneEntityCfg("other", body_names=["panda_hand"])],
            scene_cfg=scene,
            cfg_label=_CFG_LABEL,
            entry_a_field="mjwarp_bodies",
            entry_b_field="mpm_bodies",
        )


def test_partition_unclaimed_bodies_raises():
    model = _model_with_two_bodies()
    with pytest.raises(ValueError, match="unclaimed"):
        partition_model_by_entities(
            model,
            entry_a_bodies=[SceneEntityCfg("robot", body_names=["panda_link0"])],
            entry_b_bodies=[],
            scene_cfg=_robot_scene(),
            cfg_label=_CFG_LABEL,
            entry_a_field="mjwarp_bodies",
            entry_b_field="mpm_bodies",
        )


##
# select_proxy_bodies
##


def test_select_proxy_bodies_filters_to_collide_shapes():
    model = _model_with_two_bodies(with_shapes=True)
    model.shape_flags = _FakeArray(np.asarray([0, int(ShapeFlags.COLLIDE_SHAPES)], dtype=np.int32))

    proxy_ids = select_proxy_bodies(
        model,
        proxy_bodies=[SceneEntityCfg("robot", body_names=["panda_link0", "panda_hand"])],
        scene_cfg=_robot_scene(),
        cfg_label=_CFG_LABEL,
    )
    assert proxy_ids == [1]


def test_select_proxy_bodies_requires_body_names():
    model = _model_with_two_bodies(with_shapes=True)
    with pytest.raises(ValueError, match="requires `body_names`"):
        select_proxy_bodies(
            model,
            proxy_bodies=[SceneEntityCfg("robot")],
            scene_cfg=_robot_scene(),
            cfg_label=_CFG_LABEL,
        )


def test_select_proxy_bodies_empty_input_returns_empty():
    proxy_ids = select_proxy_bodies(
        model=_FakeModel(body_count=0, body_label=[]),
        proxy_bodies=[],
        scene_cfg=None,
        cfg_label=_CFG_LABEL,
    )
    assert proxy_ids == []


def test_select_proxy_bodies_deduplicates_across_entries():
    model = _model_with_two_bodies(with_shapes=True)
    proxy_ids = select_proxy_bodies(
        model,
        proxy_bodies=[
            SceneEntityCfg("robot", body_names=["panda_hand"]),
            SceneEntityCfg("robot", body_names=["panda_hand"]),
        ],
        scene_cfg=_robot_scene(),
        cfg_label=_CFG_LABEL,
    )
    assert proxy_ids == [1]


##
# Raw prim-path strings as selectors
##


def test_resolve_string_prefix_claims_all_bodies_under_path():
    model = _model_with_two_bodies()
    body_ids = resolve_entity_to_body_ids(
        model,
        spec="/World/envs/env_.*/Robot",
        scene_cfg=None,
        cfg_label=_CFG_LABEL,
        field="mjwarp_bodies",
    )
    assert body_ids == [0, 1]


def test_resolve_string_narrows_to_a_single_body():
    model = _model_with_two_bodies()
    body_ids = resolve_entity_to_body_ids(
        model,
        spec="/World/envs/env_.*/Robot/panda_hand",
        scene_cfg=None,
        cfg_label=_CFG_LABEL,
        field="proxy_bodies",
    )
    assert body_ids == [1]


def test_resolve_string_no_matches_raises():
    model = _model_with_two_bodies()
    with pytest.raises(ValueError, match="matched no bodies"):
        resolve_entity_to_body_ids(
            model,
            spec="/World/envs/env_.*/Nonexistent",
            scene_cfg=None,
            cfg_label=_CFG_LABEL,
            field="mjwarp_bodies",
        )


def test_partition_accepts_mixed_string_and_scene_entity():
    model = _model_with_two_bodies(with_shapes=True, with_joints=True)
    mjc_b, mpm_b, _, _, _, _ = partition_model_by_entities(
        model,
        entry_a_bodies=[SceneEntityCfg("robot", body_names=["panda_link0"])],
        entry_b_bodies=["/World/envs/env_.*/Robot/panda_hand"],
        scene_cfg=_robot_scene(),
        cfg_label=_CFG_LABEL,
        entry_a_field="mjwarp_bodies",
        entry_b_field="mpm_bodies",
    )
    assert mjc_b == [0]
    assert mpm_b == [1]


def test_select_proxy_bodies_accepts_string_without_body_names():
    model = _model_with_two_bodies(with_shapes=True)
    proxy_ids = select_proxy_bodies(
        model,
        proxy_bodies=["/World/envs/env_.*/Robot/panda_hand"],
        scene_cfg=None,
        cfg_label=_CFG_LABEL,
    )
    assert proxy_ids == [1]


##
# Manager existence smoke (constructor not invoked — just import path)
##


def test_manager_class_importable_from_config_class_type():
    """The new config's ``class_type`` resolves to the new manager file."""
    from isaaclab_contrib.deformable.newton_manager_cfg import ProxyCoupledMJWarpMPMSolverCfg

    cfg = ProxyCoupledMJWarpMPMSolverCfg()
    assert "NewtonProxyCoupledMJWarpMPMManager" in str(cfg.class_type)
    assert "proxy_coupled_mjwarp_mpm_manager" in str(cfg.class_type)
