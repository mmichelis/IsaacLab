# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for ``NewtonManager.update_visualization_state`` and shadow-model build.

When the active sim backend is PhysX and a Newton-native visualizer/renderer is in
use, :meth:`NewtonManager._ensure_visualization_model` must build the manager's
``_model`` / ``_state_0`` directly from the USD stage, and
:meth:`NewtonManager.update_visualization_state` must copy fresh transforms and
simulation nodes into ``_state_0`` through :class:`~isaaclab.scene_data.SceneDataProvider`.
Newton then evaluates deformable visual meshes from that state.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.integration

_DEFAULT = object()


def _reset_newton_manager_state():
    from isaaclab_newton.physics import NewtonManager

    NewtonManager._builder = None
    NewtonManager._model = None
    NewtonManager._state_0 = None
    NewtonManager._num_envs = None
    NewtonManager._scene_data = None
    NewtonManager._scene_data_mapping = None
    NewtonManager._scene_data_points = None
    NewtonManager._scene_data_geometry_mapping = None
    NewtonManager._scene_data_geometry_mapping_ready = False
    NewtonManager._deformable_visuals = None


def _make_env_stage(num_envs: int = 1):
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/envs")
    for env_id in range(num_envs):
        UsdGeom.Xform.Define(stage, f"/World/envs/env_{env_id}")
    return stage


def _make_standalone_stage():
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/Robot")
    return stage


def _set_sim_context(monkeypatch, nm, clone_plan=_DEFAULT, scene_data_provider=_DEFAULT):
    clone_plan = SimpleNamespace() if clone_plan is _DEFAULT else clone_plan
    scene_data_provider = SimpleNamespace() if scene_data_provider is _DEFAULT else scene_data_provider
    sim = SimpleNamespace(
        get_clone_plan=lambda: clone_plan,
        get_scene_data_provider=lambda: scene_data_provider,
    )
    monkeypatch.setattr(nm.SimulationContext, "instance", classmethod(lambda cls: sim))
    return sim


def _make_points_backend(points, geometry_paths: list[str], geometry_counts: list[int]):
    """Build a minimal SceneData backend that exposes flattened geometry points."""
    import warp as wp

    from isaaclab.scene_data.scene_data_backend import SceneDataBackend, SceneDataFormat

    class _PointsBackend(SceneDataBackend):
        def __init__(self):
            self._points = wp.array(points, dtype=wp.vec3f, device="cpu")
            self._points_data = SceneDataFormat.Points()
            self._points_data.points = self._points
            self._geometry_paths = list(geometry_paths)
            self._geometry_counts = list(geometry_counts)

        @property
        def transforms(self) -> SceneDataFormat.Transform:
            return SceneDataFormat.Transform()

        @property
        def transform_count(self) -> int:
            return 0

        @property
        def transform_paths(self) -> list[str]:
            return []

        @property
        def points(self) -> SceneDataFormat.Points:
            return self._points_data

        @property
        def point_count(self) -> int:
            return int(self._points.shape[0])

        @property
        def geometry_paths(self) -> list[str]:
            return self._geometry_paths

        @property
        def geometry_counts(self) -> list[int]:
            return self._geometry_counts

    return _PointsBackend()


def _make_finalize_builder(*, body_count: int, particle_count: int = 0, finalize: bool = True):
    """ModelBuilder stub used by ``_ensure_visualization_model`` tests."""

    class _Builder:
        pass

    builder = _Builder()
    builder.body_count = body_count
    builder.particle_count = particle_count
    if finalize:

        def _finalize(device):
            return SimpleNamespace(
                deformable_visual_meshes=[], state=lambda: SimpleNamespace(body_q=None, particle_q=None)
            )

        builder.finalize = _finalize
    return builder


def test_physics_manager_close_only_clears_active_manager_binding(monkeypatch):
    """Only the active physics manager can clear shared SimulationContext state."""
    from isaaclab.physics import PhysicsManager

    class _ActiveManager(PhysicsManager):
        _callbacks = {}

    class _InactiveManager(PhysicsManager):
        pass

    _ActiveManager.close()
    assert PhysicsManager._sim is None

    active_sim = SimpleNamespace(physics_manager=_ActiveManager)
    monkeypatch.setattr(PhysicsManager, "_sim", active_sim, raising=False)
    monkeypatch.setattr(PhysicsManager, "_cfg", "active-cfg", raising=False)
    monkeypatch.setattr(PhysicsManager, "_sim_time", 1.25, raising=False)

    monkeypatch.setattr(PhysicsManager, "_callbacks", {1: (None, lambda _: None, 0, "stale", None)}, raising=False)
    _InactiveManager.close()
    assert PhysicsManager._callbacks == {}
    assert (PhysicsManager._sim, PhysicsManager._cfg, PhysicsManager._sim_time) == (active_sim, "active-cfg", 1.25)

    _ActiveManager.close()
    assert (PhysicsManager._sim, PhysicsManager._cfg, PhysicsManager._sim_time) == (None, None, 0.0)


def test_ensure_visualization_model_noop_when_backend_is_newton(monkeypatch):
    """When sim backend is Newton, the manager keeps its own model/state untouched."""
    from isaaclab_newton.physics import NewtonManager

    _reset_newton_manager_state()
    monkeypatch.setattr(NewtonManager, "_backend_is_newton", classmethod(lambda cls, scene_data_provider=None: True))
    NewtonManager._ensure_visualization_model()
    assert NewtonManager._model is None
    assert NewtonManager._state_0 is None


def test_ensure_visualization_model_builds_from_stage_when_backend_is_physx(monkeypatch):
    """With a PhysX sim backend, the shadow Newton model is built directly from the stage."""
    from isaaclab_newton.physics import NewtonManager
    from isaaclab_newton.physics import newton_manager as nm

    _reset_newton_manager_state()
    monkeypatch.setattr(NewtonManager, "_backend_is_newton", classmethod(lambda cls, scene_data_provider=None: False))
    monkeypatch.setattr(nm, "get_current_stage", lambda *args, **kwargs: _make_env_stage())
    monkeypatch.setattr(nm.PhysicsManager, "_sim", None, raising=False)
    _set_sim_context(monkeypatch, nm)
    monkeypatch.setattr(nm.PhysicsManager, "_device", "cpu", raising=False)

    finalize_calls: list[str] = []
    builder = _make_finalize_builder(body_count=3)

    def _finalize(device):
        finalize_calls.append(device)
        return SimpleNamespace(deformable_visual_meshes=[], state=lambda: SimpleNamespace(body_q=None, particle_q=None))

    builder.finalize = _finalize
    monkeypatch.setattr(nm, "build_visualization_builder_from_stage_envs", lambda *args, **kwargs: builder)

    NewtonManager._ensure_visualization_model()

    assert finalize_calls == ["cpu"]
    assert NewtonManager._model is not None
    assert NewtonManager._state_0 is not None


def test_ensure_visualization_model_empty_builder_supports_marker_only_scene(monkeypatch, caplog):
    """An empty shadow model supports marker-only and geometry-only scenes."""
    from isaaclab_newton.physics import NewtonManager
    from isaaclab_newton.physics import newton_manager as nm

    _reset_newton_manager_state()
    monkeypatch.setattr(NewtonManager, "_backend_is_newton", classmethod(lambda cls, scene_data_provider=None: False))
    monkeypatch.setattr(nm, "get_current_stage", lambda *args, **kwargs: _make_standalone_stage())
    monkeypatch.setattr(nm.PhysicsManager, "_sim", None, raising=False)
    _set_sim_context(monkeypatch, nm, clone_plan=None)
    monkeypatch.setattr(nm.PhysicsManager, "_device", "cpu", raising=False)

    builder = _make_finalize_builder(body_count=0, particle_count=0)
    monkeypatch.setattr(nm, "build_visualization_builder_from_stage_envs", lambda *args, **kwargs: builder)

    with caplog.at_level("INFO"):
        NewtonManager._ensure_visualization_model()

    assert NewtonManager._model is not None
    assert NewtonManager._state_0 is not None
    assert any("no Newton bodies or particles" in r.message for r in caplog.records)


def test_ensure_visualization_model_empty_builder_logs_and_skips(monkeypatch, caplog):
    """When a cloned stage walk produces no bodies or particles, model/state stay unset."""
    from isaaclab_newton.physics import NewtonManager
    from isaaclab_newton.physics import newton_manager as nm

    _reset_newton_manager_state()
    monkeypatch.setattr(NewtonManager, "_backend_is_newton", classmethod(lambda cls, scene_data_provider=None: False))
    monkeypatch.setattr(nm, "get_current_stage", lambda *args, **kwargs: _make_env_stage())
    monkeypatch.setattr(nm.PhysicsManager, "_sim", None, raising=False)
    _set_sim_context(monkeypatch, nm)

    builder = _make_finalize_builder(body_count=0, particle_count=0, finalize=False)
    monkeypatch.setattr(nm, "build_visualization_builder_from_stage_envs", lambda *args, **kwargs: builder)

    with caplog.at_level("ERROR"):
        NewtonManager._ensure_visualization_model()

    assert NewtonManager._model is None
    assert NewtonManager._state_0 is None
    assert any("no Newton bodies or particles" in r.message for r in caplog.records)


def test_ensure_visualization_model_populates_num_envs_when_backend_is_physx(monkeypatch):
    """Shadow-model build must populate ``_num_envs`` so ``get_num_envs`` is correct under PhysX."""
    from isaaclab_newton.physics import NewtonManager
    from isaaclab_newton.physics import newton_manager as nm

    _reset_newton_manager_state()
    monkeypatch.setattr(NewtonManager, "_backend_is_newton", classmethod(lambda cls, scene_data_provider=None: False))
    monkeypatch.setattr(nm, "get_current_stage", lambda *args, **kwargs: _make_env_stage(num_envs=4))
    monkeypatch.setattr(nm.PhysicsManager, "_sim", None, raising=False)
    _set_sim_context(monkeypatch, nm)
    monkeypatch.setattr(nm.PhysicsManager, "_device", "cpu", raising=False)

    builder = _make_finalize_builder(body_count=3)
    monkeypatch.setattr(nm, "build_visualization_builder_from_stage_envs", lambda *args, **kwargs: builder)

    NewtonManager._ensure_visualization_model()

    assert NewtonManager.get_num_envs() == 4
    assert NewtonManager._model.num_envs == 4


def test_ensure_visualization_model_builds_single_world_for_standalone_scene(monkeypatch):
    """A scene outside ``/World/envs`` is imported as one visualization world."""
    from isaaclab_newton.physics import NewtonManager
    from isaaclab_newton.physics import newton_manager as nm

    _reset_newton_manager_state()
    monkeypatch.setattr(NewtonManager, "_backend_is_newton", classmethod(lambda cls, scene_data_provider=None: False))
    monkeypatch.setattr(nm, "get_current_stage", lambda *args, **kwargs: _make_standalone_stage())
    monkeypatch.setattr(nm.PhysicsManager, "_sim", None, raising=False)
    _set_sim_context(monkeypatch, nm, clone_plan=None)
    monkeypatch.setattr(nm.PhysicsManager, "_device", "cpu", raising=False)

    build_calls = []
    builder = _make_finalize_builder(body_count=1)

    def _build(stage, env_paths, clone_plan, *, up_axis, device="cpu"):
        build_calls.append((stage, env_paths, clone_plan, up_axis, device))
        return builder

    monkeypatch.setattr(nm, "build_visualization_builder_from_stage_envs", _build)

    NewtonManager._ensure_visualization_model()

    assert build_calls[0][1:3] == ([], None)
    assert build_calls[0][4] == "cpu"
    assert NewtonManager.get_num_envs() == 1
    assert NewtonManager._model.num_envs == 1


def test_ensure_visualization_model_missing_stage_leaves_state_unset(monkeypatch, caplog):
    """When no USD stage is available, model/state stay unset and an error is logged."""
    from isaaclab_newton.physics import NewtonManager
    from isaaclab_newton.physics import newton_manager as nm

    _reset_newton_manager_state()
    monkeypatch.setattr(NewtonManager, "_backend_is_newton", classmethod(lambda cls, scene_data_provider=None: False))
    monkeypatch.setattr(nm, "get_current_stage", lambda *args, **kwargs: None)

    with caplog.at_level("ERROR"):
        NewtonManager._ensure_visualization_model()

    assert NewtonManager._model is None
    assert NewtonManager._state_0 is None
    assert any("No USD stage available" in r.message for r in caplog.records)


def test_update_visualization_state_noop_when_backend_is_newton(monkeypatch):
    """When sim backend is Newton, update_visualization_state is a no-op."""
    from isaaclab_newton.physics import NewtonManager

    _reset_newton_manager_state()
    monkeypatch.setattr(NewtonManager, "_backend_is_newton", classmethod(lambda cls, scene_data_provider=None: True))
    monkeypatch.setattr(NewtonManager, "get_scene_data_provider", classmethod(lambda cls: SimpleNamespace()))

    # Pre-set sentinel values to ensure update doesn't touch them.
    NewtonManager._model = "live-model"
    NewtonManager._state_0 = "live-state"
    NewtonManager.update_visualization_state()
    assert NewtonManager._model == "live-model"
    assert NewtonManager._state_0 == "live-state"


@pytest.mark.parametrize("newton_active", [True, False])
def test_get_state_forwards_only_for_live_newton_state(monkeypatch, newton_active):
    """PhysX shadow state keeps its visualization update without entering Newton FK."""
    from isaaclab_newton.physics import NewtonManager

    events: list[str] = []
    state = object()
    monkeypatch.setattr(NewtonManager, "_fk_reset_mask", object(), raising=False)
    monkeypatch.setattr(
        NewtonManager,
        "_backend_is_newton",
        classmethod(lambda cls, provider=None: newton_active),
    )
    monkeypatch.setattr(NewtonManager, "forward", classmethod(lambda cls: events.append("forward")))
    monkeypatch.setattr(
        NewtonManager,
        "update_visualization_state",
        classmethod(lambda cls, provider=None: events.append("visualization")),
    )
    monkeypatch.setattr(NewtonManager, "get_state_0", classmethod(lambda cls: state))

    assert NewtonManager.get_state() is state
    expected = ["forward", "visualization"] if newton_active else ["visualization"]
    assert events == expected


@pytest.mark.parametrize("kind_name, expected_dirty", [("BODY", True), ("PARTICLE", False), ("TET", False)])
def test_pre_render_marks_body_deformable_visuals_dirty(monkeypatch, kind_name, expected_dirty):
    """Body-driven deformable visuals are refreshed after transform changes."""
    from isaaclab_newton.physics import NewtonManager
    from newton import DeformableVisualMesh

    from isaaclab.physics import PhysicsManager

    events = []
    mesh = SimpleNamespace(kind=getattr(DeformableVisualMesh.Kind, kind_name))
    monkeypatch.setattr(NewtonManager, "_model", SimpleNamespace(deformable_visual_meshes=[mesh]))
    monkeypatch.setattr(NewtonManager, "_particles_dirty", False)
    monkeypatch.setattr(NewtonManager, "_fk_reset_mask", object())
    monkeypatch.setattr(NewtonManager, "_transforms_may_change_on_graph_replay", True)
    monkeypatch.setattr(PhysicsManager, "_device", None)
    monkeypatch.setattr(NewtonManager, "forward", classmethod(lambda cls: events.append("forward")))
    monkeypatch.setattr(
        NewtonManager,
        "sync_transforms_to_usd",
        classmethod(lambda cls: events.append("transforms")),
    )
    monkeypatch.setattr(NewtonManager, "sync_cables_to_usd", classmethod(lambda cls: events.append("cables")))
    monkeypatch.setattr(
        NewtonManager,
        "sync_particles_to_usd",
        classmethod(lambda cls: events.append(("particles", cls._particles_dirty))),
    )

    NewtonManager.pre_render()

    assert events == ["forward", "transforms", "cables", ("particles", expected_dirty)]


def test_scene_data_reads_through_public_state_boundary(monkeypatch):
    """SceneData does not bypass the coherent Newton state accessor."""
    import warp as wp
    from isaaclab_newton.physics import NewtonManager
    from isaaclab_newton.physics import newton_manager as nm

    events: list[str] = []
    body_q = wp.zeros(1, dtype=wp.transformf, device="cpu")
    state = SimpleNamespace(body_q=body_q)
    backend = nm.NewtonSceneDataBackend()
    monkeypatch.setattr(
        NewtonManager,
        "get_state",
        classmethod(lambda cls, provider=None: events.append("state") or state),
    )

    transforms = backend.transforms

    assert events == ["state"]
    assert transforms.transforms is body_q


def test_resolve_scene_data_body_paths_uses_joint_body_targets():
    """PhysX visualization sync maps Newton joint labels to the actual body prim path."""
    pytest.importorskip("pxr")
    from isaaclab_newton.physics import NewtonManager

    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    body_prim = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot/robot0_forearm").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body_prim)
    joint = UsdPhysics.FixedJoint.Define(stage, "/World/envs/env_0/Robot/joints/robot0_forearm")
    joint.GetBody1Rel().SetTargets([body_prim.GetPath()])

    body_paths = ["/World/envs/env_0/Robot/joints/robot0_forearm"]
    resolved_paths = NewtonManager._resolve_scene_data_body_paths(body_paths, stage)

    assert resolved_paths == ["/World/envs/env_0/Robot/robot0_forearm"]


def test_update_visualization_state_writes_scene_points_into_particle_state(monkeypatch):
    """PhysX SceneData writes simulation nodes directly into Newton particle state."""
    import warp as wp
    from isaaclab_newton.physics import NewtonManager
    from isaaclab_newton.physics import newton_manager as nm

    from pxr import Usd, UsdGeom

    from isaaclab.scene_data.scene_data_provider import SceneDataProvider

    body_path = "/World/envs/env_0/Cloth"
    sim_path = f"{body_path}/simulation"
    stage = Usd.Stage.CreateInMemory()
    body_prim = UsdGeom.Xform.Define(stage, body_path).GetPrim()
    body_prim.AddAppliedSchema("OmniPhysicsDeformableBodyAPI")
    UsdGeom.Mesh.Define(stage, sim_path)
    monkeypatch.setattr(SceneDataProvider, "usd_stage", property(lambda self: stage))

    provider = SceneDataProvider(
        _make_points_backend(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [body_path],
            [2],
        )
    )
    view_count = 0

    class _DeformableView:
        labels = [sim_path]

        def __init__(self):
            nonlocal view_count
            view_count += 1

        @staticmethod
        def ranges(element: str):
            assert element == "particle"
            return [(1, 3)]

    particle_q = wp.array(
        [
            wp.vec3(9.0, 9.0, 9.0),
            wp.vec3(),
            wp.vec3(),
        ],
        dtype=wp.vec3f,
        device="cpu",
    )
    _reset_newton_manager_state()
    monkeypatch.setattr(NewtonManager, "_backend_is_newton", classmethod(lambda cls, provider=None: False))
    monkeypatch.setattr(nm, "DeformableView", lambda *args, **kwargs: _DeformableView())
    NewtonManager._model = SimpleNamespace(body_label=[], deformable_visual_meshes=[])
    NewtonManager._state_0 = SimpleNamespace(body_q=None, particle_q=particle_q)

    NewtonManager.update_visualization_state(provider)
    NewtonManager.update_visualization_state(provider)

    assert NewtonManager._state_0.particle_q is particle_q
    assert particle_q.numpy().tolist() == [[9.0, 9.0, 9.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert view_count == 2


def test_update_visualization_state_supports_global_and_world_deformables(monkeypatch):
    """SceneData maps global and per-world deformable groups separately."""
    import warp as wp
    from isaaclab_newton.physics import NewtonManager
    from isaaclab_newton.physics import newton_manager as nm

    from isaaclab.scene_data.scene_data_provider import SceneDataProvider

    labels = ["/World/GlobalCloth", "/World/envs/env_0/Cloth"]
    provider = SceneDataProvider(_make_points_backend([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], labels, [1, 1]))

    class _DeformableView:
        def __init__(self, selected):
            self.labels = selected

        def ranges(self, element: str):
            assert element == "particle"
            return [(labels.index(label), labels.index(label) + 1) for label in self.labels]

    def _view(_model, pattern, *, family, verbose):
        if family == "volume":
            raise KeyError
        if pattern == "*":
            raise ValueError
        return _DeformableView([label for label in labels if pattern.fullmatch(label)])

    particle_q = wp.zeros(2, dtype=wp.vec3f, device="cpu")
    _reset_newton_manager_state()
    monkeypatch.setattr(NewtonManager, "_backend_is_newton", classmethod(lambda cls, provider=None: False))
    monkeypatch.setattr(nm, "DeformableView", _view)
    NewtonManager._model = SimpleNamespace(
        body_label=[],
        deformable_visual_meshes=[],
        surface_label=labels,
        surface_world=[-1, 0],
        volume_label=[],
        volume_world=[],
    )
    NewtonManager._state_0 = SimpleNamespace(body_q=None, particle_q=particle_q)

    NewtonManager.update_visualization_state(provider)

    assert particle_q.numpy().tolist() == [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]


def test_get_deformable_visuals_evaluates_current_particle_state(monkeypatch: pytest.MonkeyPatch):
    """Newton evaluates an embedded visual point from the current particle state."""
    import numpy as np
    import warp as wp
    from isaaclab_newton.physics import NewtonManager
    from newton import ModelBuilder

    builder = ModelBuilder()
    builder.add_soft_mesh(
        pos=wp.vec3(),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(),
        vertices=[
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(1.0, 0.0, 0.0),
            wp.vec3(0.0, 1.0, 0.0),
            wp.vec3(0.0, 0.0, 1.0),
        ],
        indices=[0, 1, 2, 3],
        density=1.0,
        k_mu=1.0,
        k_lambda=1.0,
        k_damp=0.0,
    )
    builder.add_deformable_visual_mesh(
        vertices=np.zeros((1, 3), dtype=np.float32),
        indices=[0, 0, 0],
        kind="tet",
        tet_range=(0, 1),
        parent=[0],
        weights=np.array([[0.25, 0.25, 0.25, 0.25]], dtype=np.float32),
        label="/World/envs/env_0/Soft/visual",
    )
    model = builder.finalize(device="cpu")
    state = model.state()
    particle_q = state.particle_q.numpy()
    particle_q[:, 0] += 1.0
    state.particle_q.assign(particle_q)

    _reset_newton_manager_state()
    NewtonManager._model = model
    NewtonManager._state_0 = model.state()
    NewtonManager._deformable_visuals = model.deformable_visuals()
    monkeypatch.setattr(NewtonManager, "get_state", classmethod(lambda cls: state))

    visuals = NewtonManager.get_deformable_visuals()

    assert visuals is NewtonManager._deformable_visuals
    np.testing.assert_allclose(visuals.points.numpy(), [[1.25, 0.25, 0.25]])


def test_fabric_deformable_sync_evaluates_internal_state_without_public_accessor(monkeypatch: pytest.MonkeyPatch):
    """Initial Fabric sync evaluates visuals without invoking pre-solver forward kinematics."""
    import sys

    from isaaclab_newton.physics import NewtonManager

    fake_usdrt = SimpleNamespace(
        Sdf=SimpleNamespace(ValueTypeNames=SimpleNamespace(Point3fArray=object(), UInt=object(), Matrix4d=object())),
        Usd=SimpleNamespace(Access=SimpleNamespace(Read=object(), ReadWrite=object())),
    )
    monkeypatch.setitem(sys.modules, "usdrt", fake_usdrt)

    class _Model:
        def __init__(self):
            self.updated = False

        def update_deformable_visuals(self, state, visuals):
            assert state is NewtonManager._state_0
            assert visuals is NewtonManager._deformable_visuals
            self.updated = True

    class _Stage:
        def SelectPrims(self, **kwargs):
            return SimpleNamespace(GetCount=lambda: 0)

    _reset_newton_manager_state()
    NewtonManager._model = _Model()
    NewtonManager._state_0 = object()
    NewtonManager._deformable_visuals = SimpleNamespace()
    NewtonManager._usdrt_stage = _Stage()
    monkeypatch.setattr(
        NewtonManager,
        "get_deformable_visuals",
        classmethod(lambda cls: pytest.fail("public visualization accessor must not run during internal sync")),
    )

    NewtonManager._sync_fabric_mesh_particles()

    assert NewtonManager._model.updated
