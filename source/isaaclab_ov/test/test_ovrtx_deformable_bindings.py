# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for OVRTX deformable mesh point bindings."""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

_REQUIRED_MODULES = ("isaaclab_ov", "ovrtx", "pxr", "isaaclab_newton")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]

pytestmark = [
    pytest.mark.isaacsim_ci,
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}",
    ),
]

if not _MISSING_MODULES:
    import isaaclab_ov.renderers.ovrtx_renderer as ovrtx_renderer_module  # noqa: E402
    from isaaclab_newton.physics import NewtonManager  # noqa: E402
    from isaaclab_ov.renderers import OVRTXRendererCfg  # noqa: E402
    from isaaclab_ov.renderers.ovrtx_renderer import OVRTXRenderer  # noqa: E402
    from ovrtx import BindingFlag, DataAccess  # noqa: E402
else:
    NewtonManager = None
    OVRTXRenderer = None
    OVRTXRendererCfg = None
    ovrtx_renderer_module = None
    BindingFlag = None
    DataAccess = None


class _FakePointsBinding:
    """Capture array writes made through an OVRTX array attribute binding."""

    def __init__(self, attribute_name: str):
        self.attribute_name = attribute_name
        self.written = None
        self.write_kwargs: dict | None = None

    def write(self, data, **kwargs):
        self.written = data
        self.write_kwargs = kwargs

    def map(self, device=None, device_id=0):  # noqa: ARG002
        raise RuntimeError("bind_array_attribute bindings do not expose mapped point buffers")

    def unbind(self):
        pass


class _FakeOVRTXBackend:
    """Minimal OVRTX backend stub for deformable binding setup."""

    def __init__(self):
        self.bindings = {}
        self.calls = []
        self.writes = []

    def bind_array_attribute(self, **kwargs):
        self.calls.append(kwargs)
        binding = _FakePointsBinding(kwargs["attribute_name"])
        self.bindings[kwargs["attribute_name"]] = binding
        return binding

    def bind_attribute(self, **kwargs):
        self.calls.append(kwargs)
        binding = _FakePointsBinding(kwargs["attribute_name"])
        self.bindings[kwargs["attribute_name"]] = binding
        return binding

    def query_prims(self, **kwargs):  # noqa: ARG002
        return {
            "/World/envs/env_0/Deformable/mesh": {},
            "/World/envs/env_0/Deformable/geometry/mesh": {},
        }

    def write_attribute(self, **kwargs):
        self.writes.append(kwargs)


def _make_renderer_without_backend(device: str = "cpu") -> tuple[OVRTXRenderer, _FakeOVRTXBackend]:
    renderer = OVRTXRenderer.__new__(OVRTXRenderer)
    renderer.cfg = OVRTXRendererCfg()
    renderer._device = device
    renderer._camera_rel_path = "Camera"
    renderer._renderer = _FakeOVRTXBackend()
    renderer._deformable_points_binding = None
    renderer._deformable_visual_meshes = []
    renderer._particle_points_binding = None
    renderer._particle_visual_offsets = []
    renderer._particle_visual_counts = []
    renderer._particle_workaround_applied = False
    renderer._use_ovstage = False
    return renderer, renderer._renderer


def test_points_array_binding_uses_write_not_map():
    """OVRTX array bindings accept ``List[DLTensor]`` via ``write()``, not mapped tensors."""
    binding = _FakePointsBinding("points")
    with pytest.raises(RuntimeError, match="do not expose mapped point buffers"):
        binding.map()


def _set_deformable_model(monkeypatch: pytest.MonkeyPatch, meshes: list[SimpleNamespace]) -> None:
    model = SimpleNamespace(deformable_visual_meshes=meshes)
    monkeypatch.setattr(NewtonManager, "get_model", classmethod(lambda cls: model))


def test_setup_deformable_bindings_uses_newton_visual_paths(monkeypatch: pytest.MonkeyPatch):
    """OVRTX binds the exact graphics path for every Newton deformable visual mesh."""
    renderer, backend = _make_renderer_without_backend()
    meshes = [
        SimpleNamespace(
            graphics_path="/World/envs/env_0/Surface/visual",
            label="/World/envs/env_0/Surface/fallback",
            world=-1,
            index=0,
        ),
        SimpleNamespace(
            graphics_path="/World/envs/env_0/Volume/visual",
            label="/World/envs/env_0/Volume/fallback",
            world=-1,
            index=1,
        ),
    ]
    _set_deformable_model(monkeypatch, meshes)

    renderer._setup_deformable_bindings(num_envs=1)

    assert backend.calls[0]["prim_paths"] == [
        "/World/envs/env_0/Surface/visual",
        "/World/envs/env_0/Volume/visual",
    ]
    assert renderer._deformable_visual_meshes == meshes
    assert renderer._deformable_points_binding is backend.bindings["points"]
    assert [write["attribute_name"] for write in backend.writes] == [
        "omni:resetXformStack",
        "omni:xform",
    ]


def test_setup_deformable_bindings_skips_visuals_without_graphics_paths(monkeypatch: pytest.MonkeyPatch):
    """OVRTX ignores Newton visual meshes that are not bound to USD graphics prims."""
    renderer, backend = _make_renderer_without_backend()
    unbound = SimpleNamespace(graphics_path=None, label="programmatic", world=-1, index=0)
    bound = SimpleNamespace(graphics_path="/World/Cloth/visual", label="cloth", world=-1, index=1)
    _set_deformable_model(monkeypatch, [unbound, bound])

    renderer._setup_deformable_bindings(num_envs=1)

    assert backend.calls[0]["prim_paths"] == ["/World/Cloth/visual"]
    assert renderer._deformable_visual_meshes == [bound]


def test_setup_deformable_bindings_supports_body_visuals(monkeypatch: pytest.MonkeyPatch):
    """Body-driven Newton deformable visuals do not require particle state."""
    renderer, backend = _make_renderer_without_backend()
    mesh = SimpleNamespace(
        graphics_path="/World/Cable/visual",
        label="/World/Cable/fallback",
        world=-1,
        index=0,
    )
    _set_deformable_model(monkeypatch, [mesh])

    renderer._setup_deformable_bindings(num_envs=1)

    assert backend.calls[0]["prim_paths"] == ["/World/Cable/visual"]


def test_setup_deformable_bindings_skips_missing_model(monkeypatch: pytest.MonkeyPatch):
    """OVRTX skips deformable setup before a Newton model is available."""
    renderer, backend = _make_renderer_without_backend()
    monkeypatch.setattr(NewtonManager, "get_model", classmethod(lambda cls: None))

    renderer._setup_deformable_bindings(num_envs=1)

    assert backend.calls == []


def test_setup_deformable_bindings_skips_empty_visuals(monkeypatch: pytest.MonkeyPatch):
    """OVRTX skips deformable setup when Newton has no visual meshes."""
    renderer, backend = _make_renderer_without_backend()
    _set_deformable_model(monkeypatch, [])

    renderer._setup_deformable_bindings(num_envs=1)

    assert backend.calls == []


def test_update_deformable_points_uses_newton_visuals(monkeypatch: pytest.MonkeyPatch):
    """OVRTX writes Newton-evaluated deformable points."""
    renderer, _backend = _make_renderer_without_backend()
    renderer._deformable_points_binding = _FakePointsBinding("points")
    mesh = SimpleNamespace(index=0)
    renderer._deformable_visual_meshes = [mesh]
    points = wp.array(
        [wp.vec3f(1.0, 2.0, 3.0), wp.vec3f(4.0, 5.0, 6.0)],
        dtype=wp.vec3f,
        device="cpu",
    )

    class _FakeVisuals:
        def __init__(self):
            self.waited_on = None

        def wait(self, stream):
            self.waited_on = stream

        def get_points(self, requested_mesh):
            assert requested_mesh is mesh
            return points

    visuals = _FakeVisuals()
    monkeypatch.setattr(NewtonManager, "get_state", classmethod(lambda cls: SimpleNamespace(particle_q=None)))
    monkeypatch.setattr(NewtonManager, "get_deformable_visuals", classmethod(lambda cls: visuals))

    class _FakeStream:
        cuda_stream = 42

    stream = _FakeStream()
    monkeypatch.setattr(ovrtx_renderer_module.wp, "get_stream", lambda device: stream)  # noqa: ARG005

    renderer.update_geometries()

    assert visuals.waited_on is stream
    assert renderer._deformable_points_binding.written == [points]
    assert renderer._deformable_points_binding.write_kwargs["data_access"] is DataAccess.ASYNC
    assert renderer._deformable_points_binding.write_kwargs["cuda_stream"] == 42


def test_setup_particle_points_bindings_binds_mpm_visual_prims(monkeypatch: pytest.MonkeyPatch):
    """MPM particle visual prims create an OPTIMIZE ``points`` array binding."""
    renderer, backend = _make_renderer_without_backend()
    particle_visual_prims = {
        "/World/envs/env_0/Media/Particles": SimpleNamespace(offset=10, count=5),
        "/World/envs/env_1/Media/Particles": SimpleNamespace(offset=15, count=5),
    }

    monkeypatch.setattr(NewtonManager, "_particle_visual_prims", particle_visual_prims)

    renderer._setup_particle_bindings()

    assert len(backend.calls) == 1
    assert backend.calls[0]["prim_paths"] == [
        "/World/envs/env_0/Media/Particles",
        "/World/envs/env_1/Media/Particles",
    ]
    assert backend.calls[0]["attribute_name"] == "points"
    assert backend.calls[0]["flags"] is BindingFlag.OPTIMIZE
    assert renderer._particle_points_binding is backend.bindings["points"]
    assert renderer._particle_workaround_applied is False
    assert renderer._particle_visual_offsets == [10, 15]
    assert renderer._particle_visual_counts == [5, 5]
    assert len(backend.writes) == 2
    assert backend.writes[0]["attribute_name"] == "omni:resetXformStack"
    assert backend.writes[1]["attribute_name"] == "omni:xform"


def test_setup_particle_points_bindings_binds_multiple_mpm_assets(monkeypatch: pytest.MonkeyPatch):
    """Multiple MPM assets bind as ``num_assets * num_envs`` points prims, like deformables."""
    renderer, backend = _make_renderer_without_backend()
    particle_visual_prims = {
        "/World/envs/env_0/Media/Particles": SimpleNamespace(offset=0, count=5),
        "/World/envs/env_1/Media/Particles": SimpleNamespace(offset=5, count=5),
        "/World/envs/env_0/Foam/Particles": SimpleNamespace(offset=10, count=3),
        "/World/envs/env_1/Foam/Particles": SimpleNamespace(offset=13, count=3),
    }

    monkeypatch.setattr(NewtonManager, "_particle_visual_prims", particle_visual_prims)

    renderer._setup_particle_bindings()

    # Binding order follows dict insertion order (no path sort).
    assert backend.calls[0]["prim_paths"] == [
        "/World/envs/env_0/Media/Particles",
        "/World/envs/env_1/Media/Particles",
        "/World/envs/env_0/Foam/Particles",
        "/World/envs/env_1/Foam/Particles",
    ]
    assert renderer._particle_visual_offsets == [0, 5, 10, 13]
    assert renderer._particle_visual_counts == [5, 5, 3, 3]


def test_update_particle_points_primes_with_host_sync_then_gpu_async(monkeypatch: pytest.MonkeyPatch):
    """First MPM ``points`` update host-SYNC primes via binding; later frames use GPU ASYNC."""
    renderer, backend = _make_renderer_without_backend()
    renderer._particle_points_binding = _FakePointsBinding("points")
    renderer._particle_visual_offsets = [2]
    renderer._particle_visual_counts = [2]
    renderer._particle_workaround_applied = False
    particle_q = wp.array(
        [
            wp.vec3f(0.0, 0.0, 0.0),
            wp.vec3f(1.0, 0.0, 0.0),
            wp.vec3f(2.0, 3.0, 4.0),
            wp.vec3f(5.0, 6.0, 7.0),
        ],
        dtype=wp.vec3f,
        device="cpu",
    )
    monkeypatch.setattr(NewtonManager, "get_state", classmethod(lambda cls: SimpleNamespace(particle_q=particle_q)))

    class _FakeStream:
        cuda_stream = 42

    monkeypatch.setattr(ovrtx_renderer_module.wp, "get_stream", lambda device: _FakeStream())  # noqa: ARG005

    renderer.update_geometries()

    assert renderer._particle_workaround_applied is True
    assert len(backend.writes) == 0
    written = renderer._particle_points_binding.written
    assert written is not None
    assert len(written) == 1
    assert isinstance(written[0], np.ndarray)
    assert written[0].tolist() == [
        [2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0],
    ]
    assert renderer._particle_points_binding.write_kwargs["data_access"] is DataAccess.SYNC
    assert "cuda_stream" not in renderer._particle_points_binding.write_kwargs

    renderer.update_geometries()

    written = renderer._particle_points_binding.written
    assert written is not None
    assert len(written) == 1
    assert written[0].ptr == particle_q[2:4].ptr
    assert renderer._particle_points_binding.write_kwargs["data_access"] is DataAccess.ASYNC
    assert renderer._particle_points_binding.write_kwargs["cuda_stream"] == 42
    assert len(backend.writes) == 0


def test_update_geometries_writes_deformable_and_mpm_bindings(monkeypatch: pytest.MonkeyPatch):
    """Deformable visuals and MPM particles use their independent update paths."""
    renderer, backend = _make_renderer_without_backend()
    renderer._deformable_points_binding = _FakePointsBinding("deformable_points")
    mesh = SimpleNamespace(index=0)
    renderer._deformable_visual_meshes = [mesh]
    visual_points = wp.array(
        [wp.vec3f(0.0, 0.0, 0.0), wp.vec3f(1.0, 0.0, 0.0)],
        dtype=wp.vec3f,
        device="cpu",
    )
    visuals = SimpleNamespace(
        wait=lambda stream: None,
        get_points=lambda requested_mesh: visual_points,
    )
    renderer._particle_points_binding = _FakePointsBinding("points")
    renderer._particle_visual_offsets = [2]
    renderer._particle_visual_counts = [2]
    renderer._particle_workaround_applied = False
    particle_q = wp.array(
        [
            wp.vec3f(0.0, 0.0, 0.0),
            wp.vec3f(1.0, 0.0, 0.0),
            wp.vec3f(2.0, 3.0, 4.0),
            wp.vec3f(5.0, 6.0, 7.0),
        ],
        dtype=wp.vec3f,
        device="cpu",
    )
    monkeypatch.setattr(NewtonManager, "get_state", classmethod(lambda cls: SimpleNamespace(particle_q=particle_q)))
    monkeypatch.setattr(NewtonManager, "get_deformable_visuals", classmethod(lambda cls: visuals))

    class _FakeStream:
        cuda_stream = 42

    monkeypatch.setattr(ovrtx_renderer_module.wp, "get_stream", lambda device: _FakeStream())  # noqa: ARG005

    renderer.update_geometries()

    assert renderer._deformable_points_binding.written == [visual_points]
    assert renderer._deformable_points_binding.write_kwargs["data_access"] is DataAccess.ASYNC
    assert renderer._particle_workaround_applied is True
    assert len(backend.writes) == 0
    mpm_written = renderer._particle_points_binding.written
    assert isinstance(mpm_written[0], np.ndarray)
    assert mpm_written[0].tolist() == [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]
    assert renderer._particle_points_binding.write_kwargs["data_access"] is DataAccess.SYNC

    renderer.update_geometries()

    mpm_written = renderer._particle_points_binding.written
    assert len(mpm_written) == 1
    assert mpm_written[0].ptr == particle_q[2:4].ptr
    assert renderer._particle_points_binding.write_kwargs["data_access"] is DataAccess.ASYNC
    assert len(backend.writes) == 0
