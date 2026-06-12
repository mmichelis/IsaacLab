# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the cable asset, registry, and replicate-hook plumbing."""

import math

import newton
import numpy as np
import pytest
import warp as wp
from isaaclab_newton.assets.articulation.articulation import Articulation
from isaaclab_newton.physics import FeatherstoneSolverCfg, NewtonCfg, NewtonManager
from isaaclab_newton.sim.spawners.materials import NewtonCableMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.cable import CableObject, CableObjectCfg
from isaaclab_contrib.cable.cable_object import (
    CableRegistryEntry,
    add_cable_entry_to_builder,
    add_registered_cables_to_builder,
    install_cable_builder_hooks,
)
from isaaclab_contrib.deformable.newton_manager_cfg import VBDSolverCfg
from isaaclab_contrib.deformable.vbd_manager import NewtonVBDManager


def test_install_cable_builder_hooks_is_idempotent(monkeypatch):
    """Repeated install must not duplicate registrations on _per_world_builder_hooks."""
    # Reset state so the test is self-contained.
    monkeypatch.setattr(NewtonManager, "_per_world_builder_hooks", [], raising=False)
    monkeypatch.delattr(NewtonManager, "_cable_registry", raising=False)

    install_cable_builder_hooks()
    install_cable_builder_hooks()
    install_cable_builder_hooks()

    assert NewtonManager._cable_registry == []
    matches = [h for h in NewtonManager._per_world_builder_hooks if h is add_registered_cables_to_builder]
    assert len(matches) == 1, "install_cable_builder_hooks must be idempotent"


def test_add_registered_cables_iterates_registry(monkeypatch):
    """The loop function dispatches to add_cable_entry_to_builder per registry entry."""
    monkeypatch.setattr(NewtonManager, "_per_world_builder_hooks", [], raising=False)

    calls = []

    def _fake_entry_hook(builder, entry, env_idx, env_pos, env_rot, cable_idx=0):
        calls.append((entry.prim_path, env_idx, cable_idx))

    monkeypatch.setattr(
        "isaaclab_contrib.cable.cable_object.add_cable_entry_to_builder",
        _fake_entry_hook,
    )
    entries = [
        CableRegistryEntry(
            prim_path="/World/cable_a",
            node_positions=[wp.vec3(0, 0, 0), wp.vec3(1, 0, 0)],
            edges=[(0, 1)],
            radius=0.005,
        ),
        CableRegistryEntry(
            prim_path="/World/cable_b",
            node_positions=[wp.vec3(0, 0, 0), wp.vec3(1, 0, 0)],
            edges=[(0, 1)],
            radius=0.005,
        ),
    ]
    monkeypatch.setattr(NewtonManager, "_cable_registry", entries, raising=False)

    add_registered_cables_to_builder(builder=None, world_idx=3, env_position=[0, 0, 0], env_rotation=[0, 0, 0, 1])

    assert calls == [("/World/cable_a", 3, 0), ("/World/cable_b", 3, 1)]


class _FakeBuilder:
    """Mocks the main ModelBuilder: records the per-env ``add_builder`` clone calls."""

    up_axis = "Z"
    gravity = -9.81

    def __init__(self):
        self.body_count = 0
        self.clones = []  # list of (proto, xform, label_prefix)

    def add_builder(self, proto, xform=None, label_prefix=None):
        self.clones.append((proto, xform, label_prefix))
        self.body_count += proto.body_count


@pytest.mark.parametrize(
    "env_rotation, env_position, init_pos, init_rot, expected_np0, expected_np1",
    [
        # Identity case: verifies field-forwarding + translation composition.
        (
            [0.0, 0.0, 0.0, 1.0],  # env identity
            [1.0, 0.0, 0.0],  # env_t = (1, 0, 0)
            (0.0, 0.0, 1.0),  # init_t = (0, 0, 1)
            (0.0, 0.0, 0.0, 1.0),  # init identity
            (1.0, 0.0, 1.0),  # node[0] world = env_t + init_t = (1, 0, 1)
            (1.1, 0.0, 1.0),  # node[1] world = (1.1, 0, 1)
        ),
        # 90° CCW about Z: verifies composed rotation.
        (
            [0.0, 0.0, math.sqrt(2.0) / 2.0, math.sqrt(2.0) / 2.0],
            [0.0, 0.0, 0.0],
            (0.0, 1.0, 0.0),  # init_t = (0, 1, 0)
            (0.0, 0.0, 0.0, 1.0),
            (-1.0, 0.0, 0.0),  # R_z(90°)·(0, 1, 0) = (-1, 0, 0)
            (-1.0, 0.1, 0.0),  # node[1] = (-1, 0, 0) + R_z(90°)·(0.1, 0, 0) = (-1, 0.1, 0)
        ),
    ],
    ids=["identity", "env_rotation_z90"],
)
def test_add_cable_entry_to_builder(
    monkeypatch, env_rotation, env_position, init_pos, init_rot, expected_np0, expected_np1
):
    """The prototype is built from canonical node positions with all material/geometry params
    forwarded to ``add_rod_graph``; the per-env world transform is applied via the
    ``add_builder`` xform (not baked into the node positions)."""
    rod_calls = []
    real_add_rod_graph = newton.ModelBuilder.add_rod_graph

    def _spy_add_rod_graph(self, **kwargs):
        rod_calls.append(kwargs)
        return real_add_rod_graph(self, **kwargs)

    monkeypatch.setattr(newton.ModelBuilder, "add_rod_graph", _spy_add_rod_graph)

    entry = CableRegistryEntry(
        prim_path="/World/Cable",
        node_positions=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.1, 0.0, 0.0)],
        edges=[(0, 1)],
        radius=0.005,
        init_pos=init_pos,
        init_rot=init_rot,
        stretch_stiffness=2.0e9,
        bend_stiffness=1.0e-3,
        stretch_damping=0.0,
        bend_damping=1.0e-4,
        density=1200.0,
    )
    builder = _FakeBuilder()
    add_cable_entry_to_builder(builder, entry, env_idx=0, env_position=env_position, env_rotation=env_rotation)

    # add_rod_graph runs once on the prototype with CANONICAL (untransformed) node positions.
    assert len(rod_calls) == 1
    call = rod_calls[0]
    assert [tuple(round(float(c), 6) for c in p) for p in call["node_positions"]] == [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0)]

    # Field forwarding to add_rod_graph.
    assert call["edges"] == [(0, 1)]
    assert call["radius"] == pytest.approx(0.005)
    assert call["stretch_stiffness"] == pytest.approx(2.0e9)
    assert call["bend_stiffness"] == pytest.approx(1.0e-3)
    assert call["bend_damping"] == pytest.approx(1.0e-4)
    assert call["label"] == "cable"
    assert float(call["cfg"].density) == pytest.approx(1200.0)

    # The per-env transform is applied via the add_builder xform + per-env label prefix.
    assert len(builder.clones) == 1
    _proto, xform, label_prefix = builder.clones[0]
    assert label_prefix == "/World/Cable"
    # World node positions = xform applied to the canonical nodes.
    for node, expected in zip(entry.node_positions, (expected_np0, expected_np1)):
        world = xform.p + wp.quat_rotate(xform.q, node)
        assert float(world[0]) == pytest.approx(expected[0], abs=1e-5)
        assert float(world[1]) == pytest.approx(expected[1], abs=1e-5)
        assert float(world[2]) == pytest.approx(expected[2], abs=1e-5)


def test_add_cable_entry_populates_body_offsets_and_last_edge_length():
    """``add_cable_entry_to_builder`` records per-env body offsets and the last edge length."""

    class _CloneCountingBuilder:
        up_axis = "Z"
        gravity = -9.81

        def __init__(self):
            self.body_count = 0

        def add_builder(self, proto, xform=None, label_prefix=None):
            self.body_count += proto.body_count

    entry = CableRegistryEntry(
        prim_path="/World/Cable",
        node_positions=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.2, 0.0, 0.0), wp.vec3(0.5, 0.0, 0.0), wp.vec3(0.9, 0.0, 0.0)],
        edges=[(0, 1), (1, 2), (2, 3)],
        radius=0.005,
    )
    builder = _CloneCountingBuilder()
    builder.body_count = 7
    add_cable_entry_to_builder(builder, entry, env_idx=0, env_position=[0, 0, 0], env_rotation=[0, 0, 0, 1])
    builder.body_count += 5
    add_cable_entry_to_builder(builder, entry, env_idx=1, env_position=[1, 0, 0], env_rotation=[0, 0, 0, 1])

    assert entry.body_offsets == [7, 15]
    assert entry.last_edge_length == pytest.approx(0.4)


def test_cable_object_cfg_defaults():
    """CableObjectCfg overrides actuators and articulation_root_prim_path."""
    cfg = CableObjectCfg(
        prim_path="/World/Cable",
        spawn=sim_utils.CableCfg(
            positions=[(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.2, 0.0, 0.0)],
            width=0.01,
            physics_material=NewtonCableMaterialCfg(),
        ),
    )
    assert cfg.articulation_root_prim_path == "/cable_articulation"
    assert cfg.actuators == {}


@pytest.mark.parametrize(
    "setup_registry, spawn, expected_exc, expected_match",
    [
        # spawn=None → ValueError mentioning "CableCfg"
        (True, None, ValueError, "CableCfg"),
        # registry not installed → RuntimeError mentioning the VBD solver requirement
        (False, "valid", RuntimeError, "VBD"),
    ],
    ids=["spawn_none", "hooks_not_installed"],
)
def test_cable_object_init_failure_paths(monkeypatch, setup_registry, spawn, expected_exc, expected_match):
    """CableObject.__init__ raises clear errors on invalid cfg or missing setup."""
    if setup_registry:
        monkeypatch.setattr(NewtonManager, "_cable_registry", [], raising=False)
    else:
        monkeypatch.delattr(NewtonManager, "_cable_registry", raising=False)
    monkeypatch.setattr(Articulation, "__init__", lambda self, cfg: setattr(self, "cfg", cfg))

    # "valid" sentinel → construct a real CableCfg
    if spawn == "valid":
        spawn_value = sim_utils.CableCfg(
            positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
            width=0.01,
            physics_material=NewtonCableMaterialCfg(),
        )
    else:
        spawn_value = spawn

    cfg = CableObjectCfg(prim_path="/World/Cable", spawn=spawn_value)
    with pytest.raises(expected_exc, match=expected_match):
        CableObject(cfg)


def test_cable_replicate_body_count():
    """Spawn 2 cables in env_0, replicate to 4 envs, verify total body count.

    Each cable has 3 control points → 2 segments per cable.
    Total cable bodies in builder = 4 envs × 2 cables × 2 segments = 16.
    """
    cable_spawn = sim_utils.CableCfg(
        positions=[(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.2, 0.0, 0.0)],
        width=0.01,
        physics_material=NewtonCableMaterialCfg(),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )

    @configclass
    class _SceneCfg(InteractiveSceneCfg):
        num_envs: int = 4
        env_spacing: float = 1.0
        cable_a: CableObjectCfg = CableObjectCfg(prim_path="{ENV_REGEX_NS}/CableA", spawn=cable_spawn)
        cable_b: CableObjectCfg = CableObjectCfg(prim_path="{ENV_REGEX_NS}/CableB", spawn=cable_spawn)

    # Cables need install_cable_builder_hooks called once before scene init.
    # This mirrors how NewtonVBDManager.initialize() calls
    # install_deformable_builder_hooks() before the deformable scene is set up.
    install_cable_builder_hooks()

    newton_sim_cfg = SimulationCfg(
        physics=NewtonCfg(solver_cfg=FeatherstoneSolverCfg()),
    )

    with build_simulation_context(device="cuda:0", sim_cfg=newton_sim_cfg, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        InteractiveScene(_SceneCfg())
        sim.reset()  # triggers newton_physics_replicate, materializing cable bodies

        model = NewtonManager.get_model()

        # Newton labels each cable body as "{prim_path}_cable_edge_body_{i}" before
        # label renaming and "{env_dest}/cable_edge_body_{i}" after.
        # Both forms contain the substring "cable_edge_body_".
        cable_body_count = sum(1 for label in model.body_label if "cable_edge_body_" in label)
        assert cable_body_count == 16, f"expected 16 cable bodies, got {cable_body_count}"


def test_forward_preserves_cable_body_q():
    """Regression test: :meth:`NewtonVBDManager.forward` must not corrupt cable ``body_q``.

    Newton's ``eval_fk`` has no case for :attr:`newton.JointType.CABLE`, so an unmasked FK
    pass collapses cable rod segments onto their parent anchors. ``forward`` lazily builds
    :attr:`~NewtonVBDManager._fk_mask` (via :meth:`~NewtonVBDManager._build_fk_mask`) to
    exclude CABLE/FREE articulations so VBD-owned ``body_q`` is preserved bit-identically.
    """
    cable_spawn = sim_utils.CableCfg(
        positions=[(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.2, 0.0, 0.0)],
        width=0.01,
        physics_material=NewtonCableMaterialCfg(),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )

    @configclass
    class _SceneCfg(InteractiveSceneCfg):
        num_envs: int = 1
        env_spacing: float = 1.0
        cable: CableObjectCfg = CableObjectCfg(prim_path="{ENV_REGEX_NS}/Cable", spawn=cable_spawn)

    newton_sim_cfg = SimulationCfg(physics=NewtonCfg(solver_cfg=VBDSolverCfg()))

    with build_simulation_context(device="cuda:0", sim_cfg=newton_sim_cfg, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        InteractiveScene(_SceneCfg())
        sim.reset()  # triggers replicate + start_simulation

        body_q_before = NewtonVBDManager._state_0.body_q.numpy().copy()

        # forward() is what Kit-style visualizers invoke each render; it lazily builds the
        # FK mask. With the mask, cable articulations are excluded from the FK pass and body_q
        # is bit-identical. Without it, JointType.CABLE relative transforms fall through to
        # identity, snapping each rod segment onto its parent anchor.
        NewtonVBDManager.forward()

        # The mask must have been built since cables are registered.
        assert NewtonVBDManager._fk_mask is not None, "Expected _fk_mask to be built when cables are registered."

        body_q_after = NewtonVBDManager._state_0.body_q.numpy()
        np.testing.assert_array_equal(
            body_q_after,
            body_q_before,
            err_msg="forward() altered body_q — cable mask did not exclude cable articulations.",
        )


def test_start_simulation_preserves_curved_cable_body_q():
    """Regression test for the cable body_q restoration after start_simulation's eval_fk.

    :meth:`NewtonManager.start_simulation` ends with an unmasked ``eval_fk`` to seed
    ``state_0.body_q`` from joint coordinates. Newton's ``eval_fk`` has no case for
    :attr:`newton.JointType.CABLE`, so cable joints fall through to identity and each
    child capsule collapses onto its parent joint anchor — rotating curved cables onto
    the root segment's local +Z axis.

    For a *straight* cable the corruption is invisible (eval_fk's identity output matches
    the layout produced by ``add_rod_graph``), so a non-collinear node layout is required
    to expose the bug. :meth:`NewtonVBDManager._restore_cable_body_q` undoes the corruption
    by copying ``model.body_q`` (untouched by ``eval_fk``) back into ``state_0.body_q`` for
    cable bodies.
    """
    # Curved cable: three nodes whose edges (0->1 along +x, 1->2 along +y) point in
    # different directions, so adjacent capsule orientations differ. eval_fk's identity
    # output would collapse body[1] onto body[0]'s +Z axis (still pointing +x), but the
    # rest pose has body[1] rotated to align +Z with +y.
    cable_spawn = sim_utils.CableCfg(
        positions=[(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.1, 0.1, 0.0)],
        width=0.01,
        physics_material=NewtonCableMaterialCfg(),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )

    @configclass
    class _SceneCfg(InteractiveSceneCfg):
        num_envs: int = 1
        env_spacing: float = 1.0
        cable: CableObjectCfg = CableObjectCfg(prim_path="{ENV_REGEX_NS}/Cable", spawn=cable_spawn)

    newton_sim_cfg = SimulationCfg(physics=NewtonCfg(solver_cfg=VBDSolverCfg()))

    with build_simulation_context(device="cuda:0", sim_cfg=newton_sim_cfg, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        InteractiveScene(_SceneCfg())
        sim.reset()  # triggers start_simulation -> unmasked eval_fk -> _restore_cable_body_q

        # ``model.body_q`` holds the rest pose produced by ``add_rod_graph`` and is never
        # written by ``eval_fk``. With the restoration in place, ``state_0.body_q`` for cable
        # bodies must match ``model.body_q`` bit-for-bit. Without the fix, the second cable
        # body's quaternion differs (eval_fk reuses the root segment's orientation).
        assert NewtonVBDManager._cable_registry, "Cable registry empty — replicate hook did not run."

        body_q_state = NewtonVBDManager._state_0.body_q.numpy()
        body_q_model = NewtonVBDManager._model.body_q.numpy()

        cable_body_indices: list[int] = []
        for entry in NewtonVBDManager._cable_registry:
            for body_offset in entry.body_offsets:
                cable_body_indices.extend(range(body_offset, body_offset + len(entry.edges)))

        np.testing.assert_allclose(
            body_q_state[cable_body_indices],
            body_q_model[cable_body_indices],
            err_msg=(
                "Cable body_q in state_0 does not match model.body_q after start_simulation."
                " The unmasked eval_fk corrupted cable bodies and _restore_cable_body_q did not"
                " restore them."
            ),
        )


def test_cable_object_reset_restores_body_state():
    """``NewtonVBDManager.reset(soft=True)`` snaps cable bodies back to the rest pose.

    Steps the sim to drift the cable away from its spawn pose, calls the soft VBD
    reset, and verifies that:

    1. ``state.body_q`` matches ``model.body_q`` for the cable's bodies.
    2. ``state.body_qd`` is zero for the cable's bodies.
    3. ``solver.body_q_prev`` is refreshed to the rest pose (otherwise AVBD's
       implicit velocity ``(body_q - body_q_prev) / dt`` would produce
       hundreds of m/s on the next step).
    4. ``solver.body_inertia_q`` is zero (matches solver-init default).
    5. One more ``sim.step()`` keeps ``|body_qd|`` bounded (regression for the
       ~700 m/s spurious-velocity bug).
    """
    cable_spawn = sim_utils.CableCfg(
        positions=[(0.0, 0.0, 0.0), (0.05, 0.0, 0.0), (0.1, 0.0, 0.0)],
        width=0.01,
        physics_material=NewtonCableMaterialCfg(),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )

    @configclass
    class _SceneCfg(InteractiveSceneCfg):
        num_envs: int = 1
        env_spacing: float = 1.0
        cable: CableObjectCfg = CableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cable",
            spawn=cable_spawn,
            init_state=CableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
        )

    newton_sim_cfg = SimulationCfg(physics=NewtonCfg(solver_cfg=VBDSolverCfg(iterations=10), num_substeps=4), dt=0.01)

    with build_simulation_context(device="cuda:0", sim_cfg=newton_sim_cfg, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        scene = InteractiveScene(_SceneCfg())
        sim.reset()

        cable = scene["cable"]
        entry = cable._registry_entry
        body_indices = list(range(entry.body_offsets[0], entry.body_offsets[0] + len(entry.edges)))

        body_q_model = NewtonVBDManager._model.body_q.numpy()[body_indices]

        # Step under gravity so the cable's body slice drifts away from the rest pose.
        for _ in range(20):
            sim.step()
        body_q_drifted = NewtonVBDManager._state_0.body_q.numpy()[body_indices]
        assert not np.allclose(body_q_drifted, body_q_model, atol=1e-4), (
            "Sim did not advance: cable body_q matches model.body_q without stepping."
        )

        NewtonVBDManager.reset(soft=True)

        body_q_after = NewtonVBDManager._state_0.body_q.numpy()[body_indices]
        body_qd_after = NewtonVBDManager._state_0.body_qd.numpy()[body_indices]
        body_q_prev_after = NewtonVBDManager._solver.body_q_prev.numpy()[body_indices]
        body_inertia_q_after = NewtonVBDManager._solver.body_inertia_q.numpy()[body_indices]

        np.testing.assert_allclose(
            body_q_after,
            body_q_model,
            err_msg="state.body_q was not restored to model.body_q after NewtonVBDManager.reset(soft=True).",
        )
        np.testing.assert_array_equal(
            body_qd_after,
            np.zeros_like(body_qd_after),
            err_msg="state.body_qd was not zeroed after NewtonVBDManager.reset(soft=True).",
        )
        np.testing.assert_allclose(
            body_q_prev_after,
            body_q_model,
            err_msg="solver.body_q_prev was not refreshed to model.body_q after NewtonVBDManager.reset(soft=True).",
        )
        np.testing.assert_array_equal(
            body_inertia_q_after,
            np.zeros_like(body_inertia_q_after),
            err_msg="solver.body_inertia_q was not zeroed after NewtonVBDManager.reset(soft=True).",
        )

        # One step of free-fall should add at most ~g*dt = ~0.1 m/s. A failure
        # here (e.g. ~700 m/s) indicates AVBD picked up stale solver-side state.
        sim.step()
        max_speed = float(np.abs(NewtonVBDManager._state_0.body_qd.numpy()[body_indices]).max())
        assert max_speed < 1.0, f"body_qd exploded after first post-reset step: |body_qd|_max={max_speed}"
