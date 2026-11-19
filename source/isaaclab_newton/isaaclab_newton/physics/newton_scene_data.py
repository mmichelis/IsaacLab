# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Publish Newton state through Isaac Lab's scene-data interface."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import warp as wp
from newton import JointType, Model, State

from isaaclab.scene_data import SceneDataBackend, SceneDataFormat

if TYPE_CHECKING:
    from .newton_manager import NewtonManager


def _periodic_curve_body_ranges(model: Model) -> set[tuple[int, int]]:
    """Find native curves whose owned ROD joints form one directed cycle, excluding roots and attachments."""
    periodic = set()
    if not model.curve_label:
        return periodic
    joint_type = model.joint_type.numpy()
    parent, child = model.joint_parent.numpy(), model.joint_child.numpy()
    for start, end, joint_start, joint_end in zip(
        model.curve_body_start.numpy(),
        model.curve_body_end.numpy(),
        model.curve_joint_start.numpy(),
        model.curve_joint_end.numpy(),
        strict=True,
    ):
        parents, children = parent[joint_start:joint_end], child[joint_start:joint_end]
        internal = (joint_type[joint_start:joint_end] == JointType.ROD) & (parents >= start) & (parents < end)
        internal &= (children >= start) & (children < end)
        parents, children = parents[internal], children[internal]
        bodies = np.arange(start, end)
        if len(bodies) < 2 or not (
            np.array_equal(np.sort(parents), bodies) and np.array_equal(np.sort(children), bodies)
        ):
            continue
        # Degree alone would also accept multiple disconnected cycles in one native curve object.
        successors = dict(zip(parents, children, strict=True))
        visited, body = set(), start
        while body not in visited:
            visited.add(body)
            body = successors[body]
        if len(visited) == len(bodies):
            periodic.add((int(start), int(end)))
    return periodic


def _capsule_endpoint_indices(shape_ids: Sequence[int] | np.ndarray, *, periodic: bool) -> np.ndarray:
    """Build int32 rows of (left shape, left sign, right shape, right sign) in curve order."""
    ids = np.asarray(shape_ids, dtype=np.int32)
    if periodic:
        left, right = np.roll(ids, 1), ids
    else:
        left = np.concatenate((ids[:1], ids))
        right = np.concatenate((ids, ids[-1:]))
    left_sign = np.ones(len(left), dtype=np.int32)
    right_sign = -np.ones(len(right), dtype=np.int32)
    if not periodic:
        left_sign[0], right_sign[-1] = -1, 1
    return np.column_stack((left, left_sign, right, right_sign))


class NewtonSceneDataBackend(SceneDataBackend):
    """Scene data backend that reads rigid body transforms from Newton's simulation state.

    The backend reads ``body_q`` (an array of :class:`wp.transformf`) from Newton's current state and exposes it as
    :class:`SceneDataFormat.Transform`. Body paths come from the model's ``body_label`` attribute.
    """

    def __init__(self, manager: NewtonManager):
        """Initialize the scene data backend.

        Args:
            manager: Newton manager whose backend state to publish; the backend changes on every hard reset.
        """
        self._manager = manager
        self._transforms = SceneDataFormat.Transform()
        self.transforms_timestamp = 0
        self.geometry_timestamp = 0
        self._geometry_batches = []

    def initialize_geometry(self, geometry_batches: list, cable_bindings: dict[str, list[int]]) -> None:
        """Bind imported geometry paths to native particle ranges and capsule endpoints.

        Args:
            geometry_batches: Deformable and particle geometry recorded by replication.
            cable_bindings: Native capsule shape indices of each open or periodic cable, in curve order.
        """
        self._geometry_batches = list(geometry_batches)
        self.geometry_timestamp += 1
        if not cable_bindings:
            return
        endpoints, ranges = [], {}
        offset = 0
        model = self.model
        periodic_ranges = _periodic_curve_body_ranges(model)
        shape_body = model.shape_body.numpy()
        for path, shapes in cable_bindings.items():
            bodies = np.unique(shape_body[shapes])
            # Match a complete native curve, independently of the caller's mutable shape ordering.
            periodic = len(bodies) == len(shapes) and len(bodies) == bodies[-1] - bodies[0] + 1
            periodic = periodic and (int(bodies[0]), int(bodies[-1]) + 1) in periodic_ranges
            indices = _capsule_endpoint_indices(shapes, periodic=periodic)
            endpoints.append(indices)
            ranges[path] = (offset, len(indices))
            offset += len(indices)
        source = SceneDataFormat.CapsuleEndpoints()
        source.transforms = self._manager.backend.state_0.body_q
        source.shape_body = model.shape_body
        source.shape_transform = model.shape_transform
        source.shape_scale = model.shape_scale
        source.endpoints = wp.array(np.concatenate(endpoints), dtype=wp.vec4i, device=model.device)
        self._geometry_batches.append((source, ranges))

    @property
    def native_geometry_formats(self) -> tuple[type, ...]:
        return (SceneDataFormat.Points, SceneDataFormat.WeightedPoints, SceneDataFormat.CapsuleEndpoints)

    def get_geometry_batches(self, output_format=SceneDataFormat.Points):
        """Publish native arrays; SDP derives cable endpoints and applies destination layouts."""
        state = self.state
        for source, _ in self._geometry_batches:
            attribute = "transforms" if source._cls is SceneDataFormat.CapsuleEndpoints else "points"
            data = state.particle_q if attribute == "points" else state.body_q
            if getattr(source, attribute) is not data:
                setattr(source, attribute, data)
                self.geometry_timestamp += 1
        return self._geometry_batches

    @property
    def transforms(self) -> SceneDataFormat.Transform:
        """Publish the authoritative native pointer, including solver state-buffer swaps."""
        transforms = self.state.body_q
        if self._transforms.transforms is not transforms:
            self._transforms.transforms = transforms
            self.transforms_timestamp += 1
        return self._transforms

    @property
    def transform_count(self) -> int:
        """Return the number of rigid body transforms in the Newton sim."""
        return self.model.body_count

    @property
    def transform_paths(self) -> list[str]:
        """Return the prim paths for each rigid body transform."""
        if self.model.body_label is not None:
            return list(self.model.body_label)
        return []

    @property
    def model(self) -> Model | None:
        backend = self._manager.backend
        return None if backend is None else backend.model

    @property
    def state(self) -> State | None:
        """Return native physics state, consistent with authored state, without entering the rendering path."""
        manager = self._manager
        backend = manager.backend
        if backend is None:
            return None
        if backend.transforms_may_change_on_graph_replay:
            # Raw external graph replays bypass Python invalidation, so these reads must stay conservative.
            self.transforms_timestamp += 1
            self.geometry_timestamp += 1
        if backend.solver is not None:
            # Recording a graph must leave pending FK/reset work for the next eager read.
            if backend.device.is_capturing:
                backend.notify_model_changes()
            else:
                manager.forward()
        return backend.state_0
