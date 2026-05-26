# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curated state container for :class:`CableObject`.

Wraps a Newton :class:`~newton.selection.ArticulationView` and exposes only the
fields that are meaningful for a 1D rod: root state, per-segment body state,
and the internal cable-joint state. Joint control / actuator / tendon /
wrench-composer / Jacobian / mass-matrix surfaces of
:class:`~isaaclab_newton.assets.ArticulationData` are intentionally absent.

This class is fully independent of :class:`ArticulationData`: it owns its own
sim-binding plumbing for the curated surface. The bindings, lazy buffers and
:class:`~isaaclab.utils.warp.ProxyArray` wrappers are populated by
:meth:`_create_simulation_bindings`, which is called on initial setup and
re-called by :class:`CableObject` on every ``PHYSICS_READY`` event so the
data class survives full solver resets.
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

import warp as wp
from isaaclab_newton.assets import kernels as shared_kernels
from isaaclab_newton.assets.articulation import kernels as articulation_kernels
from isaaclab_newton.physics import NewtonManager as SimulationManager

from isaaclab.utils.buffers import TimestampedBufferWarp as TimestampedBuffer
from isaaclab.utils.warp import ProxyArray
from isaaclab.utils.warp.utils import capture_unsafe

if TYPE_CHECKING:
    from newton.selection import ArticulationView


_LAZY_CAPTURE_REASON = (
    "This is a lazily-computed derived property guarded by a Python timestamp check "
    "that is invisible during graph replay.  Use Tier 1 base data (root_link_pose_w, "
    "root_com_vel_w, body_link_pose_w, body_com_vel_w, joint_pos, joint_vel) and "
    "inline the computation in your warp kernel.  See GRAPH_CAPTURE_MIGRATION.md."
)


class CableData:
    """State container for :class:`CableObject` — see module docstring."""

    def __init__(self, root_view: ArticulationView, device: str):
        """Initialize the data container.

        Args:
            root_view: Newton ArticulationView selecting this cable across all envs.
            device: Compute device string (e.g. ``"cuda:0"``).
        """
        # Hold the view as a weak reference to avoid circular references between
        # CableObject and CableData.
        self._root_view_ref = weakref.ref(root_view)
        self._device = device
        self._sim_timestamp: float = 0.0
        self._fk_timestamp: float = 0.0
        self._is_primed: bool = False

        # Direct sim bindings (populated by :meth:`_create_simulation_bindings`).
        self._sim_bind_root_link_pose_w: wp.array | None = None
        self._sim_bind_root_com_vel_w: wp.array | None = None
        self._sim_bind_body_link_pose_w: wp.array | None = None
        self._sim_bind_body_com_vel_w: wp.array | None = None
        self._sim_bind_joint_pos: wp.array | None = None
        self._sim_bind_joint_vel: wp.array | None = None
        # ``body_com_pos_b`` is needed internally by the COM→link velocity
        # kernels even though it is not part of the public cable surface.
        self._sim_bind_body_com_pos_b: wp.array | None = None

        # Lazy buffers backing computed properties. Allocated by
        # :meth:`_create_simulation_bindings` once the view's counts are known.
        self._root_link_vel_w: TimestampedBuffer | None = None
        self._body_link_vel_w: TimestampedBuffer | None = None
        self._joint_acc: TimestampedBuffer | None = None
        self._previous_joint_vel: wp.array | None = None

        # ProxyArray wrappers — pinned by :meth:`_pin_proxy_arrays` on first
        # init and again on any rebind. ``None`` until then.
        self._root_link_pose_w_ta: ProxyArray | None = None
        self._root_com_vel_w_ta: ProxyArray | None = None
        self._body_link_pose_w_ta: ProxyArray | None = None
        self._body_com_vel_w_ta: ProxyArray | None = None
        self._joint_pos_ta: ProxyArray | None = None
        self._joint_vel_ta: ProxyArray | None = None
        self._root_link_vel_w_ta: ProxyArray | None = None
        self._body_link_vel_w_ta: ProxyArray | None = None
        self._joint_acc_ta: ProxyArray | None = None

        # Sliced ProxyArrays + their backing wp.array views.
        self._root_link_pos_w: wp.array | None = None
        self._root_link_pos_w_ta: ProxyArray | None = None
        self._root_link_quat_w: wp.array | None = None
        self._root_link_quat_w_ta: ProxyArray | None = None
        self._root_link_lin_vel_w: wp.array | None = None
        self._root_link_lin_vel_w_ta: ProxyArray | None = None
        self._root_link_ang_vel_w: wp.array | None = None
        self._root_link_ang_vel_w_ta: ProxyArray | None = None
        self._body_link_pos_w: wp.array | None = None
        self._body_link_pos_w_ta: ProxyArray | None = None
        self._body_link_quat_w: wp.array | None = None
        self._body_link_quat_w_ta: ProxyArray | None = None
        self._body_link_lin_vel_w: wp.array | None = None
        self._body_link_lin_vel_w_ta: ProxyArray | None = None
        self._body_link_ang_vel_w: wp.array | None = None
        self._body_link_ang_vel_w_ta: ProxyArray | None = None
        self._body_com_lin_vel_w: wp.array | None = None
        self._body_com_lin_vel_w_ta: ProxyArray | None = None
        self._body_com_ang_vel_w: wp.array | None = None
        self._body_com_ang_vel_w_ta: ProxyArray | None = None

        # Counts cached at bind time.
        self._num_instances: int = 0
        self._num_bodies: int = 0
        self._num_joints: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def device(self) -> str:
        """Compute device string."""
        return self._device

    @property
    def is_primed(self) -> bool:
        """Whether the data container has been fully initialized."""
        return self._is_primed

    @is_primed.setter
    def is_primed(self, value: bool) -> None:
        self._is_primed = bool(value)

    def update(self, dt: float) -> None:
        """Advance the internal sim timestamp and refresh finite-difference buffers.

        Mirrors :meth:`isaaclab_newton.assets.articulation.ArticulationData.update`
        restricted to the cable's curated surface.

        Args:
            dt: Simulation step [s].
        """
        # update the simulation timestamp
        self._sim_timestamp += dt
        # FK is current after a sim step — keep fk_timestamp in sync unless it
        # was explicitly invalidated.
        if self._fk_timestamp >= 0.0:
            self._fk_timestamp = self._sim_timestamp
        # Trigger an update of the joint acceleration buffer (finite diff) only
        # once sim bindings exist. Pre-bind (e.g. unit tests with a MagicMock
        # view) we skip this so the constructor stays mock-friendly.
        if self._joint_acc is not None:
            _ = self.joint_acc

    def _ensure_fk_fresh(self) -> None:
        """Run forward kinematics if joint state has changed since the last FK update.

        Mirrors :meth:`ArticulationData._ensure_fk_fresh`. Newton's
        ``state.body_q`` (per-body world transforms) is updated by ``eval_fk``,
        invoked here through ``SimulationManager.forward()``. After a manual
        joint or root write that bypassed the sim step,
        ``_fk_timestamp`` is set to ``-1.0`` to force a refresh on the next
        read of any property that depends on body poses.
        """
        if self._fk_timestamp < self._sim_timestamp:
            SimulationManager.forward()
            self._fk_timestamp = self._sim_timestamp

    # ------------------------------------------------------------------
    # Counts (from the view)
    # ------------------------------------------------------------------

    @property
    def num_instances(self) -> int:
        """Number of cable instances (one per env)."""
        return self._root_view_ref().count

    @property
    def num_bodies(self) -> int:
        """Number of capsule bodies per cable instance."""
        return self._root_view_ref().link_count

    @property
    def num_joints(self) -> int:
        """Number of internal cable joints per instance."""
        return self._root_view_ref().joint_dof_count

    # ------------------------------------------------------------------
    # Root state
    # ------------------------------------------------------------------

    @property
    def root_link_pose_w(self) -> ProxyArray:
        """Root body pose in world frame [m, quat], shape ``(num_instances, 7)``."""
        return self._root_link_pose_w_ta

    @property
    def root_link_pos_w(self) -> ProxyArray:
        """Root body position in world frame [m], shape ``(num_instances, 3)``."""
        self._root_link_pos_w = self._get_pos_from_transform(self._root_link_pos_w, self.root_link_pose_w.warp)
        if self._root_link_pos_w_ta is None:
            self._root_link_pos_w_ta = ProxyArray(self._root_link_pos_w)
        return self._root_link_pos_w_ta

    @property
    def root_link_quat_w(self) -> ProxyArray:
        """Root body orientation as quaternion ``(x, y, z, w)``, shape ``(num_instances, 4)``."""
        self._root_link_quat_w = self._get_quat_from_transform(self._root_link_quat_w, self.root_link_pose_w.warp)
        if self._root_link_quat_w_ta is None:
            self._root_link_quat_w_ta = ProxyArray(self._root_link_quat_w)
        return self._root_link_quat_w_ta

    @property
    def root_com_vel_w(self) -> ProxyArray:
        """Root body velocity at the COM in world frame [m/s, rad/s], shape ``(num_instances, 6)``."""
        return self._root_com_vel_w_ta

    @property
    @capture_unsafe(_LAZY_CAPTURE_REASON)
    def root_link_vel_w(self) -> ProxyArray:
        """Root body velocity at the link frame in world frame [m/s, rad/s], shape ``(num_instances, 6)``."""
        if self._root_link_vel_w.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.get_root_link_vel_from_root_com_vel,
                dim=self._num_instances,
                inputs=[
                    self.root_com_vel_w.warp,
                    self.root_link_pose_w.warp,
                    self._sim_bind_body_com_pos_b,
                ],
                outputs=[
                    self._root_link_vel_w.data,
                ],
                device=self.device,
            )
            self._root_link_vel_w.timestamp = self._sim_timestamp

        return self._root_link_vel_w_ta

    @property
    def root_link_lin_vel_w(self) -> ProxyArray:
        """Root body linear velocity at the link frame [m/s], shape ``(num_instances, 3)``."""
        self._root_link_lin_vel_w = self._get_top_from_spatial_vector(
            self._root_link_lin_vel_w, self.root_link_vel_w.warp
        )
        if self._root_link_lin_vel_w_ta is None:
            self._root_link_lin_vel_w_ta = ProxyArray(self._root_link_lin_vel_w)
        return self._root_link_lin_vel_w_ta

    @property
    def root_link_ang_vel_w(self) -> ProxyArray:
        """Root body angular velocity at the link frame [rad/s], shape ``(num_instances, 3)``."""
        self._root_link_ang_vel_w = self._get_bottom_from_spatial_vector(
            self._root_link_ang_vel_w, self.root_link_vel_w.warp
        )
        if self._root_link_ang_vel_w_ta is None:
            self._root_link_ang_vel_w_ta = ProxyArray(self._root_link_ang_vel_w)
        return self._root_link_ang_vel_w_ta

    # Shorthand aliases.
    @property
    def root_pos_w(self) -> ProxyArray:
        """Shorthand for :attr:`root_link_pos_w`."""
        return self.root_link_pos_w

    @property
    def root_quat_w(self) -> ProxyArray:
        """Shorthand for :attr:`root_link_quat_w`."""
        return self.root_link_quat_w

    @property
    def root_vel_w(self) -> ProxyArray:
        """Shorthand for :attr:`root_link_vel_w`."""
        return self.root_link_vel_w

    @property
    def root_lin_vel_w(self) -> ProxyArray:
        """Shorthand for :attr:`root_link_lin_vel_w`."""
        return self.root_link_lin_vel_w

    @property
    def root_ang_vel_w(self) -> ProxyArray:
        """Shorthand for :attr:`root_link_ang_vel_w`."""
        return self.root_link_ang_vel_w

    # ------------------------------------------------------------------
    # Per-segment body state
    # ------------------------------------------------------------------

    @property
    def body_link_pose_w(self) -> ProxyArray:
        """Per-segment body pose in world frame [m, quat], shape ``(num_instances, num_bodies, 7)``."""
        self._ensure_fk_fresh()
        return self._body_link_pose_w_ta

    @property
    def body_link_pos_w(self) -> ProxyArray:
        """Per-segment body position in world frame [m], shape ``(num_instances, num_bodies, 3)``."""
        self._body_link_pos_w = self._get_pos_from_transform(self._body_link_pos_w, self.body_link_pose_w.warp)
        if self._body_link_pos_w_ta is None:
            self._body_link_pos_w_ta = ProxyArray(self._body_link_pos_w)
        return self._body_link_pos_w_ta

    @property
    def body_link_quat_w(self) -> ProxyArray:
        """Per-segment body orientation ``(x, y, z, w)``, shape ``(num_instances, num_bodies, 4)``."""
        self._body_link_quat_w = self._get_quat_from_transform(self._body_link_quat_w, self.body_link_pose_w.warp)
        if self._body_link_quat_w_ta is None:
            self._body_link_quat_w_ta = ProxyArray(self._body_link_quat_w)
        return self._body_link_quat_w_ta

    @property
    @capture_unsafe(_LAZY_CAPTURE_REASON)
    def body_link_vel_w(self) -> ProxyArray:
        """Per-segment body velocity at the link frame [m/s, rad/s], shape ``(num_instances, num_bodies, 6)``."""
        if self._body_link_vel_w.timestamp < self._sim_timestamp:
            wp.launch(
                shared_kernels.get_body_link_vel_from_body_com_vel,
                dim=(self._num_instances, self._num_bodies),
                inputs=[
                    self.body_com_vel_w.warp,
                    self.body_link_pose_w.warp,
                    self._sim_bind_body_com_pos_b,
                ],
                outputs=[
                    self._body_link_vel_w.data,
                ],
                device=self.device,
            )
            self._body_link_vel_w.timestamp = self._sim_timestamp

        return self._body_link_vel_w_ta

    @property
    def body_link_lin_vel_w(self) -> ProxyArray:
        """Per-segment body linear velocity at the link frame [m/s], shape ``(num_instances, num_bodies, 3)``."""
        self._body_link_lin_vel_w = self._get_top_from_spatial_vector(
            self._body_link_lin_vel_w, self.body_link_vel_w.warp
        )
        if self._body_link_lin_vel_w_ta is None:
            self._body_link_lin_vel_w_ta = ProxyArray(self._body_link_lin_vel_w)
        return self._body_link_lin_vel_w_ta

    @property
    def body_link_ang_vel_w(self) -> ProxyArray:
        """Per-segment body angular velocity at the link frame [rad/s], shape ``(num_instances, num_bodies, 3)``."""
        self._body_link_ang_vel_w = self._get_bottom_from_spatial_vector(
            self._body_link_ang_vel_w, self.body_link_vel_w.warp
        )
        if self._body_link_ang_vel_w_ta is None:
            self._body_link_ang_vel_w_ta = ProxyArray(self._body_link_ang_vel_w)
        return self._body_link_ang_vel_w_ta

    @property
    def body_com_vel_w(self) -> ProxyArray:
        """Per-segment body velocity at the COM [m/s, rad/s], shape ``(num_instances, num_bodies, 6)``."""
        return self._body_com_vel_w_ta

    @property
    def body_com_lin_vel_w(self) -> ProxyArray:
        """Per-segment body linear velocity at the COM [m/s], shape ``(num_instances, num_bodies, 3)``."""
        self._body_com_lin_vel_w = self._get_top_from_spatial_vector(self._body_com_lin_vel_w, self.body_com_vel_w.warp)
        if self._body_com_lin_vel_w_ta is None:
            self._body_com_lin_vel_w_ta = ProxyArray(self._body_com_lin_vel_w)
        return self._body_com_lin_vel_w_ta

    @property
    def body_com_ang_vel_w(self) -> ProxyArray:
        """Per-segment body angular velocity at the COM [rad/s], shape ``(num_instances, num_bodies, 3)``."""
        self._body_com_ang_vel_w = self._get_bottom_from_spatial_vector(
            self._body_com_ang_vel_w, self.body_com_vel_w.warp
        )
        if self._body_com_ang_vel_w_ta is None:
            self._body_com_ang_vel_w_ta = ProxyArray(self._body_com_ang_vel_w)
        return self._body_com_ang_vel_w_ta

    # ------------------------------------------------------------------
    # Internal cable joint state
    # ------------------------------------------------------------------

    @property
    def joint_pos(self) -> ProxyArray:
        """Cable joint positions [m or rad, depending on joint type], shape ``(num_instances, num_joints)``."""
        return self._joint_pos_ta

    @property
    def joint_vel(self) -> ProxyArray:
        """Cable joint velocities [m/s or rad/s, depending on joint type], shape ``(num_instances, num_joints)``."""
        return self._joint_vel_ta

    @property
    def joint_acc(self) -> ProxyArray:
        """Cable joint accelerations (finite difference) [m/s² or rad/s², depending on joint type].

        Shape ``(num_instances, num_joints)``.
        """
        if self._joint_acc.timestamp < self._sim_timestamp:
            # note: we use finite differencing to compute acceleration
            time_elapsed = self._sim_timestamp - self._joint_acc.timestamp
            wp.launch(
                articulation_kernels.get_joint_acc_from_joint_vel,
                dim=(self._num_instances, self._num_joints),
                inputs=[
                    self.joint_vel.warp,
                    self._previous_joint_vel,
                    time_elapsed,
                ],
                outputs=[
                    self._joint_acc.data,
                ],
                device=self.device,
            )
            self._joint_acc.timestamp = self._sim_timestamp
        return self._joint_acc_ta

    # ------------------------------------------------------------------
    # Internal: sim bindings + buffers
    # ------------------------------------------------------------------

    def _create_simulation_bindings(self) -> None:
        """Create simulation bindings for the cable's curated surface.

        Called once at init and again on :attr:`PhysicsEvent.PHYSICS_READY`
        whenever the solver fully resets its buffers. Mirrors the equivalent
        method on :class:`ArticulationData`, restricted to the six direct
        fields the cable exposes plus ``body_com_pos_b`` (consumed internally
        by the COM→link velocity kernels).
        """
        view = self._root_view_ref()
        # Short-hand for the number of instances, links, and joints.
        self._num_instances = view.count
        self._num_bodies = view.link_count
        self._num_joints = view.joint_dof_count

        # -- root properties
        self._sim_bind_root_link_pose_w = view.get_root_transforms(SimulationManager.get_state_0())[:, 0]
        root_vel_w = view.get_root_velocities(SimulationManager.get_state_0())
        if root_vel_w is not None:
            if view.is_fixed_base:
                self._sim_bind_root_com_vel_w = root_vel_w[:, 0, 0]
            else:
                self._sim_bind_root_com_vel_w = root_vel_w[:, 0]
        # -- body properties
        self._sim_bind_body_com_pos_b = view.get_attribute("body_com", SimulationManager.get_model())[:, 0]
        self._sim_bind_body_link_pose_w = view.get_link_transforms(SimulationManager.get_state_0())[:, 0]
        body_com_vel_w = view.get_link_velocities(SimulationManager.get_state_0())
        if body_com_vel_w is not None:
            self._sim_bind_body_com_vel_w = body_com_vel_w[:, 0]
        # -- joint state (cables always have joints — rod_graph emits one DoF
        #    per segment, so the ``num_joints == 0`` branch from ArticulationData
        #    does not apply here).
        self._sim_bind_joint_pos = view.get_dof_positions(SimulationManager.get_state_0())[:, 0]
        self._sim_bind_joint_vel = view.get_dof_velocities(SimulationManager.get_state_0())[:, 0]

        # Allocate / re-allocate fixed-shape lazy buffers and finite-diff
        # history. Re-doing the allocation on every rebind keeps the buffers
        # paired with the current solver state.
        if root_vel_w is None:
            # Fixed-base safety net (matches ArticulationData) — the root_vel
            # sim binding does not exist, so back the field with zeros so the
            # public property still returns a meaningful array.
            self._sim_bind_root_com_vel_w = wp.zeros(
                (self._num_instances,), dtype=wp.spatial_vectorf, device=self.device
            )
        if body_com_vel_w is None:
            self._sim_bind_body_com_vel_w = wp.zeros(
                (self._num_instances, self._num_bodies), dtype=wp.spatial_vectorf, device=self.device
            )

        self._root_link_vel_w = TimestampedBuffer(
            shape=(self._num_instances,), dtype=wp.spatial_vectorf, device=self.device
        )
        self._body_link_vel_w = TimestampedBuffer(
            shape=(self._num_instances, self._num_bodies), dtype=wp.spatial_vectorf, device=self.device
        )
        self._joint_acc = TimestampedBuffer(
            shape=(self._num_instances, self._num_joints), dtype=wp.float32, device=self.device
        )
        # Finite-difference history for joint accelerations — clone the current
        # joint velocities so the very first ``joint_acc`` access yields zero
        # (instead of a spurious vel / dt impulse).
        self._previous_joint_vel = wp.clone(self._sim_bind_joint_vel)

        # Re-pin all ProxyArray wrappers to the freshly created sim bindings
        # and buffers.
        self._pin_proxy_arrays()

    def _pin_proxy_arrays(self) -> None:
        """Create or rebind all pinned :class:`ProxyArray` wrappers.

        On first init the backing fields are None; on rebind (after a full
        solver reset) we reset stale pointers into freed transform memory so
        the next access of a sliced lazy property re-derives its view from
        the fresh sim binding.
        """
        # Direct sim bindings → pinned ProxyArrays.
        self._root_link_pose_w_ta = ProxyArray(self._sim_bind_root_link_pose_w)
        self._root_com_vel_w_ta = ProxyArray(self._sim_bind_root_com_vel_w)
        self._body_link_pose_w_ta = ProxyArray(self._sim_bind_body_link_pose_w)
        self._body_com_vel_w_ta = ProxyArray(self._sim_bind_body_com_vel_w)
        self._joint_pos_ta = ProxyArray(self._sim_bind_joint_pos)
        self._joint_vel_ta = ProxyArray(self._sim_bind_joint_vel)

        # Lazy buffers → pinned ProxyArrays.
        self._root_link_vel_w_ta = ProxyArray(self._root_link_vel_w.data)
        self._body_link_vel_w_ta = ProxyArray(self._body_link_vel_w.data)
        self._joint_acc_ta = ProxyArray(self._joint_acc.data)

        # Invalidate sliced lazy ProxyArrays + their backing wp.arrays so the
        # next property access re-creates them off the fresh transform /
        # spatial-vector memory. On first init these are already None (set in
        # __init__); on rebind this resets stale strided views into freed sim
        # memory.
        self._root_link_pos_w_ta = None
        self._root_link_pos_w = None
        self._root_link_quat_w_ta = None
        self._root_link_quat_w = None
        self._root_link_lin_vel_w_ta = None
        self._root_link_lin_vel_w = None
        self._root_link_ang_vel_w_ta = None
        self._root_link_ang_vel_w = None
        self._body_link_pos_w_ta = None
        self._body_link_pos_w = None
        self._body_link_quat_w_ta = None
        self._body_link_quat_w = None
        self._body_link_lin_vel_w_ta = None
        self._body_link_lin_vel_w = None
        self._body_link_ang_vel_w_ta = None
        self._body_link_ang_vel_w = None
        self._body_com_lin_vel_w_ta = None
        self._body_com_lin_vel_w = None
        self._body_com_ang_vel_w_ta = None
        self._body_com_ang_vel_w = None

    # ------------------------------------------------------------------
    # Internal helpers (copied from ArticulationData verbatim)
    # ------------------------------------------------------------------

    def _get_pos_from_transform(self, source: wp.array | None, transform: wp.array) -> wp.array:
        """Generate a position array from a transform array.

        Args:
            source: Existing destination array; ``None`` to derive a fresh view.
            transform: The transform array. Shape is ``(N,)`` dtype=wp.transformf.

        Returns:
            The position array. Shape is ``(N,)`` dtype=wp.vec3f.
        """
        if source is None:
            if transform.is_contiguous:
                return wp.array(
                    ptr=transform.ptr,
                    shape=transform.shape,
                    dtype=wp.vec3f,
                    strides=transform.strides,
                    device=self.device,
                )
            else:
                source = wp.zeros(transform.shape, dtype=wp.vec3f, device=self.device)

        if not transform.is_contiguous:
            if len(transform.shape) > 1:
                wp.launch(
                    shared_kernels.split_transform_to_pos_2d,
                    dim=transform.shape,
                    inputs=[transform],
                    outputs=[source],
                    device=self.device,
                )
            else:
                wp.launch(
                    shared_kernels.split_transform_to_pos_1d,
                    dim=transform.shape,
                    inputs=[transform],
                    outputs=[source],
                    device=self.device,
                )
        return source

    def _get_quat_from_transform(self, source: wp.array | None, transform: wp.array) -> wp.array:
        """Generate a quaternion array from a transform array.

        Args:
            source: Existing destination array; ``None`` to derive a fresh view.
            transform: The transform array. Shape is ``(N,)`` dtype=wp.transformf.

        Returns:
            The quaternion array. Shape is ``(N,)`` dtype=wp.quatf.
        """
        if source is None:
            if transform.is_contiguous:
                return wp.array(
                    ptr=transform.ptr + 3 * 4,
                    shape=transform.shape,
                    dtype=wp.quatf,
                    strides=transform.strides,
                    device=self.device,
                )
            else:
                source = wp.zeros(transform.shape, dtype=wp.quatf, device=self.device)

        if not transform.is_contiguous:
            if len(transform.shape) > 1:
                wp.launch(
                    shared_kernels.split_transform_to_quat_2d,
                    dim=transform.shape,
                    inputs=[transform],
                    outputs=[source],
                    device=self.device,
                )
            else:
                wp.launch(
                    shared_kernels.split_transform_to_quat_1d,
                    dim=transform.shape,
                    inputs=[transform],
                    outputs=[source],
                    device=self.device,
                )
        return source

    def _get_top_from_spatial_vector(self, source: wp.array | None, spatial_vector: wp.array) -> wp.array:
        """Get the top (linear) part of a spatial vector array.

        Args:
            source: Existing destination array; ``None`` to derive a fresh view.
            spatial_vector: The spatial vector array. Shape is ``(N,)`` dtype=wp.spatial_vectorf.

        Returns:
            The top part of the spatial vector array. Shape is ``(N,)`` dtype=wp.vec3f.
        """
        if source is None:
            if spatial_vector.is_contiguous:
                return wp.array(
                    ptr=spatial_vector.ptr,
                    shape=spatial_vector.shape,
                    dtype=wp.vec3f,
                    strides=spatial_vector.strides,
                    device=self.device,
                )
            else:
                source = wp.zeros(spatial_vector.shape, dtype=wp.vec3f, device=self.device)

        if not spatial_vector.is_contiguous:
            if len(spatial_vector.shape) > 1:
                wp.launch(
                    shared_kernels.split_spatial_vector_to_top_2d,
                    dim=spatial_vector.shape,
                    inputs=[spatial_vector],
                    outputs=[source],
                    device=self.device,
                )
            else:
                wp.launch(
                    shared_kernels.split_spatial_vector_to_top_1d,
                    dim=spatial_vector.shape,
                    inputs=[spatial_vector],
                    outputs=[source],
                    device=self.device,
                )
        return source

    def _get_bottom_from_spatial_vector(self, source: wp.array | None, spatial_vector: wp.array) -> wp.array:
        """Get the bottom (angular) part of a spatial vector array.

        Args:
            source: Existing destination array; ``None`` to derive a fresh view.
            spatial_vector: The spatial vector array. Shape is ``(N,)`` dtype=wp.spatial_vectorf.

        Returns:
            The bottom part of the spatial vector array. Shape is ``(N,)`` dtype=wp.vec3f.
        """
        if source is None:
            if spatial_vector.is_contiguous:
                return wp.array(
                    ptr=spatial_vector.ptr + 3 * 4,
                    shape=spatial_vector.shape,
                    dtype=wp.vec3f,
                    strides=spatial_vector.strides,
                    device=self.device,
                )
            else:
                source = wp.zeros(spatial_vector.shape, dtype=wp.vec3f, device=self.device)

        if not spatial_vector.is_contiguous:
            if len(spatial_vector.shape) > 1:
                wp.launch(
                    shared_kernels.split_spatial_vector_to_bottom_2d,
                    dim=spatial_vector.shape,
                    inputs=[spatial_vector],
                    outputs=[source],
                    device=self.device,
                )
            else:
                wp.launch(
                    shared_kernels.split_spatial_vector_to_bottom_1d,
                    dim=spatial_vector.shape,
                    inputs=[spatial_vector],
                    outputs=[source],
                    device=self.device,
                )
        return source
