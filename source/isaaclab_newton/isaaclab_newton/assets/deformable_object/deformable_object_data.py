# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
import warp as wp

from isaaclab.assets.deformable_object.base_deformable_object_data import BaseDeformableObjectData
from isaaclab.utils.buffers import TimestampedBufferWarp as TimestampedBuffer
from isaaclab.utils.math import normalize

from isaaclab_newton.physics import NewtonManager as SimulationManager

from .kernels import compute_mean_vec3f_over_vertices, compute_nodal_state_w, vec6f


class DeformableObjectData(BaseDeformableObjectData):
    """Data container for a Newton deformable object.

    This class contains the data for a deformable object simulated via Newton's XPBD solver.
    Newton represents soft bodies as particles: positions are stored in ``state.particle_q``
    and velocities in ``state.particle_qd``, both as flat arrays over all particles in the
    scene. Each deformable instance owns a contiguous slice of that flat array.

    The data is lazily updated, meaning that the data is only updated when it is accessed.
    This is useful when the data is expensive to compute or retrieve. The data is updated
    when the timestamp of the buffer is older than the current simulation timestamp. The
    timestamp is updated whenever the data is updated.

    .. note::
        Newton does not expose a ``SoftBodyView`` abstraction. Instead, particle data is
        accessed via index-based mapping from the global particle arrays in the Newton model
        and state objects.
    """

    __backend_name__: str = "newton"
    """The name of the backend for the deformable object data."""

    def __init__(
        self,
        device: str,
    ):
        """Initialize the Newton deformable object data.

        Args:
            device: The device used for processing.
        """
        super().__init__(None, device)
        self._root_view = None

        model = SimulationManager.get_model()
        self._num_instances = model.world_count # TODO: This should be the count of deformable bodies, not just the world count

        # Compute per-world particle count from particle_world_start.
        # particle_world_start has shape (world_count + 2,): entries 0..world_count-1
        # are per-world start indices, second-last is global particle start, last is total.
        pws = model.particle_world_start.numpy()
        counts = np.diff(pws[:-1])
        self._max_sim_vertices = np.max(counts)

        # Primed flag -- once True, default buffers cannot be reassigned
        self._is_primed = False

        # Set initial time stamp
        self._sim_timestamp = 0.0

        # Convert to direction vector
        gravity = wp.to_torch(SimulationManager.get_model().gravity)[0]
        gravity_dir = torch.tensor((gravity[0], gravity[1], gravity[2]), device=self.device)
        gravity_dir = normalize(gravity_dir.unsqueeze(0)).squeeze(0)
        gravity_dir = gravity_dir.repeat(self._num_instances, 1)
        forward_vec = torch.tensor((1.0, 0.0, 0.0), device=self.device).repeat(self._num_instances, 1)

        # Initialize constants
        self.GRAVITY_VEC_W = wp.from_torch(gravity_dir, dtype=wp.vec3f)
        self.FORWARD_VEC_B = wp.from_torch(forward_vec, dtype=wp.vec3f)

        self._create_simulation_bindings()
        self._create_buffers()

    """
    Properties - Primed state.
    """

    @property
    def is_primed(self) -> bool:
        """Whether the deformable object data is fully instantiated and ready to use."""
        return self._is_primed

    @is_primed.setter
    def is_primed(self, value: bool) -> None:
        """Set whether the deformable object data is fully instantiated and ready to use.

        .. note::
            Once this quantity is set to True, it cannot be changed.

        Args:
            value: The primed state.

        Raises:
            ValueError: If the deformable object data is already primed.
        """
        if self._is_primed:
            raise ValueError("The deformable object data is already primed.")
        self._is_primed = value

    def update(self, dt: float) -> None:
        """Update the data for the deformable object.

        Args:
            dt: The time step for the update [s]. This must be a positive value.
        """
        # update the simulation timestamp
        self._sim_timestamp += dt

    """
    Defaults.
    """

    @property
    def default_nodal_state_w(self) -> wp.array:
        """Default nodal state ``[nodal_pos, nodal_vel]`` in simulation world frame.

        Shape is (num_instances, max_sim_vertices_per_body) with dtype vec6f.
        """
        return self._default_nodal_state_w

    @default_nodal_state_w.setter
    def default_nodal_state_w(self, value: wp.array) -> None:
        """Set the default nodal state.

        Args:
            value: The default nodal state.

        Raises:
            ValueError: If the deformable object data is already primed.
        """
        if self._is_primed:
            raise ValueError("The deformable object data is already primed.")
        self._default_nodal_state_w = value

    """
    Kinematic commands.
    """

    @property
    def nodal_kinematic_target(self) -> wp.array:
        """Simulation mesh kinematic targets for the deformable bodies.

        Shape is (num_instances, max_sim_vertices_per_body) with dtype vec4f.

        The kinematic targets are used to drive the simulation mesh vertices to the target positions.
        The targets are stored as (x, y, z, is_not_kinematic) where ``is_not_kinematic`` is a binary
        flag indicating whether the vertex is kinematic or not. The flag is set to 0 for kinematic vertices
        and 1 for non-kinematic vertices.
        """
        return self._nodal_kinematic_target

    @nodal_kinematic_target.setter
    def nodal_kinematic_target(self, value: wp.array) -> None:
        """Set the kinematic targets.

        Args:
            value: The kinematic targets.
        """
        self._nodal_kinematic_target = value

    """
    Properties - Nodal state.
    """

    @property
    def nodal_pos_w(self) -> wp.array:
        """Nodal positions in simulation world frame [m].

        Shape is (num_instances, max_sim_vertices_per_body) with dtype vec3f.

        This is a zero-copy strided view into Newton's ``state.particle_q``.
        """
        return self._sim_bind_particle_q

    @property
    def nodal_vel_w(self) -> wp.array:
        """Nodal velocities in simulation world frame [m/s].

        Shape is (num_instances, max_sim_vertices_per_body) with dtype vec3f.

        This is a zero-copy strided view into Newton's ``state.particle_qd``.
        """
        return self._sim_bind_particle_qd

    @property
    def nodal_state_w(self) -> wp.array:
        """Nodal state ``[nodal_pos, nodal_vel]`` in simulation world frame.

        Shape is (num_instances, max_sim_vertices_per_body) with dtype vec6f.

        Positions are in meters [m] and velocities in meters per second [m/s].
        """
        if self._nodal_state_w.timestamp < self._sim_timestamp:
            wp.launch(
                compute_nodal_state_w,
                dim=(self._num_instances, self._max_sim_vertices),
                inputs=[self.nodal_pos_w, self.nodal_vel_w],
                outputs=[self._nodal_state_w.data],
                device=self.device,
            )
            self._nodal_state_w.timestamp = self._sim_timestamp
        return self._nodal_state_w.data

    """
    Derived properties.
    """

    @property
    def root_pos_w(self) -> wp.array:
        """Root position from nodal positions of the simulation mesh for the deformable bodies
        in simulation world frame [m]. Shape is (num_instances,) with dtype vec3f.

        This quantity is computed as the mean of the nodal positions.
        """
        if self._root_pos_w.timestamp < self._sim_timestamp:
            wp.launch(
                compute_mean_vec3f_over_vertices,
                dim=(self._num_instances,),
                inputs=[self.nodal_pos_w, self._max_sim_vertices],
                outputs=[self._root_pos_w.data],
                device=self.device,
            )
            self._root_pos_w.timestamp = self._sim_timestamp
        return self._root_pos_w.data

    @property
    def root_vel_w(self) -> wp.array:
        """Root velocity from vertex velocities for the deformable bodies in simulation world
        frame [m/s]. Shape is (num_instances,) with dtype vec3f.

        This quantity is computed as the mean of the nodal velocities.
        """
        if self._root_vel_w.timestamp < self._sim_timestamp:
            wp.launch(
                compute_mean_vec3f_over_vertices,
                dim=(self._num_instances,),
                inputs=[self.nodal_vel_w, self._max_sim_vertices],
                outputs=[self._root_vel_w.data],
                device=self.device,
            )
            self._root_vel_w.timestamp = self._sim_timestamp
        return self._root_vel_w.data
    

    def _create_simulation_bindings(self) -> None:
        """Create strided views into Newton's flat particle arrays.

        Creates ``wp.array`` views with shape ``(num_instances, particles_per_world)`` pointing
        directly into ``state.particle_q`` and ``state.particle_qd`` via pointer arithmetic.
        Since all worlds are cloned from the same prototype, particle counts are uniform and
        the flat arrays are contiguous per world, allowing zero-copy reshaping.

        This method is called during init and re-called on :attr:`PhysicsEvent.PHYSICS_READY`
        when Newton recreates its model/state on full reset.
        """
        model = SimulationManager.get_model()
        state = SimulationManager.get_state_0()

        pws = model.particle_world_start.numpy()
        offset = int(pws[0])
        ppw = self._max_sim_vertices  # particles per world

        # Verify all worlds have the same particle count
        for w in range(self._num_instances):
            count = int(pws[w + 1]) - int(pws[w])
            assert count == ppw, (
                f"World {w} has {count} particles but expected {ppw}. "
                "All worlds must have the same particle count (cloned from prototype)."
            )

        # Strided view: flat particle_q → (num_instances, particles_per_world) vec3f
        flat_q = state.particle_q
        q_stride = flat_q.strides[0]
        self._sim_bind_particle_q = wp.array(
            ptr=int(flat_q.ptr) + offset * q_stride,
            dtype=wp.vec3f,
            shape=(self._num_instances, ppw),
            strides=(ppw * q_stride, q_stride),
            device=self.device,
            copy=False,
        )

        # Strided view: flat particle_qd → (num_instances, particles_per_world) vec3f
        flat_qd = state.particle_qd
        qd_stride = flat_qd.strides[0]
        self._sim_bind_particle_qd = wp.array(
            ptr=int(flat_qd.ptr) + offset * qd_stride,
            dtype=wp.vec3f,
            shape=(self._num_instances, ppw),
            strides=(ppw * qd_stride, qd_stride),
            device=self.device,
            copy=False,
        )

    def _create_buffers(self) -> None:
        """Create the buffers for the deformable object data."""
        # -- default state (set during _create_buffers, locked after priming)
        self._default_nodal_state_w = wp.zeros(
            (self._num_instances, self._max_sim_vertices), dtype=vec6f, device=self.device
        )
        # -- kinematic targets
        self._nodal_kinematic_target = wp.zeros(
            (self._num_instances, self._max_sim_vertices), dtype=wp.vec4f, device=self.device
        )
        # -- derived lazy buffers (nodal_pos_w and nodal_vel_w are direct pointer views,
        #    so they don't need TimestampedBuffers)
        self._nodal_state_w = TimestampedBuffer(
            (self._num_instances, self._max_sim_vertices), self.device, vec6f
        )
        self._root_pos_w = TimestampedBuffer((self._num_instances,), self.device, wp.vec3f)
        self._root_vel_w = TimestampedBuffer((self._num_instances,), self.device, wp.vec3f)