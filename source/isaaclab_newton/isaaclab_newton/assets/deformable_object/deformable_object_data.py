# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import weakref

import warp as wp

from isaaclab.assets.deformable_object.base_deformable_object_data import BaseDeformableObjectData
from isaaclab.utils.buffers import TimestampedBufferWarp as TimestampedBuffer

from .kernels import compute_mean_vec3f_over_vertices, compute_nodal_state_w, read_particles_to_nodal_buffer, vec6f


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

    def __init__(self, root_view, device: str, num_instances: int, max_sim_vertices: int,
                 particle_start_indices: wp.array):
        """Initialize the Newton deformable object data.

        Args:
            root_view: The root deformable body view of the object. This is None for Newton
                since no SoftBodyView exists; kept for API compatibility with the base class.
            device: The device used for processing.
            num_instances: Number of deformable body instances (environments).
            max_sim_vertices: Maximum number of simulation mesh vertices per body.
            particle_start_indices: Per-instance start index into Newton's flat particle
                arrays. Shape is (num_instances,) with dtype int32.
        """
        super().__init__(root_view, device)
        # Store as weak reference if a view is provided (for API compatibility)
        self._root_view = weakref.proxy(root_view) if root_view is not None else None

        # Store dimensions
        self._num_instances = num_instances
        self._max_sim_vertices = max_sim_vertices
        self._particle_start_indices = particle_start_indices

        # Set initial time stamp
        self._sim_timestamp = 0.0

        # References to Newton state arrays (populated by _create_simulation_bindings)
        self._sim_bind_particle_q: wp.array | None = None
        self._sim_bind_particle_qd: wp.array | None = None

        # Initialize the lazy buffers
        # -- node state in simulation world frame
        self._nodal_pos_w = TimestampedBuffer((num_instances, max_sim_vertices), device, wp.vec3f)
        self._nodal_vel_w = TimestampedBuffer((num_instances, max_sim_vertices), device, wp.vec3f)
        self._nodal_state_w = TimestampedBuffer((num_instances, max_sim_vertices), device, vec6f)
        # -- derived: root pos/vel (mean over vertices)
        self._root_pos_w = TimestampedBuffer((num_instances,), device, wp.vec3f)
        self._root_vel_w = TimestampedBuffer((num_instances,), device, wp.vec3f)

    def _create_simulation_bindings(self, particle_q: wp.array, particle_qd: wp.array) -> None:
        """Store references to Newton state particle arrays for lazy reads.

        This method should be called after Newton's model/state is built so that the
        data container can read particle positions and velocities on demand.

        .. note::
            TODO: This needs runtime validation with an actual Newton XPBD soft body
            simulation once the full solver integration is in place.

        Args:
            particle_q: Newton state particle positions [m]. Flat array of shape
                (total_particles,) with dtype vec3f.
            particle_qd: Newton state particle velocities [m/s]. Flat array of shape
                (total_particles,) with dtype vec3f.
        """
        self._sim_bind_particle_q = particle_q
        self._sim_bind_particle_qd = particle_qd

    def update(self, dt: float) -> None:
        """Update the data for the deformable object.

        Args:
            dt: The time step for the update [s]. This must be a positive value.
        """
        # update the simulation timestamp
        self._sim_timestamp += dt

    ##
    # Defaults.
    ##

    default_nodal_state_w: wp.array = None
    """Default nodal state ``[nodal_pos, nodal_vel]`` in simulation world frame.
    Shape is (num_instances, max_sim_vertices_per_body) with dtype vec6f.
    """

    ##
    # Kinematic commands.
    ##

    nodal_kinematic_target: wp.array = None
    """Simulation mesh kinematic targets for the deformable bodies.
    Shape is (num_instances, max_sim_vertices_per_body) with dtype vec4f.

    The kinematic targets are used to drive the simulation mesh vertices to the target positions.
    The targets are stored as (x, y, z, is_not_kinematic) where "is_not_kinematic" is a binary
    flag indicating whether the vertex is kinematic or not. The flag is set to 0 for kinematic vertices
    and 1 for non-kinematic vertices.
    """

    ##
    # Properties.
    ##

    @property
    def nodal_pos_w(self) -> wp.array:
        """Nodal positions in simulation world frame [m].

        Shape is (num_instances, max_sim_vertices_per_body) with dtype vec3f.
        """
        if self._nodal_pos_w.timestamp < self._sim_timestamp:
            if self._sim_bind_particle_q is not None:
                # Read from Newton's flat particle array into per-instance buffer
                wp.launch(
                    read_particles_to_nodal_buffer,
                    dim=(self._num_instances, self._max_sim_vertices),
                    inputs=[
                        self._sim_bind_particle_q,
                        self._particle_start_indices,
                        self._max_sim_vertices,
                    ],
                    outputs=[self._nodal_pos_w.data],
                    device=self.device,
                )
            self._nodal_pos_w.timestamp = self._sim_timestamp
        return self._nodal_pos_w.data

    @property
    def nodal_vel_w(self) -> wp.array:
        """Nodal velocities in simulation world frame [m/s].

        Shape is (num_instances, max_sim_vertices_per_body) with dtype vec3f.
        """
        if self._nodal_vel_w.timestamp < self._sim_timestamp:
            if self._sim_bind_particle_qd is not None:
                # Read from Newton's flat particle velocity array into per-instance buffer
                wp.launch(
                    read_particles_to_nodal_buffer,
                    dim=(self._num_instances, self._max_sim_vertices),
                    inputs=[
                        self._sim_bind_particle_qd,
                        self._particle_start_indices,
                        self._max_sim_vertices,
                    ],
                    outputs=[self._nodal_vel_w.data],
                    device=self.device,
                )
            self._nodal_vel_w.timestamp = self._sim_timestamp
        return self._nodal_vel_w.data

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

    ##
    # Derived properties.
    ##

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
