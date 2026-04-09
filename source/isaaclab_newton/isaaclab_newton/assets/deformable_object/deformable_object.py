# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.physics import PhysicsEvent
import numpy as np
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets.deformable_object.base_deformable_object import BaseDeformableObject
from isaaclab.markers import VisualizationMarkers
from isaaclab_newton.physics import NewtonManager as SimulationManager


from .deformable_object_data import DeformableObjectData
from .kernels import (
    compute_nodal_state_w,
    set_kinematic_flags_to_one,
    vec6f,
    write_nodal_vec3f_to_buffer,
    write_nodal_vec4f_to_buffer,
)

if TYPE_CHECKING:
    from isaaclab.assets.deformable_object.deformable_object_cfg import DeformableObjectCfg

logger = logging.getLogger(__name__)


class DeformableObject(BaseDeformableObject):
    """A deformable object asset class for the Newton backend.

    Deformable objects are assets that can be deformed in the simulation. They are typically used for
    soft bodies, such as stuffed animals and food items.

    Newton simulates soft bodies via particles using an XPBD solver. Particle positions are stored in
    ``state.particle_q`` and velocities in ``state.particle_qd`` as flat arrays over all particles in
    the scene. Each deformable body instance owns a contiguous slice of that flat array, identified by
    a start index and a vertex count.

    .. attention::
        This class is experimental and subject to change due to changes on the underlying Newton API
        on which it depends. The XPBD soft body solver integration is not yet complete --
        :meth:`_initialize_impl` will raise :class:`NotImplementedError` until particle index mapping
        from USD prims to the Newton model is implemented.
    """

    cfg: DeformableObjectCfg
    """Configuration instance for the deformable object."""

    __backend_name__: str = "newton"
    """The name of the backend for the deformable object."""

    def __init__(self, cfg: DeformableObjectCfg):
        """Initialize the deformable object.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)
        # Register custom vec6f type for nodal state validation.
        self._DTYPE_TO_TORCH_TRAILING_DIMS = {**self._DTYPE_TO_TORCH_TRAILING_DIMS, vec6f: (6,)}

    """
    Properties
    """

    @property
    def data(self) -> DeformableObjectData:
        return self._data

    @property
    def num_instances(self) -> int:
        return self.data._num_instances

    @property
    def num_bodies(self) -> int:
        """Number of bodies in the asset.

        This is always 1 since each object is a single deformable body.
        """
        return 1

    @property
    def root_view(self) -> None:
        """Deformable body view for the asset.

        Newton does not provide a SoftBodyView abstraction. Particle data is accessed
        directly via index-based mapping from the Newton model and state objects.

        Returns:
            None, as no view abstraction exists in Newton for soft bodies.
        """
        return None

    @property
    def max_sim_vertices_per_body(self) -> int:
        """The maximum number of simulation mesh vertices per deformable body."""
        return self.data._max_sim_vertices

    @property
    def max_sim_elements_per_body(self) -> int:
        """The maximum number of simulation mesh elements per deformable body."""
        raise NotImplementedError

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None) -> None:
        """Reset the deformable object.

        Resets particle positions and velocities to their default values for the specified
        environments. If no environment indices or mask are provided, all environments are reset.

        Args:
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. If None, then all the instances are updated.
                Shape is (num_instances,).
        """
        # resolve env_ids
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        elif env_ids is None or (isinstance(env_ids, slice) and env_ids == slice(None)):
            env_ids = self._ALL_INDICES

        # reset nodal state to defaults
        self.write_nodal_state_to_sim_index(self.data.default_nodal_state_w, env_ids=env_ids, full_data=True)

    def write_data_to_sim(self) -> None:
        """Write data to the simulation.

        Currently a no-op for Newton deformable objects. External forces on deformable
        bodies are not yet supported through this path.
        """
        pass

    def update(self, dt: float) -> None:
        """Update the simulation data.

        Args:
            dt: The time step size [s].
        """
        self.data.update(dt)

    """
    Operations - Write to simulation.
    """

    def write_nodal_state_to_sim_index(
        self,
        nodal_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set the nodal state over selected environment indices into the simulation.

        The nodal state comprises of the nodal positions and velocities. Since these are nodes,
        the velocity only has a translational component. All the quantities are in the simulation
        frame.

        Args:
            nodal_state: Nodal state in simulation frame.
                Shape is (len(env_ids), max_sim_vertices_per_body, 6) or
                (num_instances, max_sim_vertices_per_body, 6).
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        # Convert warp to torch if needed
        if isinstance(nodal_state, wp.array):
            nodal_state = wp.to_torch(nodal_state)
        # set into simulation
        self.write_nodal_pos_to_sim_index(nodal_state[..., :3], env_ids=env_ids, full_data=full_data)
        self.write_nodal_velocity_to_sim_index(nodal_state[..., 3:], env_ids=env_ids, full_data=full_data)

    def write_nodal_state_to_sim_mask(
        self,
        nodal_state: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the nodal state over selected environment mask into the simulation.

        The nodal state comprises of the nodal positions and velocities. Since these are nodes,
        the velocity only has a translational component. All the quantities are in the simulation
        frame.

        Args:
            nodal_state: Nodal state in simulation frame.
                Shape is (num_instances, max_sim_vertices_per_body, 6).
            env_mask: Environment mask. If None, then all indices are used.
        """
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        self.write_nodal_state_to_sim_index(nodal_state, env_ids=env_ids, full_data=True)

    def write_nodal_pos_to_sim_index(
        self,
        nodal_pos: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set the nodal positions over selected environment indices into the simulation.

        The nodal position comprises of individual nodal positions of the simulation mesh
        for the deformable body. The positions are in the simulation frame.

        Args:
            nodal_pos: Nodal positions in simulation frame [m].
                Shape is (len(env_ids), max_sim_vertices_per_body, 3) or
                (num_instances, max_sim_vertices_per_body, 3).
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        # resolve env_ids
        env_ids = self._resolve_env_ids(env_ids)
        if full_data:
            self.assert_shape_and_dtype(
                nodal_pos, (self.num_instances, self.max_sim_vertices_per_body), wp.vec3f, "nodal_pos"
            )
        else:
            self.assert_shape_and_dtype(
                nodal_pos, (env_ids.shape[0], self.max_sim_vertices_per_body), wp.vec3f, "nodal_pos"
            )
        # convert torch to warp if needed
        if isinstance(nodal_pos, torch.Tensor):
            nodal_pos = wp.from_torch(nodal_pos.contiguous(), dtype=wp.vec3f)
        # Write directly into Newton's particle state. nodal_pos_w is a strided view
        # into state.particle_q with shape (num_instances, particles_per_world).
        wp.launch(
            write_nodal_vec3f_to_buffer,
            dim=(env_ids.shape[0], self.max_sim_vertices_per_body),
            inputs=[nodal_pos, env_ids, full_data],
            outputs=[self.data.nodal_pos_w],
            device=self.device,
        )
        # invalidate derived buffers
        self.data._nodal_state_w.timestamp = -1.0
        self.data._root_pos_w.timestamp = -1.0

    def write_nodal_pos_to_sim_mask(
        self,
        nodal_pos: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the nodal positions over selected environment mask into the simulation.

        The nodal position comprises of individual nodal positions of the simulation mesh
        for the deformable body. The positions are in the simulation frame.

        Args:
            nodal_pos: Nodal positions in simulation frame [m].
                Shape is (num_instances, max_sim_vertices_per_body, 3).
            env_mask: Environment mask. If None, then all indices are used.
        """
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        self.write_nodal_pos_to_sim_index(nodal_pos, env_ids=env_ids, full_data=True)

    def write_nodal_velocity_to_sim_index(
        self,
        nodal_vel: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set the nodal velocity over selected environment indices into the simulation.

        The nodal velocity comprises of individual nodal velocities of the simulation mesh for the
        deformable body. Since these are nodes, the velocity only has a translational component. The
        velocities are in the simulation frame.

        Args:
            nodal_vel: Nodal velocities in simulation frame [m/s].
                Shape is (len(env_ids), max_sim_vertices_per_body, 3) or
                (num_instances, max_sim_vertices_per_body, 3).
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        # resolve env_ids
        env_ids = self._resolve_env_ids(env_ids)
        if full_data:
            self.assert_shape_and_dtype(
                nodal_vel, (self.num_instances, self.max_sim_vertices_per_body), wp.vec3f, "nodal_vel"
            )
        else:
            self.assert_shape_and_dtype(
                nodal_vel, (env_ids.shape[0], self.max_sim_vertices_per_body), wp.vec3f, "nodal_vel"
            )
        # convert torch to warp if needed
        if isinstance(nodal_vel, torch.Tensor):
            nodal_vel = wp.from_torch(nodal_vel.contiguous(), dtype=wp.vec3f)
        # Write directly into Newton's particle state. nodal_vel_w is a strided view
        # into state.particle_qd with shape (num_instances, particles_per_world).
        wp.launch(
            write_nodal_vec3f_to_buffer,
            dim=(env_ids.shape[0], self.max_sim_vertices_per_body),
            inputs=[nodal_vel, env_ids, full_data],
            outputs=[self.data.nodal_vel_w],
            device=self.device,
        )
        # invalidate derived buffers
        self.data._nodal_state_w.timestamp = -1.0
        self.data._root_vel_w.timestamp = -1.0

    def write_nodal_velocity_to_sim_mask(
        self,
        nodal_vel: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the nodal velocity over selected environment mask into the simulation.

        The nodal velocity comprises of individual nodal velocities of the simulation mesh for the
        deformable body. Since these are nodes, the velocity only has a translational component. The
        velocities are in the simulation frame.

        Args:
            nodal_vel: Nodal velocities in simulation frame [m/s].
                Shape is (num_instances, max_sim_vertices_per_body, 3).
            env_mask: Environment mask. If None, then all indices are used.
        """
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        self.write_nodal_velocity_to_sim_index(nodal_vel, env_ids=env_ids, full_data=True)

    def write_nodal_kinematic_target_to_sim_index(
        self,
        targets: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set the kinematic targets of the simulation mesh for the deformable bodies using indices.

        The kinematic targets comprise of individual nodal positions of the simulation mesh for the
        deformable body and a flag indicating whether the node is kinematically driven or not. The
        positions are in the simulation frame.

        .. note::
            The flag is set to 0.0 for kinematically driven nodes and 1.0 for free nodes.

        Args:
            targets: The kinematic targets comprising of nodal positions and flags.
                Shape is (len(env_ids), max_sim_vertices_per_body, 4) or
                (num_instances, max_sim_vertices_per_body, 4).
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        # resolve env_ids
        env_ids = self._resolve_env_ids(env_ids)
        if full_data:
            self.assert_shape_and_dtype(
                targets, (self.num_instances, self.max_sim_vertices_per_body), wp.vec4f, "targets"
            )
        else:
            self.assert_shape_and_dtype(
                targets, (env_ids.shape[0], self.max_sim_vertices_per_body), wp.vec4f, "targets"
            )
        # convert torch to warp if needed, ensuring 2D (num_envs, V, 4) -> (num_envs, V) vec4f
        if isinstance(targets, torch.Tensor):
            if targets.dim() == 2:
                targets = targets.unsqueeze(0)
            targets = wp.from_torch(targets.contiguous(), dtype=wp.vec4f)
        # write into internal buffer via kernel
        wp.launch(
            write_nodal_vec4f_to_buffer,
            dim=(env_ids.shape[0], self.max_sim_vertices_per_body),
            inputs=[targets, env_ids, full_data],
            outputs=[self.data.nodal_kinematic_target],
            device=self.device,
        )
        # Note: Newton XPBD kinematic target application is handled during the solve step.
        # The internal buffer is the source of truth and will be consumed by the solver.

    def write_nodal_kinematic_target_to_sim_mask(
        self,
        targets: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the kinematic targets of the simulation mesh for the deformable bodies using mask.

        The kinematic targets comprise of individual nodal positions of the simulation mesh for the
        deformable body and a flag indicating whether the node is kinematically driven or not. The
        positions are in the simulation frame.

        .. note::
            The flag is set to 0.0 for kinematically driven nodes and 1.0 for free nodes.

        Args:
            targets: The kinematic targets comprising of nodal positions and flags.
                Shape is (num_instances, max_sim_vertices_per_body, 4).
            env_mask: Environment mask. If None, then all indices are used.
        """
        if env_mask is not None:
            env_ids = wp.nonzero(env_mask)
        else:
            env_ids = self._ALL_INDICES
        self.write_nodal_kinematic_target_to_sim_index(targets, env_ids=env_ids, full_data=True)

    """
    Internal helper.
    """

    def _resolve_env_ids(self, env_ids: Sequence[int] | torch.Tensor | wp.array | None) -> wp.array:
        """Resolve environment indices to a warp int32 array.

        Args:
            env_ids: Environment indices. If None, then all indices are used.

        Returns:
            A warp array of environment indices.
        """
        if env_ids is None or (isinstance(env_ids, slice) and env_ids == slice(None)):
            return self._ALL_INDICES
        elif isinstance(env_ids, list):
            return wp.array(env_ids, dtype=wp.int32, device=self.device)
        elif isinstance(env_ids, torch.Tensor):
            return wp.from_torch(env_ids.to(torch.int32), dtype=wp.int32)
        return env_ids

    def _initialize_impl(self) -> None:
        """Initialize the Newton deformable object backend.

        This method maps USD deformable body prims to Newton's flat particle arrays by querying
        the built Newton model for particle counts and body labels. Each deformable instance is
        assigned a contiguous slice of the global particle arrays.

        Raises:
            NotImplementedError: Raised because Newton XPBD soft body solver integration
                is not yet complete. Particle index mapping from USD prims to the Newton model
                requires runtime validation with actual soft body scenes.
        """        
        # obtain global simulation view
        self._physics_sim_view = SimulationManager.get_physics_sim_view()

        # Create data container
        self._data = DeformableObjectData(device=self.device)

        # Register callback to rebind after full resets (model/state recreation)
        self._physics_ready_handle = SimulationManager.register_callback(
            lambda _: self.data._create_simulation_bindings(),
            PhysicsEvent.PHYSICS_READY,
            name=f"deformable_rebind_{self.cfg.prim_path}",
        )

        # Create buffers, apply init_state transform, and update
        self._create_buffers()
        self._process_cfg()
        # TODO: validate_cfg should be added as well
        self.update(0.0)
        self.data.is_primed = True

        # Initialize debug visualization
        if self._debug_vis_handle is None:
            self.set_debug_vis(self.cfg.debug_vis)

    def _clear_callbacks(self) -> None:
        """Clear all registered callbacks, including the physics-ready rebind handle."""
        super()._clear_callbacks()
        if hasattr(self, "_physics_ready_handle") and self._physics_ready_handle is not None:
            self._physics_ready_handle.deregister()
            self._physics_ready_handle = None

    def _create_buffers(self) -> None:
        """Create buffers for storing data."""
        # constants
        self._ALL_INDICES = wp.array(np.arange(self.num_instances, dtype=np.int32), device=self.device)

        # default state
        # Read initial nodal positions from Newton state (via the data container's lazy evaluation)
        nodal_positions = self.data.nodal_pos_w
        nodal_velocities = wp.zeros(
            (self.num_instances, self.max_sim_vertices_per_body), dtype=wp.vec3f, device=self.device
        )
        # compute default nodal state as vec6f
        self.data.default_nodal_state_w = wp.zeros(
            (self.num_instances, self.max_sim_vertices_per_body), dtype=vec6f, device=self.device
        )
        wp.launch(
            compute_nodal_state_w,
            dim=(self.num_instances, self.max_sim_vertices_per_body),
            inputs=[nodal_positions, nodal_velocities],
            outputs=[self.data.default_nodal_state_w],
            device=self.device,
        )

        # kinematic targets -- allocate buffer and set all nodes as non-kinematic by default
        self.data.nodal_kinematic_target = wp.zeros(
            (self.num_instances, self.max_sim_vertices_per_body), dtype=wp.vec4f, device=self.device
        )
        wp.launch(
            set_kinematic_flags_to_one,
            dim=(self.num_instances * self.max_sim_vertices_per_body,),
            inputs=[
                self.data.nodal_kinematic_target.reshape((self.num_instances * self.max_sim_vertices_per_body,))
            ],
            device=self.device,
        )

    def _process_cfg(self) -> None:
        """Post-process configuration parameters.

        Reads :attr:`cfg.init_state.pos` and :attr:`cfg.init_state.rot` and applies the
        transform to the default nodal state so that every instance starts at the configured
        pose. The translation and rotation are broadcast across all instances.
        """
        # Extract init pose from config
        pos = torch.tensor(self.cfg.init_state.pos, dtype=torch.float32, device=self.device)
        quat = torch.tensor(self.cfg.init_state.rot, dtype=torch.float32, device=self.device)

        # Tile to (num_instances, 3) and (num_instances, 4)
        pos = pos.unsqueeze(0).expand(self.num_instances, -1)
        quat = quat.unsqueeze(0).expand(self.num_instances, -1)

        # Skip if both are identity (pos=0, quat=identity)
        is_zero_pos = torch.all(pos == 0.0).item()
        is_identity_quat = torch.allclose(quat, torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device))
        if is_zero_pos and is_identity_quat:
            return

        # Transform the default nodal positions by init_state pose
        default_state_torch = wp.to_torch(self.data.default_nodal_state_w)  # (N, V, 6)
        nodal_pos = default_state_torch[..., :3].clone()  # (N, V, 3)

        # Apply translation and rotation via the base class helper
        nodal_pos = self.transform_nodal_pos(nodal_pos, pos, quat)

        # Write back into default state
        default_state_torch[..., :3] = nodal_pos
        self.data.default_nodal_state_w = wp.from_torch(default_state_torch.contiguous(), dtype=vec6f)

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidate the scene elements."""
        super()._invalidate_initialize_callback(event)

