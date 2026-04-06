# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from isaaclab.markers.visualization_markers import VisualizationMarkers
import torch
import warp as wp

import isaaclab.utils.math as math_utils

from ..asset_base import AssetBase

if TYPE_CHECKING:
    from .deformable_object_cfg import DeformableObjectCfg
    from .deformable_object_data import DeformableObjectData


class BaseDeformableObject(AssetBase):
    """A deformable object asset class.

    Deformable objects are assets that can be deformed in the simulation. They are typically used for
    soft bodies, such as stuffed animals and food items.

    Unlike rigid object assets, deformable objects have a more complex structure and require additional
    handling for simulation. The simulation of deformable objects follows a finite element approach, where
    the object is discretized into a mesh of nodes and elements. The nodes are connected by elements, which
    define the material properties of the object. The nodes can be moved and deformed, and the elements
    respond to these changes.

    The state of a deformable object comprises of its nodal positions and velocities, and not the object's root
    position and orientation. The nodal positions and velocities are in the simulation frame.

    Soft bodies can be `partially kinematic`_, where some nodes are driven by kinematic targets, and the rest are
    simulated. The kinematic targets are the desired positions of the nodes, and the simulation drives the nodes
    towards these targets. This is useful for partial control of the object, such as moving a stuffed animal's
    head while the rest of the body is simulated.

    .. attention::
        This class is experimental and subject to change due to changes on the underlying physics APIs on which
        it depends. We will try to maintain backward compatibility as much as possible but some changes may be
        necessary.

    .. _partially kinematic: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/SoftBodies.html#kinematic-soft-bodies
    """

    cfg: DeformableObjectCfg
    """Configuration instance for the deformable object."""

    __backend_name__: str = "base"
    """The name of the backend for the deformable object."""

    def __init__(self, cfg: DeformableObjectCfg):
        """Initialize the deformable object.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)

    """
    Properties
    """

    @property
    @abstractmethod
    def data(self) -> DeformableObjectData:
        raise NotImplementedError()

    @property
    @abstractmethod
    def num_instances(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def num_bodies(self) -> int:
        """Number of bodies in the asset.

        This is always 1 since each object is a single deformable body.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def root_view(self) -> Any:
        """Deformable body view for the asset.

        .. note::
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def max_sim_vertices_per_body(self) -> int:
        """The maximum number of simulation mesh vertices per deformable body."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def max_sim_elements_per_body(self) -> int:
        """The maximum number of simulation mesh elements per deformable body."""
        raise NotImplementedError()

    """
    Operations.
    """

    @abstractmethod
    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None) -> None:
        """Reset the deformable object.

        Args:
            env_ids: Environment indices. If None, then all indices are used.
            env_mask: Environment mask. If None, then all the instances are updated. Shape is (num_instances,).
        """
        raise NotImplementedError()

    @abstractmethod
    def write_data_to_sim(self) -> None:
        """Write data to the simulation."""
        raise NotImplementedError()

    @abstractmethod
    def update(self, dt: float) -> None:
        """Updates the simulation data.

        Args:
            dt: The time step size in seconds.
        """
        raise NotImplementedError()

    """
    Operations - Write to simulation.
    """

    @abstractmethod
    def write_nodal_state_to_sim_index(
        self,
        nodal_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set the nodal state over selected environment indices into the simulation.

        The nodal state comprises of the nodal positions and velocities. Since these are nodes, the velocity only has
        a translational component. All the quantities are in the simulation frame.

        Args:
            nodal_state: Nodal state in simulation frame.
                Shape is (len(env_ids), max_sim_vertices_per_body, 6) or (num_instances, max_sim_vertices_per_body, 6).
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_nodal_state_to_sim_mask(
        self,
        nodal_state: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the nodal state over selected environment mask into the simulation.

        The nodal state comprises of the nodal positions and velocities. Since these are nodes, the velocity only has
        a translational component. All the quantities are in the simulation frame.

        Args:
            nodal_state: Nodal state in simulation frame.
                Shape is (num_instances, max_sim_vertices_per_body, 6).
            env_mask: Environment mask. If None, then all indices are used.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_nodal_pos_to_sim_index(
        self,
        nodal_pos: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set the nodal positions over selected environment indices into the simulation.

        The nodal position comprises of individual nodal positions of the simulation mesh for the deformable body.
        The positions are in the simulation frame.

        Args:
            nodal_pos: Nodal positions in simulation frame.
                Shape is (len(env_ids), max_sim_vertices_per_body, 3) or (num_instances, max_sim_vertices_per_body, 3).
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_nodal_pos_to_sim_mask(
        self,
        nodal_pos: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the nodal positions over selected environment mask into the simulation.

        The nodal position comprises of individual nodal positions of the simulation mesh for the deformable body.
        The positions are in the simulation frame.

        Args:
            nodal_pos: Nodal positions in simulation frame.
                Shape is (num_instances, max_sim_vertices_per_body, 3).
            env_mask: Environment mask. If None, then all indices are used.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_nodal_velocity_to_sim_index(
        self,
        nodal_vel: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set the nodal velocity over selected environment indices into the simulation.

        The nodal velocity comprises of individual nodal velocities of the simulation mesh for the deformable
        body. Since these are nodes, the velocity only has a translational component. The velocities are in the
        simulation frame.

        Args:
            nodal_vel: Nodal velocities in simulation frame.
                Shape is (len(env_ids), max_sim_vertices_per_body, 3) or (num_instances, max_sim_vertices_per_body, 3).
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_nodal_velocity_to_sim_mask(
        self,
        nodal_vel: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the nodal velocity over selected environment mask into the simulation.

        The nodal velocity comprises of individual nodal velocities of the simulation mesh for the deformable
        body. Since these are nodes, the velocity only has a translational component. The velocities are in the
        simulation frame.

        Args:
            nodal_vel: Nodal velocities in simulation frame.
                Shape is (num_instances, max_sim_vertices_per_body, 3).
            env_mask: Environment mask. If None, then all indices are used.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_nodal_kinematic_target_to_sim_index(
        self,
        targets: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        full_data: bool = False,
    ) -> None:
        """Set the kinematic targets of the simulation mesh for the deformable bodies using indices.

        The kinematic targets comprise of individual nodal positions of the simulation mesh for the deformable body
        and a flag indicating whether the node is kinematically driven or not. The positions are in the simulation
        frame.

        .. note::
            The flag is set to 0.0 for kinematically driven nodes and 1.0 for free nodes.

        Args:
            targets: The kinematic targets comprising of nodal positions and flags.
                Shape is (len(env_ids), max_sim_vertices_per_body, 4) or (num_instances, max_sim_vertices_per_body, 4).
            env_ids: Environment indices. If None, then all indices are used.
            full_data: Whether to expect full data. Defaults to False.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_nodal_kinematic_target_to_sim_mask(
        self,
        targets: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set the kinematic targets of the simulation mesh for the deformable bodies using mask.

        The kinematic targets comprise of individual nodal positions of the simulation mesh for the deformable body
        and a flag indicating whether the node is kinematically driven or not. The positions are in the simulation
        frame.

        .. note::
            The flag is set to 0.0 for kinematically driven nodes and 1.0 for free nodes.

        Args:
            targets: The kinematic targets comprising of nodal positions and flags.
                Shape is (num_instances, max_sim_vertices_per_body, 4).
            env_mask: Environment mask. If None, then all indices are used.
        """
        raise NotImplementedError()

    """
    Internal helper.
    """

    @abstractmethod
    def _initialize_impl(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _create_buffers(self) -> None:
        raise NotImplementedError()

    """
    Internal simulation callbacks.
    """


    """
    Internal simulation callbacks.
    """

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        """Set debug visualization of kinematic targets.

        Args:
            debug_vis: Whether to enable debug visualization.
        """
        # set visibility of markers
        if debug_vis:
            if not hasattr(self, "target_visualizer"):
                self.target_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            # set their visibility to true
            self.target_visualizer.set_visibility(True)
        else:
            if hasattr(self, "target_visualizer"):
                self.target_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        """Callback for debug visualization of kinematic targets.

        Args:
            event: The visualization event.
        """
        # check where to visualize
        kinematic_target_torch = wp.to_torch(self.data.nodal_kinematic_target)
        targets_enabled = kinematic_target_torch[:, :, 3] == 0.0
        num_enabled = int(torch.sum(targets_enabled).item())
        # get positions if any targets are enabled
        if num_enabled == 0:
            # create a marker below the ground
            positions = torch.tensor([[0.0, 0.0, -10.0]], device=self.device)
        else:
            positions = kinematic_target_torch[targets_enabled][..., :3]
        # show target visualizer
        self.target_visualizer.visualize(positions)

    @abstractmethod
    def _invalidate_initialize_callback(self, event) -> None:
        """Invalidates the scene elements."""
        super()._invalidate_initialize_callback(event)

    """
    Operations - Helper.
    """

    def transform_nodal_pos(
        self, nodal_pos: torch.Tensor, pos: torch.Tensor | None = None, quat: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Transform the nodal positions based on the pose transformation.

        This function computes the transformation of the nodal positions based on the pose transformation.
        It multiplies the nodal positions with the rotation matrix of the pose and adds the translation.
        Internally, it calls the :meth:`isaaclab.utils.math.transform_points` function.

        Args:
            nodal_pos: The nodal positions in the simulation frame. Shape is (N, max_sim_vertices_per_body, 3).
            pos: The position transformation. Shape is (N, 3).
                Defaults to None, in which case the position is assumed to be zero.
            quat: The orientation transformation as quaternion (x, y, z, w). Shape is (N, 4).
                Defaults to None, in which case the orientation is assumed to be identity.

        Returns:
            The transformed nodal positions. Shape is (N, max_sim_vertices_per_body, 3).
        """
        # offset the nodal positions to center them around the origin
        mean_nodal_pos = nodal_pos.mean(dim=1, keepdim=True)
        nodal_pos = nodal_pos - mean_nodal_pos
        # transform the nodal positions based on the pose around the origin
        return math_utils.transform_points(nodal_pos, pos, quat) + mean_nodal_pos

    """
    Deprecated.
    """

    def write_nodal_state_to_sim(
        self,
        nodal_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated. Please use :meth:`write_nodal_state_to_sim_index` instead."""
        warnings.warn(
            "The method 'write_nodal_state_to_sim' is deprecated. Please use 'write_nodal_state_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_nodal_state_to_sim_index(nodal_state, env_ids=env_ids)

    def write_nodal_pos_to_sim(
        self,
        nodal_pos: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated. Please use :meth:`write_nodal_pos_to_sim_index` instead."""
        warnings.warn(
            "The method 'write_nodal_pos_to_sim' is deprecated. Please use 'write_nodal_pos_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_nodal_pos_to_sim_index(nodal_pos, env_ids=env_ids)

    def write_nodal_velocity_to_sim(
        self,
        nodal_vel: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated. Please use :meth:`write_nodal_velocity_to_sim_index` instead."""
        warnings.warn(
            "The method 'write_nodal_velocity_to_sim' is deprecated."
            " Please use 'write_nodal_velocity_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_nodal_velocity_to_sim_index(nodal_vel, env_ids=env_ids)

    def write_nodal_kinematic_target_to_sim(
        self,
        targets: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Deprecated. Please use :meth:`write_nodal_kinematic_target_to_sim_index` instead."""
        warnings.warn(
            "The method 'write_nodal_kinematic_target_to_sim' is deprecated."
            " Please use 'write_nodal_kinematic_target_to_sim_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_nodal_kinematic_target_to_sim_index(targets, env_ids=env_ids)
