.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _tune-vbd:

Tune the VBD Solver
===================

Vertex Block Descent (VBD) is the Newton solver for cloth and soft bodies. It is
selected through a :class:`~isaaclab_newton.physics.NewtonCfg` whose
``solver_cfg`` is a :class:`~isaaclab_newton.physics.VBDSolverCfg`; there is no
general ``newton_vbd`` preset, so tasks expose it through a task-specific preset
such as ``newton_mjwarp_vbd_proxy``. The generated API documentation for
:class:`~isaaclab_newton.physics.VBDSolverCfg` and
:class:`~isaaclab_newton.physics.NewtonSoftContactCfg` is authoritative for every
configuration field and its current default.

Prerequisites
-------------

First follow :doc:`/source/how-to/prepare_asset_for_newton` and confirm the task
builds under a rigid-only ``newton_mjwarp`` preset. Volume deformables need
:class:`~isaaclab_newton.sim.spawners.materials.NewtonDeformableBodyMaterialCfg`
and cloth needs
:class:`~isaaclab_newton.sim.spawners.materials.NewtonSurfaceDeformableBodyMaterialCfg`.
For preset authoring, see :ref:`backends-and-presets`.

Reproduce one problem with a fixed initial state, seed, and action sequence, then
write down the acceptance limit for each quantity the task must satisfy: maximum
rigid-deformable penetration, deformation under a known load, slip during
transport, self-intersection count, step time, and peak memory. A quantity
without a limit is a visual observation, not a result; label it as such and do
not use it to accept or reject a parameter change.

Keep the controller, asset resolution, timestep, and initial state fixed while
changing one parameter group. A setting that improves one frame but degrades the
recorded limits over the full motion is not an improvement.

Start from a maintained task
----------------------------

Confirm the installation and rendering path before tuning a new asset. The
scripted grasp is the fastest way to see what clips when a grasp fails:

.. code-block:: bash

    # volume soft body, needs the tetrahedralization extra
    uv run --extra tetrahedralization python scripts/environments/zero_agent.py --task Isaac-Lift-Soft-Franka --num_envs 1 --visualizer kit

    # surface deformable
    uv run python scripts/environments/zero_agent.py --task Isaac-Lift-Cloth-Franka --num_envs 1 --visualizer kit

    # scripted pick-and-lift check
    uv run --extra tetrahedralization python scripts/environments/state_machine/lift_franka_soft.py --num_envs 1

Copy the closest maintained preset as the starting configuration. Do not combine
independently tuned values from unrelated assets. Volume soft bodies, cloth, and
cables need different mesh resolution, material, contact, and substep choices.
The maintained presets show this: the cloth task raises
:attr:`~isaaclab_newton.physics.VBDSolverCfg.rigid_body_particle_contact_buffer_size`
to ``1024``, four times the value the soft-body task uses for the same gripper.

Choose the coupling path first
------------------------------

The coupling path decides which solver owns rigid-body integration, so fix it
before tuning VBD:

* Use :class:`~isaaclab_newton.physics.VBDSolverCfg` alone when VBD owns the
  whole scene step.
* Use :class:`~isaaclab_contrib.coupling.CouplerProxyCfg` or
  :class:`~isaaclab_contrib.coupling.CouplerAdmmCfg` named entries when a rigid
  solver advances the robot and VBD advances the deformable. The maintained
  Franka tasks use proxy coupling. See :ref:`newton-coupled-solvers` to choose.
* Use :class:`~isaaclab_contrib.custom_coupling.CoupledMJWarpVBDSolverCfg` for
  shared-model substep ordering, described in
  :doc:`/source/overview/core-concepts/physical-backends/newton/newton-manager-abstraction`.
  Keep its default ``coupling_mode="two_way"``, which prevents clipping more
  easily than one-way because contact penalties can push the robot back instead
  of only moving the deformable.

Stabilize each sub-solver on its own first. If the rigid scene or the deformable
is unstable alone, coupling parameters hide the cause instead of fixing it. When
the rigid entry is MJWarp, its ``nconmax`` and ``njmax`` must still cover the
rigid contacts and constraints in the scene; see :doc:`tune_mjwarp`.

.. _newton-vbd-proxy-coupling:

Proxy-coupled MJWarp and VBD
----------------------------

The core Franka soft-body task is the reference proxy configuration:

.. literalinclude:: ../../../../source/isaaclab_tasks/isaaclab_tasks/core/lift/config/franka_soft/franka_soft_env_cfg.py
    :language: python
    :start-at: newton_mjwarp_vbd_proxy: NewtonCfg
    :end-before: isaacsim_physx: PhysxCfg = PhysxCfg(
    :dedent: 4

The whole Franka articulation is routed to MJWarp while the deformable particles
and the static table and world shapes are routed to VBD. Only ``panda_hand`` and
the two fingers are exposed as proxies, so VBD sees three rigid proxies whatever
the arm link count. Expose only the rigid bodies that can contact the
deformable: coupling cannot return a force for a body absent from the mapping.

Validate the asset and material
-------------------------------

Check units, transforms, mesh topology, particle spacing, collision geometry, and
kinematic constraints before changing solver settings. Run the deformable without
rigid contact and confirm that gravity and a known load produce motion within the
recorded limits.

Tune material stiffness and damping in that contact-free scene. For volume
deformables the Lame parameters ``k_mu`` and ``k_lambda`` set stiffness and
``k_damp`` removes post-deformation oscillation. For cloth, ``tri_ke`` and
``tri_ka`` resist stretch and area change, ``edge_ke`` sets bending stiffness,
and the matching ``tri_kd`` and ``edge_kd`` damp the result. An object that is
already too stiff, too soft, or oscillatory cannot be repaired with contact
stiffness. Refine the mesh only when a measured limit or the contact
representation requires it, because refinement increases particle count, contact
work, and memory.

Choose timestep and substeps
----------------------------

Each VBD substep uses :attr:`~isaaclab.sim.SimulationCfg.dt` divided by
:attr:`~isaaclab_newton.physics.NewtonCfg.num_substeps`. Reduce the solver
timestep before increasing stiffness or iterations when the deformable explodes,
tunnels through colliders, or changes behavior strongly with the frame timestep.

Raise ``num_substeps`` until the recorded limits stop moving, and recheck step
time at every candidate. Do not change environment decimation to mask solver
instability, because decimation also changes the control period.

Set the VBD iteration budget
----------------------------

Raise :attr:`~isaaclab_newton.physics.VBDSolverCfg.iterations` only after the
asset, material, and solver timestep are stable. More iterations improve
convergence for stiff materials and rigid gripper contacts at higher runtime, but
cannot repair invalid geometry, reset overlap, missed contacts, or an unstable
timestep.

Sweep the iteration count against the same limits and step time, and keep the
lowest value after the limits plateau. Repeat the sweep after changing material
stiffness, contact stiffness, mesh resolution, or substeps.

Provide the intended contact surface
------------------------------------

Decide which contact surface the task needs before tuning any contact number.
Rigid-soft contact is per-vertex by default, so a thin rigid feature can pass
between soft vertices. Full-surface contact adds edge and triangle-interior
contacts through
:attr:`~isaaclab_newton.physics.NewtonCollisionPipelineCfg.enable_rigid_soft_full_surface_contact`
on the rigid-to-soft proxy mapping. The Franka soft-body and cloth tasks
already set it and the cable task does not, so read the copied preset before
assuming either way.

Check that the intended surface is actually available. Analytic shapes and
infinite planes are capable as they are. A mesh or convex proxy without a volume
SDF fails at construction, naming every such shape. Any other shape without an
analytic signed-distance field, such as a heightfield or a finite plane, only
warns and silently falls back to per-vertex contact, so read the warnings
instead of waiting for an error. Stop contact
tuning until the surface is available, because stiffness, damping, and friction
cannot recover a contact that is never generated.

Tune rigid-soft contact
-----------------------

Tune contact with self-contact disabled, in this order:

1. Raise :attr:`~isaaclab_newton.physics.NewtonSoftContactCfg.soft_contact_ke`
   only enough to control penetration. Excessive stiffness stops visible
   deformation and forces more iterations and substeps.
2. Adjust :attr:`~isaaclab_newton.physics.NewtonSoftContactCfg.soft_contact_kd`
   to reduce chatter or bounce. Excess damping makes contact sticky.
3. Adjust :attr:`~isaaclab_newton.physics.NewtonSoftContactCfg.soft_contact_mu`
   for slip once penetration and chatter are acceptable.

These are global model values set through
:attr:`~isaaclab_newton.physics.NewtonCfg.soft_contact_cfg`. Each combines with
the rigid shape's own material, so shape stiffness and friction change the result
as much as the soft-contact value does. Set shape defaults through
:class:`~isaaclab_newton.physics.NewtonShapeCfg` on ``NewtonCfg.default_shape_cfg``
or per asset through the asset's Newton contact material.

If ``soft_contact_ke`` is not enough, or ``soft_contact_mu`` has to become
implausibly high, the problem is usually the robot rather than the contact model.
Lower the arm actuator stiffness so the arm can respond to contact penalties;
prefer the arm being pushed back over the gripper clipping into the deformable.
For the gripper command, fully close the fingers and let the actuator effort
limit the squeeze. If contacts are missed, compare the material
``particle_radius`` and mesh resolution against the smallest collider feature
before hardening contact.

Raise
:attr:`~isaaclab_newton.physics.VBDSolverCfg.rigid_body_particle_contact_buffer_size`
only when Newton reports a per-body particle contact buffer overflow. It sizes a
per-body capacity, not a contact strength.

Enable and tune self-contact
----------------------------

Enable
:attr:`~isaaclab_newton.physics.VBDSolverCfg.particle_enable_self_contact` only
after rigid-soft contact is stable, then tune:

1. ``particle_self_contact_radius``, then ``particle_self_contact_margin``.
   With self-contact enabled, VBD raises at construction when the margin is
   smaller than the radius, and recommends 1.5 to 2 times the radius. Use the smallest pair that
   keeps self-intersection within its limit; larger values resist valid folds and
   make the object appear artificially stiff.
2. ``particle_collision_detection_interval``, when contacts appear or disappear
   during a fold. Refresh more often before raising iterations.
3. ``particle_vertex_contact_buffer_size`` and
   ``particle_edge_contact_buffer_size``, only when a reported overflow
   identifies a capacity problem.

``particle_topological_contact_filter_threshold`` suppresses contact between
neighboring elements of the same surface. Values above ``3`` increase compute
time sharply.

Diagnose by symptom
-------------------

.. list-table::
    :header-rows: 1
    :widths: 32 68

    * - Symptom
      - Check in this order
    * - The deformable is unstable without contact.
      - Asset units and topology, material stiffness and damping, solver timestep, then VBD iterations.
    * - Rigid bodies clip through the deformable.
      - Proxy selection and collision geometry, full-surface contact availability, particle radius, substeps, ``soft_contact_ke``, then VBD iterations.
    * - The gripper cannot carry the object.
      - Proxy body selection, gripper effort and actuator stiffness, rigid-shape friction, ``soft_contact_mu``, then coupling response.
    * - Contact chatters or bounces.
      - Reset overlap, solver timestep, ``soft_contact_kd``, material damping, then contact stiffness.
    * - The deformable barely deforms.
      - Material stiffness, ``soft_contact_ke``, kinematic constraints, and self-contact radius.
    * - Cloth or a soft body passes through itself.
      - Self-contact enablement, radius, margin, collision refresh interval, then reported buffer overflow.
    * - Self-contact is too expensive.
      - Mesh resolution, collision refresh interval, candidate margins, buffer sizing, and whether the task needs self-contact at all.
    * - Behavior changes when increasing ``num_envs``.
      - Reported capacity overflow or out-of-memory errors first. Reproduce the failing environment count before changing any solver setting.

Optimize only after validation
------------------------------

Once the single-environment result meets its limits, test the full reset and
command distribution, re-run the worst penetration, deformation, and self-contact
cases, then raise the environment count in bounded steps while watching warnings,
memory, and step time. Remove unused iteration, substep, collision-refresh, and
buffer headroom one group at a time, and record the final configuration with the
measurements that justify each non-default value.

Keep task-specific values in the task preset. Do not present a value tuned for
one asset as a general VBD default.

The diagnose-first order is: validate the asset and material; choose the coupling
path; set timestep and substeps; set the iteration budget; provide the contact
surface; tune rigid-soft contact; then enable self-contact.

See also
--------

* :class:`~isaaclab_newton.physics.VBDSolverCfg`
* :ref:`newton-coupled-solvers`
* :doc:`/source/overview/core-concepts/physical-backends/newton/newton-manager-abstraction`
* :doc:`tune_mjwarp`
