Using the VBD Solver
====================

Vertex Block Descent (VBD) is Newton's deformable-body solver for cloth and soft-body simulation.
Isaac Lab exposes the current VBD integration through :mod:`isaaclab_contrib.deformable` because the feature is still
experimental and is not part of the required Isaac Lab core.

.. note::

   The VBD integration is experimental. APIs, tuning defaults, and supported
   workflows may change while Newton and the Isaac Lab integration are under
   active development. Treat the current support as an early-access path for
   deformable-body experiments.

VBD is most useful in two workflows:

* deformable-only scenes, where :class:`~isaaclab_contrib.deformable.VBDSolverCfg`
  advances the deformable particles directly;
* robot or rigid-body scenes with deformables, where
  :class:`~isaaclab_contrib.deformable.CoupledMJWarpVBDSolverCfg` or
  :class:`~isaaclab_contrib.deformable.CoupledFeatherstoneVBDSolverCfg`
  advances rigid bodies with a rigid solver and deformables with VBD.

Unlike Kamino, VBD is not currently exposed as a general ``newton_vbd`` physics
preset for all tasks. A task usually needs a deformable asset, a Newton
deformable material, and a solver configuration tuned for that scene.

Running the Example Tasks
-------------------------

The fastest way to exercise the integration is to run the experimental Franka
deformable lifting tasks:

.. code-block:: bash

   ./isaaclab.sh -p scripts/environments/zero_agent.py --task Isaac-Lift-Soft-Franka-v0 --num_envs 1 --visualizer kit

For the surface-deformable cloth variant, use:

.. code-block:: bash

   ./isaaclab.sh -p scripts/environments/zero_agent.py --task Isaac-Lift-Cloth-Franka-v0 --num_envs 1 --visualizer kit

Both tasks configure MJWarp for the rigid Franka and VBD for the deformable
object through
:class:`~isaaclab_contrib.deformable.CoupledMJWarpVBDSolverCfg`.

Configuring VBD
---------------

For a deformable-only scene, configure Newton with
:class:`~isaaclab_contrib.deformable.VBDSolverCfg`:

.. code-block:: python

   from isaaclab.sim import SimulationCfg
   from isaaclab.utils import configclass
   from isaaclab_newton.physics import NewtonCfg

   from isaaclab_contrib.deformable import NewtonModelCfg, VBDSolverCfg


   @configclass
   class DeformableNewtonCfg(NewtonCfg):
       model_cfg: NewtonModelCfg | None = None


   sim_cfg = SimulationCfg(
       dt=1 / 120,
       physics=DeformableNewtonCfg(
           solver_cfg=VBDSolverCfg(
               iterations=10,
               particle_enable_self_contact=False,
           ),
           model_cfg=NewtonModelCfg(
               soft_contact_ke=1.0e3,
               soft_contact_kd=1.0e-2,
               soft_contact_mu=0.5,
           ),
           num_substeps=5,
           use_cuda_graph=True,
       ),
   )

For a robot or rigid-body scene with deformables, prefer a coupled solver. The
rigid solver advances articulated and rigid bodies, while VBD advances the
deformable particles:

.. code-block:: python

   from isaaclab_newton.physics import MJWarpSolverCfg

   from isaaclab_contrib.deformable import (
       CoupledMJWarpVBDSolverCfg,
       NewtonModelCfg,
       VBDSolverCfg,
   )

   physics = DeformableNewtonCfg(
       solver_cfg=CoupledMJWarpVBDSolverCfg(
           rigid_solver_cfg=MJWarpSolverCfg(
               njmax=40,
               nconmax=20,
               ls_iterations=20,
               cone="pyramidal",
               integrator="implicitfast",
               ccd_iterations=100,
           ),
           soft_solver_cfg=VBDSolverCfg(
               iterations=10,
               integrate_with_external_rigid_solver=True,
               particle_enable_self_contact=False,
               particle_collision_detection_interval=-1,
           ),
           coupling_mode="two_way",
       ),
       model_cfg=NewtonModelCfg(
           soft_contact_ke=1.0e4,
           soft_contact_kd=1.0e-5,
           soft_contact_mu=5.0,
           shape_material_ke=4.0e4,
           shape_material_kd=1.0e-5,
           shape_material_mu=5.0,
       ),
       num_substeps=10,
       use_cuda_graph=True,
   )

The extra ``DeformableNewtonCfg`` wrapper is used by the current contrib
managers to carry :class:`~isaaclab_contrib.deformable.NewtonModelCfg` alongside
the normal :class:`~isaaclab_newton.physics.NewtonCfg` fields.

Solver Parameters
-----------------

The most important VBD controls are in
:class:`~isaaclab_contrib.deformable.VBDSolverCfg`:

``iterations``
   Number of VBD iterations per substep. Increasing this value improves
   deformation and contact convergence, especially for stiff materials or rigid
   gripper contacts, but increases simulation cost.

``integrate_with_external_rigid_solver``
   Set this to ``True`` when VBD is used inside a coupled solver so that the
   rigid sub-solver owns rigid-body integration. Leave it ``False`` for
   deformable-only VBD scenes.

``particle_enable_self_contact``
   Enables deformable self-contact. This is useful for cloth folds and soft
   bodies that collide with themselves, but it is more expensive and introduces
   additional contact tuning.

``particle_self_contact_radius``
   Effective self-contact thickness. VBD applies vertex-triangle and edge-edge
   self-contact response when the current primitive distance is smaller than
   this radius. Increase it when cloth or soft bodies visibly pass through
   themselves; reduce it if self-contact starts too early or keeps nearby layers
   separated.

``particle_self_contact_margin``
   Self-contact candidate search distance. VBD uses this larger envelope when
   building the vertex-triangle and edge-edge contact lists, then applies
   contact response using ``particle_self_contact_radius``. Set the margin
   greater than or equal to the radius so contacts are detected before they
   enter the active contact region. A larger margin can reduce missed contacts,
   but increases candidate count and simulation cost.

``particle_collision_detection_interval``
   Controls how often self-contact detection runs. A negative value performs
   detection before initialization only. ``0`` performs detection before and
   immediately after initialization. A positive value ``k`` runs detection
   before every ``k`` VBD iterations. Smaller positive values make self-contact
   more responsive at higher cost.

``particle_vertex_contact_buffer_size`` and ``particle_edge_contact_buffer_size``
   Preallocate storage for vertex-triangle and edge-edge self-contact
   candidates. Increase these values if dense folds or high-resolution cloth
   need more self-contact capacity.

``particle_topological_contact_filter_threshold``
   Filters contacts between mesh primitives that are close in topology. Increase
   this value to suppress contact between neighboring elements of the same
   surface. Values greater than ``3`` can significantly increase compute time.

``particle_rest_shape_contact_exclusion_radius``
   Filters self-contact candidates whose rest-configuration distance is shorter
   than the given distance. Increase this when rest-neighbor contacts produce
   unwanted resistance.


Global Contact Parameters
-------------------------

:class:`~isaaclab_contrib.deformable.NewtonModelCfg` applies contact parameters
to the finalized Newton model:

``soft_contact_ke``
   Stiffness for body-particle and particle self-contact. Increase it to reduce
   clipping through rigid shapes or through other deformable particles. If it is
   too high, the object can stop visibly deforming or require more VBD
   iterations and substeps.

``soft_contact_kd``
   Contact damping. Increase it to reduce chatter or bouncing. Too much damping
   can make contact response sticky or overdamped.

``soft_contact_mu``
   Friction coefficient for body-particle and particle self-contact. Increase
   it when a gripper cannot carry the deformable object without slipping. Very
   high values can hide other tuning problems, such as low contact stiffness or
   insufficient actuator force.

``shape_material_ke``, ``shape_material_kd``, and ``shape_material_mu``
   Optional overrides for all rigid collision-shape contact material values in
   the Newton model. Use these when the rigid-side material parsed from the asset
   is not appropriate for deformable contact. Body-particle stiffness combines
   the soft contact value and the rigid shape value, and body-particle friction
   depends on both friction coefficients.

Deformable Material Parameters
------------------------------

The deformable asset material also controls the simulation response. For volume
deformables, use
:class:`~isaaclab_newton.sim.spawners.materials.NewtonDeformableBodyMaterialCfg`:

``density``
   Material density. Higher density increases particle mass and inertia, so the
   object accelerates and deforms less for the same contact forces.

``particle_radius``
   Particle contact radius used by Newton. Increase it when contacts are missed
   or detected too late. If it is too large relative to the mesh resolution,
   contacts can start too early.

``k_mu`` and ``k_lambda``
   Lame material parameters for the volume. Higher values make the deformable
   object stiffer and usually require more VBD iterations, more substeps, or a
   smaller timestep.

``k_damp``
   Damping for tetrahedral elements. Increase it to reduce oscillations after
   deformation, but avoid overdamping if the object should rebound.

For surface deformables such as cloth, use
:class:`~isaaclab_newton.sim.spawners.materials.NewtonSurfaceDeformableBodyMaterialCfg`:

``tri_ke`` and ``tri_ka``
   Triangle stiffness terms. Increase them to reduce stretch or area change in
   cloth.

``tri_kd``
   Triangle damping. Increase it to reduce cloth vibration after stretching.

``edge_ke``
   Bending stiffness. Increase it for stiffer cloth folds; decrease it for
   softer draping.

``edge_kd``
   Bending damping. Increase it to damp fold oscillations.

Coupling Parameters
-------------------

For rigid-body interaction, the coupled solver's ``coupling_mode`` controls how
contact information flows between the rigid and deformable solvers:

``"one_way"``
   The rigid solver advances first, and VBD reacts to the updated rigid poses.
   The rigid solver does not feel particle contact forces. This can be useful for
   scripted obstacles or early debugging, but it is usually insufficient for
   grasping and lifting deformables.

``"two_way"``
   Contact reactions from the deformable are injected into the rigid solver
   before the rigid step, then VBD advances the deformable against the shared
   contacts. Use this for manipulation tasks where the robot should be pushed
   back by the deformable object.

``"kinematic"``
   Available on
   :class:`~isaaclab_contrib.deformable.CoupledFeatherstoneVBDSolverCfg`. The
   rigid bodies are kinematically updated by Featherstone, and VBD reacts to
   them. The rigid solver does not feel particle contacts.

The rigid solver parameters still matter. For example, MJWarp's ``nconmax`` and
``njmax`` must be large enough for the rigid contacts in the scene, and
``ccd_iterations`` can affect fast rigid contacts near deformables. See
:doc:`solver-transitioning` for the MJWarp-side parameters.

Tuning Workflow
---------------

Use a small visual scene before training a policy. The usual tuning order is:

1. Start from the Franka soft-body or cloth task and verify that the deformable
   spawns, moves, and renders correctly.
2. Tune deformable material stiffness and damping until the object deforms in
   the expected range without rigid contact.
3. Increase ``num_substeps`` or decrease ``dt`` if the object is unstable before
   increasing stiffness further.
4. Increase :attr:`~isaaclab_contrib.deformable.VBDSolverCfg.iterations` when
   contacts or stiff materials do not converge within a substep.
5. Tune :attr:`~isaaclab_contrib.deformable.NewtonModelCfg.soft_contact_ke` to
   reduce rigid/deformable clipping, then tune
   :attr:`~isaaclab_contrib.deformable.NewtonModelCfg.soft_contact_mu` for grip
   and :attr:`~isaaclab_contrib.deformable.NewtonModelCfg.soft_contact_kd` for
   chatter.
6. Tune ``shape_material_*`` values if the rigid-side contact material is the
   limiting factor.
7. Enable self-contact only after body-particle contact is stable, then tune the
   self-contact radius, margin, detection interval, and buffers.


Symptoms and First Parameters to Check
--------------------------------------

* Rigid bodies visibly clip through the deformable: increase
  ``soft_contact_ke``, increase VBD ``iterations``, increase ``num_substeps``,
  or increase the deformable material ``particle_radius``.
* The robot cannot lift the deformable: use ``coupling_mode="two_way"``, then
  increase ``soft_contact_mu`` and the rigid-side ``shape_material_mu``. Also
  check the gripper actuator stiffness and effort limits.
* The deformable barely deforms: reduce material stiffness, reduce
  ``soft_contact_ke``, or reduce shape contact stiffness.
* Contact chatters or bounces: increase ``soft_contact_kd`` or material damping,
  and consider using more substeps.
* Cloth passes through itself: enable ``particle_enable_self_contact``, increase
  ``particle_self_contact_radius`` if the active self-contact thickness is too
  small, increase ``particle_self_contact_margin`` if contacts are missed, and
  use a positive ``particle_collision_detection_interval``.
* Self-contact is too expensive: increase ``particle_collision_detection_interval``,
  reduce mesh resolution, or disable self-contact until the rest of the scene is
  tuned.

For implementation details of the VBD and coupled solver managers, see
:doc:`newton-manager-abstraction`.
