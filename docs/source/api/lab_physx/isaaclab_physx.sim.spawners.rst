:orphan:

.. This page is kept as a migration stub for older links.

isaaclab_physx.sim.spawners
===========================

The deformable object spawner and deformable material APIs are now
backend-neutral and live in :mod:`isaaclab.sim.spawners`.

Use :class:`isaaclab.sim.spawners.DeformableObjectSpawnerCfg`,
:class:`isaaclab.sim.spawners.materials.DeformableBodyMaterialCfg`,
:class:`isaaclab.sim.spawners.materials.SurfaceDeformableBodyMaterialCfg`, and
:func:`isaaclab.sim.spawners.materials.spawn_deformable_body_material` instead.

For migration details, see :ref:`migrating-deformables`.
