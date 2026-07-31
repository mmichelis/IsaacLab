Added
^^^^^

* Added support for ``collision_props`` alongside ``deformable_props`` on mesh spawners, so a
  deformable's collider can be tuned with
  :class:`~isaaclab.sim.schemas.CollisionFragment` fragments. The fragments are applied to the
  simulation mesh, which is the prim carrying ``UsdPhysics.CollisionAPI``.

Removed
^^^^^^^

* **Breaking:** Removed ``PhysxDeformableCollisionPropertiesCfg`` and the ``contact_offset`` /
  ``rest_offset`` fields it contributed to
  :class:`~isaaclab_physx.sim.schemas.PhysxDeformableBodyPropertiesCfg`. They authored onto the
  deformable body prim, but PhysX reads collision offsets off the collider, so they had no effect.
  Pass the offsets through the spawner's ``collision_props`` instead, for example
  ``collision_props=[PhysxCollisionCfg(rest_offset=0.0005, contact_offset=0.005)]``.
