Changed
^^^^^^^

* **Breaking:** Moved deformable body schema and material APIs from
  :mod:`isaaclab_physx.sim` to :mod:`isaaclab.sim`. Import
  :class:`~isaaclab.sim.DeformableBodyPropertiesCfg`,
  :func:`~isaaclab.sim.define_deformable_body_properties`,
  :func:`~isaaclab.sim.modify_deformable_body_properties`,
  :class:`~isaaclab.sim.DeformableBodyMaterialCfg`, and
  :func:`~isaaclab.sim.spawn_deformable_body_material` from :mod:`isaaclab.sim`
  instead of :mod:`isaaclab_physx.sim`.
