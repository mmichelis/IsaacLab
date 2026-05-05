Added
^^^^^

* Added backend-neutral deformable asset APIs, including
  :class:`~isaaclab.assets.DeformableObject`,
  :class:`~isaaclab.assets.DeformableObjectCfg`, and shared base/data classes.
* Added deformable body schema and material APIs under :mod:`isaaclab.sim`,
  including :class:`~isaaclab.sim.DeformableBodyPropertiesCfg`,
  :func:`~isaaclab.sim.define_deformable_body_properties`,
  :func:`~isaaclab.sim.modify_deformable_body_properties`,
  :class:`~isaaclab.sim.DeformableBodyMaterialCfg`, and
  :func:`~isaaclab.sim.spawn_deformable_body_material`.

Changed
^^^^^^^

* Changed USD spawning to support deformable objects whose USD assets contain
  embedded tetrahedral mesh data.
