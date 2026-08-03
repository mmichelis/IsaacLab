Changed
^^^^^^^

* **Breaking:** Replaced legacy Newton deformable fields with canonical USD Physics schemas. Use
  ``thickness``, ``stretch_stiffness``, ``shear_stiffness``, and ``bend_stiffness`` for surfaces, and
  ``youngs_modulus`` and ``poissons_ratio`` for volumes. Surface density is now volumetric. The volume
  particle contact radius is set per-object with
  :attr:`~isaaclab_newton.sim.spawners.materials.NewtonDeformableBodyMaterialCfg.particle_contact_radius`
  (replacing the removed scene-wide ``NewtonCfg.default_particle_radius``); the surface contact radius is
  seeded from ``thickness``. Damping fields are no longer authored.
