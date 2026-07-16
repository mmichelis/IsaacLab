Changed
^^^^^^^

* **Breaking:** Replaced legacy Newton deformable fields with canonical USD Physics schemas. Use
  ``thickness``, ``stretch_stiffness``, ``shear_stiffness``, and ``bend_stiffness`` for surfaces, and
  ``youngs_modulus`` and ``poissons_ratio`` for volumes. Surface density is now volumetric. Particle
  radius is configured scene-wide with :attr:`~isaaclab_newton.physics.NewtonCfg.default_particle_radius` for
  volumes and derived from ``thickness`` for surfaces. Damping fields are no longer authored.
