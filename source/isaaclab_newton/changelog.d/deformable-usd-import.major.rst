Changed
^^^^^^^

* **Breaking:** Replaced legacy Newton deformable fields with canonical USD Physics schemas. Use
  ``thickness``, ``stretch_stiffness``, ``shear_stiffness``, and ``bend_stiffness`` for surfaces, and
  ``youngs_modulus`` and ``poissons_ratio`` for volumes. Surface density is now volumetric. The volume
  contact radius now follows Newton's ``ModelBuilder.default_particle_radius`` because canonical USD has no
  equivalent; the surface contact radius is seeded from ``thickness``. Damping fields are no longer authored.
