Changed
^^^^^^^

* **Breaking:** Changed Newton deformable materials to canonical USD Physics schemas. Use ``thickness``,
  ``stretch_stiffness``, ``shear_stiffness``, and ``bend_stiffness`` for surfaces, and
  ``youngs_modulus`` and ``poissons_ratio`` for volumes. Legacy Newton material fields remain
  deprecated for one release and are translated where canonical equivalents exist. Surface density
  is now volumetric. The volume contact radius follows Newton's ``ModelBuilder.default_particle_radius``;
  the surface contact radius is seeded from ``thickness``. Legacy damping and area stiffness have no
  canonical equivalents.
* Changed deformable visualization to delegate visual-mesh import and evaluation to Newton. No user action is
  required.
