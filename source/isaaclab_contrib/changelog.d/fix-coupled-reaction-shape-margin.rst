Fixed
^^^^^

* Fixed two-way rigid-deformable coupling not applying any reaction force to rigid bodies with
  Newton 1.4. The particle-body contact evaluation was missing the new ``shape_margin`` argument,
  which made the reaction kernel fail to compile so reaction wrenches were never written back to
  the rigid solver.
