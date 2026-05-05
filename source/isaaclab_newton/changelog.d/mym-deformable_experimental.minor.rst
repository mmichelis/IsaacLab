Added
^^^^^

* Added solver-specific Newton manager classes for MuJoCo Warp, XPBD, Kamino,
  and Featherstone through :class:`~isaaclab_newton.physics.NewtonMJWarpManager`,
  :class:`~isaaclab_newton.physics.NewtonXPBDManager`,
  :class:`~isaaclab_newton.physics.NewtonKaminoManager`, and
  :class:`~isaaclab_newton.physics.NewtonFeatherstoneManager`.
* Added Newton deformable asset exports under
  :mod:`isaaclab_newton.assets.deformable_object`.

Changed
^^^^^^^

* Changed :class:`~isaaclab_newton.physics.NewtonCfg` manager selection to use
  :attr:`~isaaclab_newton.physics.NewtonSolverCfg.class_type` from the solver
  config. Existing ``NewtonCfg(solver_cfg=...)`` usage continues to work.

Deprecated
^^^^^^^^^^

* Deprecated :attr:`~isaaclab_newton.physics.NewtonSolverCfg.solver_type` in
  favor of :attr:`~isaaclab_newton.physics.NewtonSolverCfg.class_type`.
  ``solver_type`` is retained as metadata for logging and debugging only.
