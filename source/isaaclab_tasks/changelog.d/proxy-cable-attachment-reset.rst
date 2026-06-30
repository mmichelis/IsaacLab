Added
^^^^^

* Added :func:`~isaaclab_tasks.core.lift.config.franka_soft.mdp.reset_cable_uniform`
  and :func:`~isaaclab_tasks.core.lift.config.franka_soft.mdp.reset_cable_assembly_uniform`
  event terms that re-seed Newton VBD state (``state.body_q``, ``state.body_qd``,
  ``solver.body_q_prev``, ``solver.body_inertia_q``) on episode reset, applying a per-env
  translation + yaw rigid transform to the cable (and, for the assembly variant, to the
  anchor and plug rigid bodies as well). Wired into
  :class:`~isaaclab_tasks.core.lift.config.franka_soft.franka_cable_env_cfg.FrankaCableEnvCfg`
  and
  :class:`~isaaclab_tasks.core.lift.config.franka_soft.franka_cable_plug_env_cfg.FrankaCablePlugEnvCfg`
  ``EventCfg``.

Fixed
^^^^^

* Fixed cable/plug pose leaking across episodes in
  :class:`~isaaclab_tasks.core.lift.config.franka_soft.franka_cable_plug_env_cfg.FrankaCablePlugEnvCfg`,
  where the prior episode's ``state.body_q`` and ``solver.body_q_prev`` carried into the next
  episode and produced a large AVBD impulse + visible cable snap on the first post-reset step.
