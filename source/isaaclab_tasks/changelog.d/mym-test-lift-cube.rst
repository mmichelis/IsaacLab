Added
^^^^^

* Added the ``Isaac-Lift-Cube-Franka-Mjwarp-IK-Abs-v0`` task and
  :class:`~isaaclab_tasks.core.lift.config.franka.joint_pos_rigid_env_cfg.FrankaCubeLiftMjwarpIkAbsEnvCfg`,
  a pure-mjwarp rigid cube lift driven by absolute-pose differential IK so the
  pick-and-lift state machine demo can drive it. The demo script
  ``scripts/environments/state_machine/lift_franka_soft.py`` now supports this
  task, reading the cube as a rigid body for the state machine.
