Added
^^^^^

* Added :class:`~isaaclab_tasks.manager_based.manipulation.lift_franka_soft.franka_cable_plug_env_cfg.FrankaCablePlugEnvCfg`
  (gym id ``Isaac-Lift-CablePlug-Franka-v0``): a Franka Panda manipulating a Newton cable that
  is welded at one end to a kinematic anchor above the tabletop and at the other end to a rigid
  plug body. The RL task tracks the plug to a sampled target pose (a socket the plug is
  inserted into) using the existing ``object_*`` reward terms via the cable's
  :class:`~isaaclab_contrib.cable.CableAttachmentCfg` endpoints.
* Added :func:`~isaaclab_tasks.manager_based.manipulation.lift_franka_soft.mdp.assembly_velocity_out_of_bounds`,
  a termination that resets an environment whose arm-joint or assembly-body velocity exceeds a
  bound, guarding RL training against solver divergence.
