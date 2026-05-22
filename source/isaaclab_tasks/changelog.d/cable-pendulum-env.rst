Added
^^^^^

* Added :class:`~isaaclab_tasks.manager_based.manipulation.lift_franka_soft.franka_cable_pendulum_env_cfg.FrankaCablePendulumEnvCfg`
  (gym id ``Isaac-Lift-CablePendulum-Franka-v0``): a Franka Panda manipulating a Newton cable that
  is welded at one end to a kinematic anchor above the tabletop and at the other end to a rigid
  plug body. The RL task tracks the plug to a sampled target pose using the existing
  ``object_*`` reward terms via the cable's :class:`~isaaclab_contrib.cable.CableAttachmentCfg`
  endpoints.
