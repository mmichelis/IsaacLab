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
  bound or whose state becomes non-finite (NaN/Inf). Finiteness is checked across every quantity
  the rewards and observations read -- arm joints, root, and body poses, the end-effector frame,
  and the assembly bodies -- guarding RL training against solver divergence.
* Added a termination that resets an environment when the plug drops below the tabletop.
* Added ``FrankaCablePlugEnv``, the environment class behind ``Isaac-Lift-CablePlug-Franka-v0``,
  which zeros non-finite rewards (computed before the divergence reset) so a diverged coupled
  solve never propagates NaNs to the learner.
