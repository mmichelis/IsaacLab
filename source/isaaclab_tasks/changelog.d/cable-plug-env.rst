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
* Added a no-cable variant ``Isaac-Lift-Plug-Franka-v0`` (config
  :class:`~isaaclab_tasks.manager_based.manipulation.lift_franka_soft.franka_cable_plug_env_cfg.FrankaCablePlugNoCableEnvCfg`,
  toggled by ``with_cable=False``) that drops the cable and anchor and manipulates the free rigid
  plug alone. The plug spawns centered on the gripper, aligned with it and rotated 90 deg about the
  gripper y axis (long axis across the approach axis), so closing the fingers grasps it. Its
  observation and action spaces match the cable env, so a policy trained without the cable can be
  deployed directly on the cable task.
* Added :func:`~isaaclab_tasks.manager_based.manipulation.lift_franka_soft.mdp.object_grasped`, a
  reward that grants a fixed bonus once both gripper fingers exert a contact force on the plug (read
  from the coupled solver) and the end-effector has reached it. It replaces the lift bonus in the
  cable-plug reward so the task rewards grasping rather than raising the plug.
* Added :func:`~isaaclab_tasks.manager_based.manipulation.lift_franka_soft.mdp.object_grasped_goal_distance`,
  a goal-tracking reward gated on the same grasp condition, so the cable-plug task only credits goal
  tracking while the plug is grasped rather than rewarding it drifting toward the goal on its own.
