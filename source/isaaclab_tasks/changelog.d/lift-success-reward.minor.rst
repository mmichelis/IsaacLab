Added
^^^^^

* Added :func:`~isaaclab_tasks.core.lift.mdp.rewards.object_goal_reached` and wired it as a
  ``success_bonus`` reward term in the lift task, giving a per-step bonus while the object is held
  within the success threshold of the commanded goal and lifted above the minimal height.
