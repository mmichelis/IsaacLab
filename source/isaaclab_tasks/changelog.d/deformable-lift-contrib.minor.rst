Added
^^^^^

* Added the agent-free ``IsaacContrib-Lift-Soft-Franka-IK-Abs`` and
  ``IsaacContrib-Lift-Cloth-Franka-IK-Abs`` environments with absolute
  differential IK control.

Deprecated
^^^^^^^^^^

* Deprecated ``Isaac-Lift-Soft-Franka`` and ``Isaac-Lift-Cloth-Franka`` in
  favor of ``IsaacContrib-Lift-Soft-Franka-IK-Abs`` and
  ``IsaacContrib-Lift-Cloth-Franka-IK-Abs``, respectively. Their existing
  action spaces and RSL-RL configuration entry points remain available during
  deprecation.
* Deprecated :mod:`isaaclab_tasks.core.lift.config.franka_soft` in favor of
  :mod:`isaaclab_tasks.contrib.lift.config.franka`. Existing imports remain
  available during deprecation.
