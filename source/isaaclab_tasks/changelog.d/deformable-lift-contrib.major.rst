Changed
^^^^^^^

* **Breaking:** Moved the Franka soft-body and cloth lift environments from
  :mod:`isaaclab_tasks.core.lift.config.franka_soft` to
  :mod:`isaaclab_tasks.contrib.lift.config.franka`. Import them from the new
  package.
* **Breaking:** Replaced ``Isaac-Lift-Soft-Franka`` and
  ``Isaac-Lift-Cloth-Franka`` with ``IsaacContrib-Lift-Soft-Franka`` and
  ``IsaacContrib-Lift-Cloth-Franka``, respectively.
* **Breaking:** Changed the cloth lift environment from joint-position control
  to absolute differential IK control.

Removed
^^^^^^^

* **Breaking:** Removed the bundled RSL-RL PPO configurations for the
  deformable lift environments.
