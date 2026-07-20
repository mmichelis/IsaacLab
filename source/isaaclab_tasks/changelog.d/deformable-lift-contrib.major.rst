Changed
^^^^^^^

* **Breaking:** Moved the Franka soft-body and cloth lift configurations from
  ``isaaclab_tasks.core.lift.config.franka_soft`` to
  ``isaaclab_tasks.contrib.lift.config.franka``. Import the configurations from
  their new module paths.
* **Breaking:** Changed the Franka cloth lift action space from joint-position
  control to absolute differential IK. Use
  ``IsaacContrib-Lift-Cloth-Franka-IK-Abs`` for the new action space.

Deprecated
^^^^^^^^^^

* Deprecated ``Isaac-Lift-Soft-Franka`` and ``Isaac-Lift-Cloth-Franka`` in
  favor of ``IsaacContrib-Lift-Soft-Franka-IK-Abs`` and
  ``IsaacContrib-Lift-Cloth-Franka-IK-Abs``, respectively. The deprecated IDs
  retain their previous action spaces.

Removed
^^^^^^^

* **Breaking:** Removed the bundled RSL-RL PPO configurations and agent entry
  points for the Franka deformable lift tasks. Use the state-machine demo or
  provide an external training configuration.
