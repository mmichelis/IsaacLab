Fixed
^^^^^

* Fixed the lazy forward-kinematics refresh in :class:`~isaaclab_newton.assets.ArticulationData`
  and :class:`~isaaclab_newton.assets.RigidObjectData` zeroing the velocity of VBD-owned bodies in
  **every** environment whenever **any** environment reset. The refresh called the base
  :meth:`~isaaclab_newton.physics.NewtonManager.forward` (unmasked ``eval_fk``) directly, which
  recomputes ``body_qd`` from joint state for all bodies; VBD owns velocity in ``body_qd`` and
  leaves ``joint_qd`` zero, so airborne bystander bodies briefly lost their momentum and stuttered.
  The refresh now dispatches through the active physics manager so the VBD/coupled-solver
  ``forward()`` override (which skips VBD-owned bodies) is honored.
