Changed
^^^^^^^

* **Breaking:** Renamed the manipulation-target scene entity and goal command
  in the ``lift_franka_soft`` tasks from ``deformable`` to ``object`` to align
  with the rigid lift task. Affects ``Isaac-Lift-Soft-Franka``,
  ``Isaac-Lift-Cloth-Franka``, ``Isaac-Lift-Cable-Franka-v0`` and
  ``Isaac-Lift-CablePlug-Franka-v0``: scene entry ``scene.deformable`` ->
  ``scene.object`` and command ``deformable_pose`` -> ``object_pose``. The MDP
  term and function names (e.g. ``deformable_ee_distance``) are unchanged.
  Update env configs, checkpoints, and RL configs accordingly.

* Migrated ``lift_franka_soft`` (rigid + cloth variants) from
  ``DeformableNewtonCfg`` to
  :class:`~isaaclab_contrib.deformable.newton_manager_cfg.CoupledNewtonCfg`
  with the proxy-coupled MJWarp + VBD solver as the default.
