Added
^^^^^

* Added the ``Isaac-Lift-Cube-Franka-Deformable-Proxy-v0`` task and
  :class:`~isaaclab_tasks.core.lift.config.franka.joint_pos_rigid_env_cfg.FrankaCubeLiftDeformableProxyEnvCfg`,
  a proxy-coupled Franka lift where the cube is a solid VBD deformable cuboid
  (Young's modulus 1e8 Pa) of the same size as the rigid variant. The object
  position observation uses the mean of the cube's vertices and the reset
  re-seeds the deformable nodal state.
* Added :class:`~isaaclab_tasks.core.lift.mdp.observations.DeformableOrientationInRobotRootFrame`,
  which reconstructs a rigid orientation quaternion for a deformable object by
  fitting a coordinate frame to a sample of its vertices (Kabsch), so a policy
  trained on a rigid object's orientation observation can be reused.

Fixed
^^^^^

* Fixed quadratic VBD body-particle contact allocation across replicated worlds.
