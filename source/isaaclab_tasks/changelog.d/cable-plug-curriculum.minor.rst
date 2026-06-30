Added
^^^^^

* Added a step-based curriculum to the Franka cable-plug environments that incrementally widens the
  plug grasp and goal spherical reset ranges from a tight start to wider, workspace-clipped bounds
  over training, via the new :func:`~isaaclab_tasks.core.lift.config.franka_soft.mdp.step_widen_pose_range`
  curriculum term, and ramps the episode length from short to long via
  :func:`~isaaclab_tasks.core.lift.config.franka_soft.mdp.step_interpolate_value`.
* Added ``Isaac-Lift-CablePlug-Franka-Play-v0`` and ``Isaac-Lift-Plug-Franka-Play-v0`` eval
  environments that pin the reset ranges to the curriculum's full-difficulty bounds and disable the
  curriculum, so a trained policy is evaluated at full difficulty from the first reset.
* Added a termination to the Franka cable-plug environments that ends the episode when the plug
  leaves the table footprint, complementing the existing below-table drop termination.
