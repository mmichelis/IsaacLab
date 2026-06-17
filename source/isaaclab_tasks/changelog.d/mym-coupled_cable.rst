Changed
^^^^^^^

* Changed the ``plug_goal_tracking`` reward of the Franka cable-plug task to gate goal-position
  tracking on long-axis alignment: the plug only earns full credit when its body z axis points into
  the goal bore (the goal frame's +x axis), via a signed cosine clamped to ``[0, 1]``. Rotation about
  the long axis is left free.
