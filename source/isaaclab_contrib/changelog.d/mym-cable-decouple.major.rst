Changed
^^^^^^^

* **Breaking:** Changed :class:`~isaaclab_contrib.cable.CableObject` to subclass
  :class:`~isaaclab.assets.AssetBase` directly instead of
  :class:`~isaaclab.assets.Articulation`. Introduced
  :class:`~isaaclab_contrib.cable.CableData`, which replaces the inherited
  :class:`~isaaclab_newton.assets.ArticulationData` on the cable. The cable's
  user-facing surface no longer exposes joint targets, actuator state, tendons,
  wrench composers, jacobians, or mass matrices. Migrate code that called
  joint or actuator methods on a cable to read state via
  :attr:`~isaaclab_contrib.cable.CableObject.data` instead.
* **Breaking:** Moved :class:`~isaaclab_contrib.cable.CableObjectCfg` to
  :mod:`isaaclab.assets.cable_object` so that
  :class:`~isaaclab.scene.InteractiveScene` can route cable cfgs to instantiation.
  The package-level import path is preserved
  (``from isaaclab_contrib.cable import CableObjectCfg`` still works) via
  re-export in :mod:`isaaclab_contrib.cable`'s ``__init__``; the submodule path
  ``isaaclab_contrib.cable.cable_object_cfg`` is removed. Update any code that
  imports from that submodule to use either
  ``from isaaclab_contrib.cable import CableObjectCfg`` or
  ``from isaaclab.assets.cable_object import CableObjectCfg``.
