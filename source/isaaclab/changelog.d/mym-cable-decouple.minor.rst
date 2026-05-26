Added
^^^^^

* Added :class:`~isaaclab.assets.CableObjectCfg` in
  :mod:`isaaclab.assets.cable_object` as a routing handle for the cable asset
  whose concrete implementation lives in
  :mod:`isaaclab_contrib.cable`. Mirrors how
  :class:`~isaaclab.assets.DeformableObjectCfg` lives in core while
  :class:`~isaaclab_newton.assets.DeformableObject` lives in the Newton package.

Changed
^^^^^^^

* Added a :class:`~isaaclab.assets.CableObjectCfg` routing branch in
  :class:`~isaaclab.scene.InteractiveScene` so cable cfgs instantiate a
  :class:`~isaaclab_contrib.cable.CableObject` (accessible via
  :attr:`~isaaclab.scene.InteractiveScene.cables` and
  ``scene["<cable_name>"]``) instead of falling through to the bare-prim
  :class:`~isaaclab.assets.AssetBaseCfg` path.
