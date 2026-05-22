Added
^^^^^

* Added :class:`~isaaclab_contrib.deformable.MPMSolverCfg` and
  :class:`~isaaclab_contrib.deformable.ProxyCoupledMJWarpMPMSolverCfg` for
  proxy-coupled MuJoCo Warp + implicit MPM simulations, alongside the
  existing MJWarp + VBD pair. Includes a ``scripts/demos/newton_box_mpm_twoway.py``
  demo showing a rigid box dropped into an MPM sand bed.

Changed
^^^^^^^

* Moved the proxy-coupling partition / selector helpers
  (``_resolve_entity_to_body_ids``, ``_partition_model_by_entities``,
  ``_select_proxy_bodies``) into a shared ``_proxy_partition`` module so both
  proxy-coupled managers (VBD and MPM) reuse the same implementation. The
  public classmethod surface on
  :class:`~isaaclab_contrib.deformable.NewtonProxyCoupledMJWarpVBDManager`
  is preserved as thin shims.
