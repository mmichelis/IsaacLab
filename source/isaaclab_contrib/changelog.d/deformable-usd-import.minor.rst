Changed
^^^^^^^

* **Breaking:** Routed Newton deformable objects through the canonical USD importer and removed the
  experimental deformable builder hooks (``install_deformable_builder_hooks``,
  ``clear_deformable_builder_hooks``, and related construction helpers). Author canonical
  ``PhysicsDeformableBodyAPI`` schemas on deformable assets and re-import legacy USDs instead of registering
  them through the removed hooks.
