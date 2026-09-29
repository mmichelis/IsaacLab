Changed
^^^^^^^

* Delegated cloth and volume selection to Newton's family-specific deformable views while preserving stable read buffers and writes to both simulation states.
* Delegated deformable USD import and visual mesh evaluation to Newton, retaining Isaac Lab's scene-data interface.
* **Breaking:** Canonical deformable materials replaced Newton-only USD fields. Legacy configuration fields remained accepted with conversion where possible; unsupported damping, area stiffness, and volume contact-radius settings emitted warnings and were not applied. Use canonical material fields and configure volume contact radius on the Newton builder.

Fixed
^^^^^

* Rebound deformable views after model recreation without replacing user-held nodal position and velocity read buffers.
