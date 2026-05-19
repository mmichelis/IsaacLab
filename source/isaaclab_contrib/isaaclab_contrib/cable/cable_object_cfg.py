# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the cable asset class."""

from __future__ import annotations
from dataclasses import MISSING
from typing import Literal

from isaaclab.actuators import ActuatorBaseCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.utils.configclass import configclass


@configclass
class CableAttachmentCfg:
    """Weld a cable endpoint to a body on another spawned asset.

    The attachment creates a Newton fixed joint between one of the cable's end
    rod-segment bodies and a body on a separately spawned rigid asset. The joint
    is realized at Newton model-build time, after both assets are registered
    with the builder. Newton's rigid solver then enforces the constraint
    natively each step; no per-step Python synchronization is required.
    """

    target_prim_path: str = MISSING
    """Prim path of the rigid body to weld the cable endpoint to.

    Must resolve to a prim that has been registered with Newton as a rigid body
    (e.g., spawned via :class:`~isaaclab.assets.RigidObject`) prior to the cable
    being realized. Regex patterns are not supported here; the path must be the
    same concrete spawn path the rigid asset's :class:`RigidObjectCfg` uses
    (the per-world resolver matches it against ``builder.body_label`` at
    builder-hook time).
    """

    cable_anchor: Literal["head", "tail"] = "tail"
    """Which end of the cable to anchor.

    ``"head"`` is the first rod-segment body (corresponding to the BasisCurves
    point at index 0). ``"tail"`` is the last rod-segment body. The internal
    resolver maps this symbolic name to the Newton body index recorded on the
    cable's registry entry at :meth:`newton.ModelBuilder.add_rod_graph` time.
    """

    local_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Joint anchor position [m] in the target body's local frame."""

    local_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Joint anchor orientation as quaternion ``(w, x, y, z)`` in the target body's local frame."""


@configclass
class CableObjectCfg(ArticulationCfg):
    """Configuration for a cable / 1D-rod asset (Newton backend).

    Inherits all of :class:`ArticulationCfg` and overrides two defaults so the
    base :meth:`Articulation._initialize_impl` runs unchanged on cables:

    - ``articulation_root_prim_path = "/cable_articulation"`` -- the sub-label
      that :meth:`newton.ModelBuilder.add_rod_graph` produces under the cable's
      source prim path (``f"{label}_articulation"`` where ``label`` is
      ``"{prim_path}/cable"``). The base method composes this with
      ``cfg.prim_path`` and uses the result as the label pattern for
      :class:`newton.selection.ArticulationView`.
    - ``actuators = {}`` -- cables have no user-defined actuators (cable joint
      stiffness is material-like, applied internally by the solver). The
      inherited ``_process_actuators_cfg`` iterates an empty dict safely and
      emits a harmless ``logger.warning("Not all actuators are configured!")``
      -- expected and not suppressed in Phase 1.

    The ``attachments`` field carries zero or more
    :class:`CableAttachmentCfg` entries; each produces one Newton fixed joint
    between the named cable endpoint and a separately spawned rigid body.
    """

    class_type: type = "{DIR}.cable_object:CableObject"
    articulation_root_prim_path: str | None = "/cable_articulation"
    actuators: dict[str, ActuatorBaseCfg] = {}
    attachments: list[CableAttachmentCfg] = []
