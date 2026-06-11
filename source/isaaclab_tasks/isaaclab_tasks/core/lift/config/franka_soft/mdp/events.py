# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset events for the cable / cable+anchor+plug assembly.

The cable and (optional) anchor/plug bodies live on the VBD side of the
proxy-coupled MJWarp+VBD solver. On reset we re-seed their ``state.body_q`` and
AVBD companions (``body_q_prev`` / ``body_inertia_q``) per env via a single
rigid transform, keeping fixed-joint attachments self-consistent.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _get_body_ids(env: ManagerBasedEnv, cfg: SceneEntityCfg, *, is_cable: bool) -> torch.Tensor:
    """Resolve a scene asset to its per-env Newton body ids.

    Cables come straight from the cable registry as ``(num_envs, num_segments)``;
    rigid assets are resolved by matching their per-env prim path against
    ``model.body_label`` and returned as ``(num_envs,)``. Resolved fresh each
    call — resets are per-episode so the cost is negligible.
    """
    if is_cable:
        entry = env.scene[cfg.name]._registry_entry
        if not entry.segment_body_indices:
            raise RuntimeError(
                f"Cable '{cfg.name}' has no segment_body_indices; the per-world builder hook"
                " did not run. Check that the active solver is a VBD variant."
            )
        return torch.tensor(entry.segment_body_indices, dtype=torch.long, device=env.device)

    from isaaclab_contrib.deformable.vbd_manager import NewtonVBDManager

    model = NewtonVBDManager._model
    prim_path = env.scene[cfg.name].cfg.prim_path  # e.g. "/World/envs/env_.*/Plug"
    pattern = prim_path.replace("env_.*", r"env_(\d+)")
    asset_re = re.compile(rf"^{pattern}$")
    num_envs = env.scene.num_envs
    body_ids = [-1] * num_envs
    for b in range(int(model.body_count)):
        m = asset_re.match(model.body_label[b])
        if m is not None and 0 <= int(m.group(1)) < num_envs:
            body_ids[int(m.group(1))] = b
    missing = [i for i, v in enumerate(body_ids) if v < 0]
    if missing:
        raise RuntimeError(
            f"Could not resolve Newton body for asset '{cfg.name}' (prim_path={prim_path!r})"
            f" in envs {missing[:5]}{'...' if len(missing) > 5 else ''}."
        )
    return torch.tensor(body_ids, dtype=torch.long, device=env.device)


def _sample_rigid_transform(
    pose_range: dict[str, tuple[float, float]],
    num: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``num`` translations [m] and yaw-only quaternions (xyzw) from ``pose_range``.

    Keys ``"x"``, ``"y"``, ``"z"`` give translation ranges and ``"yaw"`` gives
    rotation [rad] about world Z; missing keys default to ``(0.0, 0.0)``.
    """

    def _uniform(key: str) -> torch.Tensor:
        lo, hi = pose_range.get(key, (0.0, 0.0))
        return torch.empty(num, device=device).uniform_(float(lo), float(hi))

    translation = torch.stack([_uniform("x"), _uniform("y"), _uniform("z")], dim=-1)
    half = 0.5 * _uniform("yaw")
    yaw_quat = torch.zeros(num, 4, device=device)
    yaw_quat[:, 2] = torch.sin(half)
    yaw_quat[:, 3] = torch.cos(half)
    return translation, yaw_quat


@wp.kernel(enable_backward=False)
def _scatter_reset_kernel(
    global_ids: wp.array(dtype=int),
    local_ids: wp.array(dtype=int),
    free_q_starts: wp.array(dtype=int),
    new_body_q: wp.array(dtype=wp.transformf),
    state_body_q: wp.array(dtype=wp.transformf),
    state_body_qd: wp.array(dtype=wp.spatial_vectorf),
    state_joint_q: wp.array(dtype=float),
    body_q_prev: wp.array(dtype=wp.transformf),
    body_inertia_q: wp.array(dtype=wp.transformf),
) -> None:
    """Scatter per-body reset state into VBD's parent state and AVBD companions.

    Parent state is indexed by ``global_ids``; the (possibly compacted) solver
    buffers ``body_q_prev`` / ``body_inertia_q`` by ``local_ids`` (negative
    skips). ``body_q_prev == body_q`` zeroes AVBD's velocity finite-difference.
    ``free_q_starts[tid]`` mirrors the pose into ``state_joint_q`` for FREE
    joints (negative skips, e.g. CABLE segments) so reader views don't lag.
    """
    tid = wp.tid()
    gid = global_ids[tid]
    lid = local_ids[tid]
    q = new_body_q[tid]
    state_body_q[gid] = q
    state_body_qd[gid] = wp.spatial_vectorf()
    if lid >= 0:
        body_q_prev[lid] = q
        body_inertia_q[lid] = wp.transformf()
    q0 = free_q_starts[tid]
    if q0 >= 0:
        p = wp.transform_get_translation(q)
        r = wp.transform_get_rotation(q)
        state_joint_q[q0 + 0] = p[0]
        state_joint_q[q0 + 1] = p[1]
        state_joint_q[q0 + 2] = p[2]
        state_joint_q[q0 + 3] = r[0]
        state_joint_q[q0 + 4] = r[1]
        state_joint_q[q0 + 5] = r[2]
        state_joint_q[q0 + 6] = r[3]


def _apply_and_reset(
    env: ManagerBasedEnv,
    body_ids: torch.Tensor,
    delta_trans: torch.Tensor | None = None,
    delta_yaw_quat: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scatter VBD reset state for ``body_ids`` and return their new world poses.

    With a transform, the new pose is the build-time rest pose transformed by
    ``(delta_trans, delta_yaw_quat)`` (cable / anchor / plug); with no transform,
    it is the bodies' current ``state.body_q`` (proxy ``body_q_prev`` flush).

    Args:
        body_ids: Global Newton body ids, any shape (flattened internally).
        delta_trans: Per-env translation [m], or ``None`` to keep the current pose.
        delta_yaw_quat: Per-env yaw quaternion ``(x, y, z, w)``, or ``None``.

    Returns:
        Per-body world poses in ``wp.transformf`` layout.
    """
    from newton import JointType

    from isaaclab_contrib.deformable.vbd_manager import NewtonVBDManager

    model = NewtonVBDManager._model
    state = NewtonVBDManager._state_0
    if model is None or model.body_q is None:
        raise RuntimeError("Newton model is not initialized; cannot resolve body poses.")

    if delta_trans is None:
        # Keep current pose -> scatter sets body_q_prev == body_q (zero proxy velocity).
        new_body_q = wp.to_torch(state.body_q).to(env.device)[body_ids]
    else:
        init_q = wp.to_torch(model.body_q).to(env.device)[body_ids]  # build-time rest pose
        init_pos, init_quat = init_q[..., 0:3], init_q[..., 3:7]
        yaw = delta_yaw_quat.unsqueeze(1).expand(-1, init_pos.shape[1], -1)
        new_pos = quat_apply(yaw, init_pos) + delta_trans.unsqueeze(1)
        new_quat = quat_mul(yaw, init_quat)
        new_body_q = torch.cat([new_pos, new_quat], dim=-1)

    # SolverCoupledProxy indexes body_q_prev / body_inertia_q by compacted local ids
    # (via .solver("dst")); plain SolverVBD has local == global.
    from newton.solvers import SolverVBD

    solver = NewtonVBDManager._solver
    if solver is None:
        raise RuntimeError("VBD solver is not initialized; cannot reset body state.")
    body_count = int(model.body_count)

    flat_global = body_ids.reshape(-1).to(dtype=torch.int32).contiguous()
    flat_q = new_body_q.reshape(-1, 7).contiguous()

    # Per-body joint_q offset for FREE joints (-1 otherwise; e.g. CABLE segments).
    joint_type = wp.to_torch(model.joint_type).to(env.device)
    free = (joint_type == int(JointType.FREE)).nonzero(as_tuple=True)[0]
    body_to_q_start = torch.full((int(model.body_count),), -1, dtype=torch.int32, device=env.device)
    body_to_q_start[wp.to_torch(model.joint_child).to(env.device)[free].long()] = (
        wp.to_torch(model.joint_q_start).to(env.device)[free].to(torch.int32)
    )
    flat_q_starts = body_to_q_start[flat_global.long()].contiguous()

    # Bounds-check globals to fail fast instead of a CUDA assert.
    min_id, max_id = int(flat_global.min().item()), int(flat_global.max().item())
    if min_id < 0 or max_id >= body_count:
        raise IndexError(
            f"reset body state: body id out of range [0, {body_count}); got [{min_id}, {max_id}]."
            " The cached `segment_body_indices` / rigid-body map is likely stale or wrong."
        )

    if hasattr(solver, "solver") and callable(getattr(solver, "solver")):
        vbd_solver = solver.solver("dst")
        if not isinstance(vbd_solver, SolverVBD):
            raise RuntimeError(
                "reset body state: destination entry of the coupled solver is"
                f" {type(vbd_solver).__name__}, not `SolverVBD`. The cable reset path writes to"
                " VBD-specific buffers (`body_q_prev`, `body_inertia_q`); configure"
                " `CoupledProxySolverCfg.dst_solver_cfg` as a `VBDSolverCfg`."
            )
        global_to_local = wp.to_torch(solver._entries["dst"].body_global_to_local)
        flat_local = global_to_local[flat_global.long()].to(dtype=torch.int32).contiguous()
        unowned = (flat_local < 0).nonzero(as_tuple=False).flatten()
        if unowned.numel():
            bad = flat_global[unowned[:5]].tolist()
            raise RuntimeError(
                "reset body state: at least one body is not owned by the destination entry of the"
                f" proxy-coupled solver (first few global ids: {bad}). Ensure cable/anchor/plug are"
                " listed in `CoupledProxySolverCfg.dst_bodies`."
            )
    else:
        if not isinstance(solver, SolverVBD):
            raise RuntimeError(
                f"reset body state: active solver is {type(solver).__name__}, not `SolverVBD` or a"
                " coupled solver with a VBD destination entry. The cable reset path writes to"
                " VBD-specific buffers (`body_q_prev`, `body_inertia_q`)."
            )
        vbd_solver, flat_local = solver, flat_global

    wp.launch(
        _scatter_reset_kernel,
        dim=flat_global.shape[0],
        inputs=[
            wp.from_torch(flat_global, dtype=wp.int32),
            wp.from_torch(flat_local, dtype=wp.int32),
            wp.from_torch(flat_q_starts, dtype=wp.int32),
            wp.from_torch(flat_q, dtype=wp.transformf),
        ],
        outputs=[
            state.body_q,
            state.body_qd,
            state.joint_q,
            vbd_solver.body_q_prev,
            vbd_solver.body_inertia_q,
        ],
        device=state.body_q.device,
    )
    NewtonVBDManager._mark_state_dirty()
    return new_body_q


def reset_cable_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    cable_cfg: SceneEntityCfg,
) -> None:
    """Reset a free cable by re-seeding its VBD state with a per-env rigid transform.

    Args:
        env: The RL environment.
        env_ids: Environment indices to reset.
        pose_range: Per-axis uniform ranges. Keys ``"x"``, ``"y"``, ``"z"``
            give translation [m] ranges; ``"yaw"`` gives rotation [rad] about
            world Z. Missing keys default to ``(0.0, 0.0)``.
        cable_cfg: Scene-entity reference to the :class:`CableObject`.
    """
    from isaaclab_contrib.deformable.vbd_manager import NewtonVBDManager

    delta_trans, delta_yaw_quat = _sample_rigid_transform(pose_range, env_ids.shape[0], env.device)
    body_ids = _get_body_ids(env, cable_cfg, is_cable=True)[env_ids]  # (n_envs, n_seg)
    _apply_and_reset(env, body_ids, delta_trans, delta_yaw_quat)
    # Flag curve buffers dirty so the next render rebuilds points from the new body_q.
    NewtonVBDManager._mark_curves_dirty()


def reset_proxy_body_prev(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
) -> None:
    """Snap proxy bodies' ``body_q_prev`` onto their post-reset pose.

    The proxy-coupled solver drives ``(body_q - body_q_prev) / dt`` into the
    cable each step, so a stale ``body_q_prev`` after the arm-joint reset
    teleports the gripper flings the cable away. Must run after
    ``reset_robot_joints``; no-op unless the solver is proxy-coupled.

    Args:
        env: The RL environment.
        env_ids: Environment indices to reset.
    """
    from isaaclab_contrib.deformable.vbd_manager import NewtonVBDManager

    solver = NewtonVBDManager._solver
    mappings = getattr(solver, "_proxy_mappings", None)
    if not mappings:  # not proxy-coupled, or no proxies configured
        return

    # Refresh body_q from the just-reset joint_q (eval_fk skips VBD-owned cable).
    NewtonVBDManager.forward()
    proxy_ids = torch.cat([wp.to_torch(m.src_body_ids) for m in mappings]).to(env.device).long()
    # Scope to the reset envs (one world per env).
    body_world = NewtonVBDManager._model.body_world
    if body_world is not None:
        world = wp.to_torch(body_world).to(env.device)[proxy_ids]
        proxy_ids = proxy_ids[torch.isin(world, env_ids.to(env.device))]
    if proxy_ids.numel():
        _apply_and_reset(env, proxy_ids)


def reset_cable_assembly_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    cable_cfg: SceneEntityCfg,
    anchor_cfg: SceneEntityCfg,
    plug_cfg: SceneEntityCfg,
) -> None:
    """Reset an anchor+cable+plug assembly with one rigid transform per env.

    The same sampled transform is applied to the cable, the anchor, and the
    plug, so the Newton fixed-joint attachments retain their build-time rest
    geometry exactly — the solver sees zero constraint residual on the next
    step and the assembly does not snap.

    Args:
        env: The RL environment.
        env_ids: Environment indices to reset.
        pose_range: Per-axis uniform ranges; see :func:`reset_cable_uniform`.
        cable_cfg: Scene-entity reference to the :class:`CableObject`.
        anchor_cfg: Scene-entity reference to the kinematic anchor :class:`RigidObject`.
        plug_cfg: Scene-entity reference to the rigid plug :class:`RigidObject`.
    """
    from isaaclab_contrib.deformable.vbd_manager import NewtonVBDManager

    delta_trans, delta_yaw_quat = _sample_rigid_transform(pose_range, env_ids.shape[0], env.device)
    cable_ids = _get_body_ids(env, cable_cfg, is_cable=True)[env_ids]  # (n_envs, n_seg)
    _apply_and_reset(env, cable_ids, delta_trans, delta_yaw_quat)
    NewtonVBDManager._mark_curves_dirty()
    for rigid_cfg in (anchor_cfg, plug_cfg):
        rigid_ids = _get_body_ids(env, rigid_cfg, is_cable=False)[env_ids].unsqueeze(-1)  # (n_envs, 1)
        new_q = _apply_and_reset(env, rigid_ids, delta_trans, delta_yaw_quat)
        # Mirror the new pose into IsaacLab's RigidObjectData buffer so per-env
        # observations don't see one frame of stale pose.
        env.scene[rigid_cfg.name].write_root_link_pose_to_sim_index(
            root_pose=new_q.squeeze(1).contiguous(), env_ids=env_ids
        )
