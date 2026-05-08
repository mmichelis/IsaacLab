# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms for the Franka deformable lifting environment."""

from __future__ import annotations

import torch
import warp as wp

from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms


_ACTION_SCALE = (0.04, 0.04, 0.02)
_RESIDUAL_SCALE = 0.02
_JOINT_RESIDUAL_SCALE = 0.02
_GRASP_HEIGHT = 0.02
_HOVER_HEIGHT = 0.14
_LIFT_HEIGHT = 0.13
_HOVER_STEPS = 60
_REACH_STEPS = 130
_CLOSE_STEPS = 170
_GOAL_STEPS = 250
_END_STEPS = 290
_DESCEND_STEPS = _REACH_STEPS - _HOVER_STEPS
_LIFT_STEPS = _GOAL_STEPS - _CLOSE_STEPS
_GOAL_MOVE_STEPS = _END_STEPS - _GOAL_STEPS
_GRIPPER_CLOSE_STEPS = _CLOSE_STEPS - _REACH_STEPS
_EE_KP = 4.0
_EE_KD = 0.2
_EE_DAMPING = 0.1
_MAX_JOINT_STEP = 0.035


def _smoothstep(alpha: torch.Tensor) -> torch.Tensor:
    """Return a smooth interpolation factor in ``[0, 1]``."""
    alpha = torch.clamp(alpha, min=0.0, max=1.0)
    return alpha * alpha * (3.0 - 2.0 * alpha)


def _body_pos_w(robot, body_id: int) -> torch.Tensor:
    """Return a rigid-body position in world frame."""
    if hasattr(robot.data, "body_pos_w"):
        return robot.data.body_pos_w.torch[:, body_id]
    if hasattr(robot.data, "body_link_pos_w"):
        return robot.data.body_link_pos_w.torch[:, body_id]
    return robot.data.body_com_pos_w.torch[:, body_id]


def _damped_least_squares(jacobian: torch.Tensor, velocity: torch.Tensor, damping: float) -> torch.Tensor:
    """Solve ``J qdot = velocity`` with damped least squares."""
    jacobian_t = jacobian.transpose(1, 2)
    identity = torch.eye(jacobian.shape[1], device=jacobian.device, dtype=jacobian.dtype).unsqueeze(0)
    lhs = jacobian @ jacobian_t + (damping**2) * identity
    return (jacobian_t @ torch.linalg.solve(lhs, velocity.unsqueeze(-1))).squeeze(-1)


class _TimedScriptedPrior:
    """Mixin implementing the timed scripted deformable grasp prior."""

    def _init_scripted_prior(self) -> None:
        self._scripted_step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._scripted_initial_com_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._scripted_initial_com_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _reset_scripted_prior(self, env_ids) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._scripted_step_count[env_ids] = 0
        self._scripted_initial_com_valid[env_ids] = False

    def _scripted_target_action(self) -> torch.Tensor:
        deformable = self._env.scene["deformable"]
        robot = self._env.scene["robot"]
        ee_frame = self._env.scene["ee_frame"]

        root_pos_w = robot.data.root_pos_w.torch
        root_quat_w = robot.data.root_quat_w.torch
        com_w = wp.to_torch(deformable.data.root_pos_w)
        com_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, com_w)
        reset_mask = ~self._scripted_initial_com_valid
        if reset_mask.any():
            self._scripted_initial_com_w[reset_mask] = com_b[reset_mask]
            self._scripted_initial_com_valid[reset_mask] = True

        initial_com_b = self._scripted_initial_com_w
        hover_b = initial_com_b + torch.tensor([0.0, 0.0, _HOVER_HEIGHT], device=self.device)
        grasp_b = initial_com_b + torch.tensor([0.0, 0.0, _GRASP_HEIGHT], device=self.device)
        lift_b = initial_com_b + torch.tensor([0.0, 0.0, _LIFT_HEIGHT], device=self.device)

        command = self._env.command_manager.get_command("deformable_pose")
        goal_b = command[:, :3]

        step_count = self._scripted_step_count
        target_b = torch.where((step_count < _HOVER_STEPS).unsqueeze(1), hover_b, grasp_b)
        target_b = torch.where((step_count >= _CLOSE_STEPS).unsqueeze(1), lift_b, target_b)
        target_b = torch.where((step_count >= _GOAL_STEPS).unsqueeze(1), goal_b, target_b)

        ee_w = ee_frame.data.target_pos_w.torch[..., 0, :]
        ee_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, ee_w)
        scale = torch.tensor(_ACTION_SCALE, device=self.device).unsqueeze(0)
        delta_b = target_b - ee_b
        arm_action = torch.clamp(delta_b / scale, min=-1.0, max=1.0)
        gripper_action = torch.where(
            step_count < _REACH_STEPS,
            torch.ones(self.num_envs, device=self.device),
            -torch.ones(self.num_envs, device=self.device),
        )
        finite = torch.isfinite(arm_action).all(dim=1) & torch.isfinite(gripper_action)
        arm_action = torch.where(finite.unsqueeze(1), arm_action, torch.zeros_like(arm_action))
        gripper_action = torch.where(finite, gripper_action, torch.ones_like(gripper_action))
        self._scripted_step_count += 1
        return torch.cat((arm_action, gripper_action.unsqueeze(1)), dim=1)


class ScriptedResidualJointPositionAction(JointPositionAction, _TimedScriptedPrior):
    """Joint-position action around a scripted end-effector PD grasp/lift prior."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._init_scripted_prior()
        self._scripted_joint_pos_command = self._asset.data.joint_pos.torch[:, self._joint_ids].clone()
        self._scripted_prev_ee_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._scripted_prev_target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._scripted_initial_ee_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._scripted_initial_com_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._scripted_joint_prior_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        hand_body_ids, _ = self._asset.find_bodies("panda_hand")
        self._scripted_hand_body_id = hand_body_ids[0]
        self._scripted_hand_jacobian_id = (
            self._scripted_hand_body_id - 1 if self._asset.is_fixed_base else self._scripted_hand_body_id
        )
        if isinstance(self._joint_ids, slice):
            self._scripted_arm_joint_ids = list(range(self._asset.num_joints))[self._joint_ids]
        else:
            self._scripted_arm_joint_ids = list(self._joint_ids)

    def _scripted_point_jacobian_w(self, ee_pos_w: torch.Tensor) -> torch.Tensor:
        """Compute the translational Jacobian of the end-effector point in world frame."""
        if hasattr(self._asset.root_view, "get_jacobians"):
            jacobian = wp.to_torch(self._asset.root_view.get_jacobians())[
                :, self._scripted_hand_jacobian_id, :, self._scripted_arm_joint_ids
            ]
        elif hasattr(self._asset.root_view, "eval_jacobian"):
            from isaaclab_newton.physics import NewtonManager as SimulationManager

            jacobian = wp.to_torch(self._asset.root_view.eval_jacobian(SimulationManager.get_state_0()))
            jacobian = jacobian.reshape(jacobian.shape[0], jacobian.shape[1] // 6, 6, jacobian.shape[2])
            jacobian = jacobian[:, self._scripted_hand_body_id, :, self._scripted_arm_joint_ids]
        else:
            raise AttributeError("Articulation root view does not provide a supported Jacobian API.")

        body_pos_w = _body_pos_w(self._asset, self._scripted_hand_body_id)
        point_offset_w = ee_pos_w - body_pos_w
        lin_jacobian = jacobian[:, :3, :]
        ang_jacobian = jacobian[:, 3:, :]
        return lin_jacobian + torch.cross(
            ang_jacobian.transpose(1, 2), point_offset_w.unsqueeze(1), dim=-1
        ).transpose(1, 2)

    def _scripted_target_pos_w(self, ee_pos_w: torch.Tensor) -> torch.Tensor:
        """Compute the current scripted end-effector position target."""
        deformable = self._env.scene["deformable"]
        robot = self._env.scene["robot"]
        com_w = wp.to_torch(deformable.data.root_pos_w)
        reset_mask = ~self._scripted_joint_prior_valid
        if reset_mask.any():
            current_joint_pos = self._asset.data.joint_pos.torch[:, self._joint_ids]
            self._scripted_initial_com_pos_w[reset_mask] = com_w[reset_mask]
            self._scripted_initial_ee_pos_w[reset_mask] = ee_pos_w[reset_mask]
            self._scripted_prev_ee_pos_w[reset_mask] = ee_pos_w[reset_mask]
            self._scripted_joint_pos_command[reset_mask] = current_joint_pos[reset_mask]
            self._scripted_joint_prior_valid[reset_mask] = True

        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, -1)
        hover_pos_w = self._scripted_initial_com_pos_w + _HOVER_HEIGHT * z_axis
        grasp_pos_w = self._scripted_initial_com_pos_w + _GRASP_HEIGHT * z_axis
        lift_pos_w = self._scripted_initial_com_pos_w + _LIFT_HEIGHT * z_axis

        command = self._env.command_manager.get_command("deformable_pose")
        goal_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, command[:, :3])
        goal_pos_w = goal_pos_w.clone()
        goal_pos_w[:, 2] += _GRASP_HEIGHT
        goal_pos_w[:, 2] = torch.maximum(goal_pos_w[:, 2], lift_pos_w[:, 2])

        step_count = self._scripted_step_count.to(dtype=torch.float32)
        hover_alpha = _smoothstep(step_count / max(_HOVER_STEPS, 1))
        descend_alpha = _smoothstep((step_count - _HOVER_STEPS) / max(_DESCEND_STEPS, 1))
        lift_alpha = _smoothstep((step_count - _CLOSE_STEPS) / max(_LIFT_STEPS, 1))
        goal_alpha = _smoothstep((step_count - _GOAL_STEPS) / max(_GOAL_MOVE_STEPS, 1))

        target_pos_w = torch.lerp(self._scripted_initial_ee_pos_w, hover_pos_w, hover_alpha.unsqueeze(1))
        descend_target_w = torch.lerp(hover_pos_w, grasp_pos_w, descend_alpha.unsqueeze(1))
        lift_target_w = torch.lerp(grasp_pos_w, lift_pos_w, lift_alpha.unsqueeze(1))
        goal_target_w = torch.lerp(lift_pos_w, goal_pos_w, goal_alpha.unsqueeze(1))
        target_pos_w = torch.where((step_count >= _HOVER_STEPS).unsqueeze(1), descend_target_w, target_pos_w)
        target_pos_w = torch.where((step_count >= _CLOSE_STEPS).unsqueeze(1), lift_target_w, target_pos_w)
        target_pos_w = torch.where((step_count >= _GOAL_STEPS).unsqueeze(1), goal_target_w, target_pos_w)
        if reset_mask.any():
            self._scripted_prev_target_pos_w[reset_mask] = target_pos_w[reset_mask]
        return target_pos_w

    def process_actions(self, actions: torch.Tensor):
        """Blend policy residual joint targets with the scripted grasp/lift prior."""
        self._raw_actions[:] = actions

        ee_frame = self._env.scene["ee_frame"]
        ee_pos_w = ee_frame.data.target_pos_w.torch[..., 0, :]
        target_pos_w = self._scripted_target_pos_w(ee_pos_w)
        target_vel_w = (target_pos_w - self._scripted_prev_target_pos_w) / self._env.step_dt
        ee_vel_w = (ee_pos_w - self._scripted_prev_ee_pos_w) / self._env.step_dt
        ee_vel_des_w = _EE_KP * (target_pos_w - ee_pos_w) + _EE_KD * (target_vel_w - ee_vel_w)

        jacobian = self._scripted_point_jacobian_w(ee_pos_w)
        joint_vel_target = _damped_least_squares(jacobian, ee_vel_des_w, _EE_DAMPING)
        joint_step = torch.clamp(joint_vel_target * self._env.step_dt, min=-_MAX_JOINT_STEP, max=_MAX_JOINT_STEP)
        joint_step = torch.nan_to_num(joint_step, nan=0.0, posinf=0.0, neginf=0.0)

        joint_pos_limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids, :]
        self._scripted_joint_pos_command[:] = torch.clamp(
            self._scripted_joint_pos_command + joint_step,
            joint_pos_limits[..., 0],
            joint_pos_limits[..., 1],
        )
        residual = _JOINT_RESIDUAL_SCALE * torch.clamp(actions, min=-1.0, max=1.0)
        self._processed_actions[:] = torch.clamp(
            self._scripted_joint_pos_command + residual,
            joint_pos_limits[..., 0],
            joint_pos_limits[..., 1],
        )

        finite = torch.isfinite(self._processed_actions).all(dim=1)
        if (~finite).any():
            current_joint_pos = self._asset.data.joint_pos.torch[:, self._joint_ids]
            self._processed_actions[~finite] = current_joint_pos[~finite]
            self._scripted_joint_pos_command[~finite] = current_joint_pos[~finite]

        self._scripted_prev_ee_pos_w[:] = torch.nan_to_num(ee_pos_w, nan=0.0, posinf=0.0, neginf=0.0)
        self._scripted_prev_target_pos_w[:] = torch.nan_to_num(target_pos_w, nan=0.0, posinf=0.0, neginf=0.0)
        self._scripted_step_count += 1

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        self._reset_scripted_prior(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._scripted_joint_pos_command[env_ids] = self._asset.data.joint_pos.torch[:, self._joint_ids][env_ids]
        self._scripted_joint_prior_valid[env_ids] = False


class ScriptedResidualDifferentialInverseKinematicsAction(DifferentialInverseKinematicsAction, _TimedScriptedPrior):
    """Differential IK action around the scripted deformable grasp action prior."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._init_scripted_prior()

    def process_actions(self, actions: torch.Tensor):
        """Blend policy residual arm actions with the scripted grasp prior."""
        target_actions = self._scripted_target_action()[:, : self.action_dim]
        residual_actions = torch.clamp(target_actions + _RESIDUAL_SCALE * actions, min=-1.0, max=1.0)
        super().process_actions(residual_actions)

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        self._reset_scripted_prior(env_ids)


class ScriptedResidualBinaryJointPositionAction(BinaryJointPositionAction, _TimedScriptedPrior):
    """Binary gripper action around the scripted deformable grasp action prior."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._init_scripted_prior()

    def process_actions(self, actions: torch.Tensor):
        """Ramp the gripper from open to closed during the scripted grasp hold."""
        step_count = self._scripted_step_count
        closing = step_count >= _REACH_STEPS
        target_actions = torch.where(
            closing, -torch.ones(self.num_envs, device=self.device), torch.ones(self.num_envs, device=self.device)
        )
        self._raw_actions[:] = target_actions.unsqueeze(1)

        close_alpha = _smoothstep((step_count.to(dtype=torch.float32) - _REACH_STEPS) / max(_GRIPPER_CLOSE_STEPS, 1))
        command = torch.lerp(
            self._open_command.unsqueeze(0).expand(self.num_envs, -1),
            self._close_command.unsqueeze(0).expand(self.num_envs, -1),
            close_alpha.unsqueeze(1),
        )
        self._processed_actions[:] = torch.nan_to_num(command, nan=0.05, posinf=0.05, neginf=0.05)
        self._scripted_step_count += 1

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        self._reset_scripted_prior(env_ids)
