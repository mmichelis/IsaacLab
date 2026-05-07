# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the Lift Soft Franka environment with a simple end-effector position PD controller.

The controller tracks a scripted end-effector position and sends a binary
open/close command to the gripper. It solves a damped least-squares joint update
from the end-effector translational Jacobian, then feeds the result through the
environment's joint-position action term.

Usage::

    ./isaaclab.sh -p scripts/demos/franka_deformable_ee_pd.py --num_envs 1 --visualizer newton

"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import warp as wp

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.math import combine_frame_transforms
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config

TASK = "Isaac-Lift-Soft-Franka-v0"

parser = argparse.ArgumentParser(description="Run Lift Soft Franka with an end-effector position PD controller.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--task", type=str, default=TASK, help="Task name.")
parser.add_argument(
    "--max_steps",
    type=int,
    default=360,
    help="Maximum number of environment steps. Defaults to one episode without a visualizer.",
)
parser.add_argument("--episode_length_s", type=float, default=12.0, help="Episode length [s].")
parser.add_argument("--cycle_steps", type=int, default=360, help="Controller phase cycle length [env steps].")
parser.add_argument("--kp", type=float, default=0.7, help="End-effector position proportional gain [1/s].")
parser.add_argument("--kd", type=float, default=0.3, help="End-effector position derivative gain.")
parser.add_argument("--damping", type=float, default=0.05, help="Damped least-squares Jacobian damping.")
parser.add_argument("--max_joint_step", type=float, default=0.04, help="Maximum joint command step [rad/env step].")
parser.add_argument("--approach_steps", type=int, default=90, help="Steps used to blend from the reset pose to hover.")
parser.add_argument("--hover_height", type=float, default=0.22, help="Hover target above the deformable COM [m].")
parser.add_argument("--grasp_height", type=float, default=-0.02, help="Grasp target above the deformable COM [m].")
parser.add_argument("--print_interval", type=int, default=30, help="Print tracking error every N env steps.")
parser.add_argument("--record_video", action="store_true", default=False, help="Record Kit camera frames.")
parser.add_argument(
    "--record_dir",
    type=str,
    default=os.path.join("scripts", "demos", "output", "franka_deformable_ee_pd"),
    help="Directory where recording frames and video are written.",
)
parser.add_argument("--record_name", type=str, default="franka_deformable_ee_pd.mp4", help="Output video filename.")
parser.add_argument("--record_fps", type=int, default=60, help="Frame rate for the ffmpeg output.")
parser.add_argument("--record_width", type=int, default=1920, help="Recording width in pixels.")
parser.add_argument("--record_height", type=int, default=1080, help="Recording height in pixels.")
parser.add_argument("--record_every", type=int, default=1, help="Save one frame every N environment steps.")
parser.add_argument("--record_warmup", type=int, default=2, help="Warmup frames to render before saving.")
parser.add_argument(
    "--record_camera_prim_path",
    type=str,
    default="/World/RecordingCamera",
    help="USD camera prim used for recording.",
)
parser.add_argument(
    "--record_camera_position",
    type=float,
    nargs=3,
    default=(2.2, 0.8, 0.9),
    metavar=("X", "Y", "Z"),
    help="Recording camera position [m].",
)
parser.add_argument(
    "--record_camera_target",
    type=float,
    nargs=3,
    default=(-0.25, -0.2, 0.0),
    metavar=("X", "Y", "Z"),
    help="Recording camera look-at target [m].",
)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args


def _as_float3(values: tuple[float, float, float] | list[float]) -> tuple[float, float, float]:
    """Convert a three-value CLI sequence to a float tuple."""
    return (float(values[0]), float(values[1]), float(values[2]))


def _ensure_kit_recording_launch_args() -> None:
    """Ensure recording launches Kit and enables camera rendering."""
    if not args_cli.record_video:
        return

    args_cli.enable_cameras = True
    visualizer = getattr(args_cli, "visualizer", None)
    if visualizer is None:
        tokens = []
    elif isinstance(visualizer, str):
        tokens = [token.strip().lower() for token in visualizer.split(",") if token.strip()]
    else:
        tokens = [str(token).strip().lower() for token in visualizer if str(token).strip()]

    tokens = [token for token in tokens if token != "none"]
    if "kit" not in tokens:
        tokens.append("kit")
    args_cli.visualizer = ",".join(tokens)


def _prepare_recording_paths() -> tuple[Path, Path]:
    """Prepare frame and video output paths."""
    record_dir = Path(args_cli.record_dir).expanduser()
    if not record_dir.is_absolute():
        record_dir = Path.cwd() / record_dir

    frames_dir = record_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for frame_path in frames_dir.glob("frame_*.png"):
        frame_path.unlink()

    video_path = Path(args_cli.record_name).expanduser()
    if not video_path.is_absolute():
        video_path = record_dir / video_path
    video_path.parent.mkdir(parents=True, exist_ok=True)
    return frames_dir, video_path


def _create_recording_camera():
    """Spawn a Kit USD camera and return an RGB capture helper."""
    from isaaclab_physx.video_recording.isaacsim_kit_perspective_video import create_isaacsim_kit_perspective_video
    from isaaclab_physx.video_recording.isaacsim_kit_perspective_video_cfg import IsaacsimKitPerspectiveVideoCfg

    import isaaclab.sim as sim_utils

    camera_cfg = sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.1, 100.0),
    )
    camera_cfg.func(args_cli.record_camera_prim_path, camera_cfg)

    capture_cfg = IsaacsimKitPerspectiveVideoCfg(
        camera_prim_path=args_cli.record_camera_prim_path,
        eye=_as_float3(args_cli.record_camera_position),
        lookat=_as_float3(args_cli.record_camera_target),
        window_width=args_cli.record_width,
        window_height=args_cli.record_height,
    )
    return create_isaacsim_kit_perspective_video(capture_cfg)


def _save_rgb_frame(frame: np.ndarray, file_path: Path) -> None:
    """Save an RGB frame as a PNG image."""
    from isaaclab.sensors import save_images_to_file

    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[-1] < 3:
        raise RuntimeError(f"Expected an RGB frame with shape (H, W, C), got {frame.shape}.")
    frame = frame[..., :3]
    if frame.dtype != np.uint8:
        if frame.size > 0 and float(np.nanmax(frame)) <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    image = torch.from_numpy(np.ascontiguousarray(frame)).to(dtype=torch.float32).unsqueeze(0) / 255.0
    save_images_to_file(image, str(file_path))


def _run_ffmpeg(frames_dir: Path, video_path: Path, frame_count: int) -> None:
    """Use ffmpeg to turn saved PNG frames into an MP4 animation."""
    if frame_count == 0:
        print("[WARN]: No frames were recorded; skipping ffmpeg.")
        return

    ffmpeg_path = shutil.which("ffmpeg")
    input_pattern = frames_dir / "frame_%06d.png"
    cmd = [
        ffmpeg_path or "ffmpeg",
        "-y",
        "-framerate",
        str(args_cli.record_fps),
        "-start_number",
        "0",
        "-i",
        str(input_pattern),
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(video_path),
    ]

    if ffmpeg_path is None:
        print(f"[WARN]: ffmpeg was not found. Frames are saved in: {frames_dir}")
        print(f"[WARN]: Install ffmpeg and run: {shlex.join(cmd)}")
        return

    try:
        subprocess.run(cmd, check=True)
        print(f"[INFO]: Saved video to: {video_path}")
    except subprocess.CalledProcessError:
        print(f"[WARN]: ffmpeg failed. Frames are saved in: {frames_dir}")
        print(f"[WARN]: Failed command: {shlex.join(cmd)}")


def _to_torch(value) -> torch.Tensor:
    """Return a torch view for Isaac Lab tensor wrappers or Warp arrays."""
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "torch"):
        return value.torch
    return wp.to_torch(value)


def _smoothstep(x: torch.Tensor) -> torch.Tensor:
    """Smooth interpolation weight on ``[0, 1]``."""
    x = torch.clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _action_term_slice(env, term_name: str) -> slice:
    """Return the column slice for an action-manager term."""
    start = 0
    for name, dim in zip(env.action_manager.active_terms, env.action_manager.action_term_dim):
        stop = start + dim
        if name == term_name:
            return slice(start, stop)
        start = stop
    raise KeyError(f"Action term {term_name!r} was not found. Active terms: {env.action_manager.active_terms}.")


def _joint_position_to_raw_action(arm_action, joint_pos_target: torch.Tensor) -> torch.Tensor:
    """Convert arm joint targets [rad] into the raw joint-position action space."""
    raw_action = joint_pos_target - arm_action._offset
    if isinstance(arm_action._scale, torch.Tensor):
        raw_action = raw_action / arm_action._scale
    else:
        raw_action = raw_action / float(arm_action._scale)
    return raw_action


def _body_pos_w(robot, body_id: int) -> torch.Tensor:
    """Return a body reference position in world frame [m]."""
    if hasattr(robot.data, "body_pos_w"):
        return _to_torch(robot.data.body_pos_w)[:, body_id]
    if hasattr(robot.data, "body_link_pos_w"):
        return _to_torch(robot.data.body_link_pos_w)[:, body_id]
    return _to_torch(robot.data.body_com_pos_w)[:, body_id]


def _ee_position_jacobian(
    robot,
    hand_body_id: int,
    hand_jacobian_id: int,
    arm_joint_ids: list[int],
    ee_pos_w: torch.Tensor,
) -> torch.Tensor:
    """Compute the linear Jacobian for the end-effector point in world frame."""
    if hasattr(robot.root_view, "get_jacobians"):
        jacobian = _to_torch(robot.root_view.get_jacobians())[:, hand_jacobian_id, :, arm_joint_ids]
    elif hasattr(robot.root_view, "eval_jacobian"):
        from isaaclab_newton.physics import NewtonManager as SimulationManager

        jacobian = _to_torch(robot.root_view.eval_jacobian(SimulationManager.get_state_0()))
        jacobian = jacobian.reshape(jacobian.shape[0], jacobian.shape[1] // 6, 6, jacobian.shape[2])
        jacobian = jacobian[:, hand_body_id, :, arm_joint_ids]
    else:
        raise AttributeError("The robot root view does not provide a supported Jacobian API.")

    body_pos_w = _body_pos_w(robot, hand_body_id)
    lin_jacobian = jacobian[:, :3, :]
    ang_jacobian = jacobian[:, 3:, :]
    point_offset_w = ee_pos_w - body_pos_w
    point_jacobian = lin_jacobian + torch.cross(
        ang_jacobian.transpose(1, 2), point_offset_w.unsqueeze(1), dim=-1
    ).transpose(1, 2)
    return point_jacobian


def _solve_damped_least_squares(jacobian: torch.Tensor, velocity: torch.Tensor, damping: float) -> torch.Tensor:
    """Solve ``J qdot = velocity`` with damped least squares."""
    jacobian_t = jacobian.transpose(1, 2)
    identity = torch.eye(jacobian.shape[1], device=jacobian.device, dtype=jacobian.dtype).unsqueeze(0)
    lhs = jacobian @ jacobian_t + (damping**2) * identity
    return (jacobian_t @ torch.linalg.solve(lhs, velocity.unsqueeze(-1))).squeeze(-1)


def _deformable_com_w(env) -> torch.Tensor:
    """Return the deformable COM in world frame [m]."""
    return _to_torch(env.scene["deformable"].data.root_pos_w)


def _deformable_goal_w(env) -> torch.Tensor:
    """Return the commanded deformable goal in world frame [m]."""
    robot = env.scene["robot"]
    command = env.command_manager.get_command("deformable_pose")
    goal_pos_w, _ = combine_frame_transforms(
        _to_torch(robot.data.root_pos_w), _to_torch(robot.data.root_quat_w), command[:, :3]
    )
    return goal_pos_w


def _target_ee_pos_w(env, step_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the scripted end-effector position target and gripper command."""
    device = env.device
    deformable_pos_w = _deformable_com_w(env)
    goal_pos_w = _deformable_goal_w(env)
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=device).expand(env.num_envs, -1)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device).expand(env.num_envs, -1)

    hover_pos_w = deformable_pos_w + args_cli.hover_height * z_axis
    grasp_pos_w = deformable_pos_w + args_cli.grasp_height * z_axis #+ 0.05 * y_axis
    lift_pos_w = goal_pos_w

    phase = (step_count % max(args_cli.cycle_steps, 1)) / max(args_cli.cycle_steps, 1)
    if phase < 0.25:
        ee_target_pos_w = hover_pos_w
        gripper_action = torch.ones(env.num_envs, 1, device=device)
    elif phase < 0.55:
        alpha = torch.tensor((phase - 0.25) / 0.20, device=device)
        ee_target_pos_w = torch.lerp(hover_pos_w, grasp_pos_w, _smoothstep(alpha))
        gripper_action = torch.ones(env.num_envs, 1, device=device)
    elif phase < 0.65:
        ee_target_pos_w = grasp_pos_w
        gripper_action = -torch.ones(env.num_envs, 1, device=device)
    elif phase < 0.85:
        alpha = torch.tensor((phase - 0.58) / 0.27, device=device)
        ee_target_pos_w = torch.lerp(grasp_pos_w, lift_pos_w, _smoothstep(alpha))
        gripper_action = -torch.ones(env.num_envs, 1, device=device)
    else:
        ee_target_pos_w = lift_pos_w
        gripper_action = -torch.ones(env.num_envs, 1, device=device)

    return ee_target_pos_w, gripper_action


def main():
    """Run the end-effector position PD demo."""
    torch.manual_seed(42)

    env_cfg, _ = resolve_task_config(args_cli.task, "")
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.episode_length_s = args_cli.episode_length_s
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.record_video:
        _ensure_kit_recording_launch_args()
        env_cfg.viewer.eye = _as_float3(args_cli.record_camera_position)
        env_cfg.viewer.lookat = _as_float3(args_cli.record_camera_target)
        env_cfg.viewer.resolution = (args_cli.record_width, args_cli.record_height)

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg)
        env.reset()
        unwrapped = env.unwrapped
        device = unwrapped.device
        sim = unwrapped.sim
        record_capture = None
        frames_dir = None
        video_path = None
        frame_count = 0

        robot = unwrapped.scene["robot"]
        ee_frame = unwrapped.scene["ee_frame"]
        arm_action = unwrapped.action_manager.get_term("arm_action")
        arm_action_slice = _action_term_slice(unwrapped, "arm_action")
        gripper_action_slice = _action_term_slice(unwrapped, "gripper_action")
        if isinstance(arm_action._joint_ids, slice):
            arm_joint_ids = list(range(robot.num_joints))[arm_action._joint_ids]
        else:
            arm_joint_ids = list(arm_action._joint_ids)
        hand_body_id = robot.find_bodies("panda_hand")[0][0]
        hand_jacobian_id = hand_body_id - 1 if robot.is_fixed_base else hand_body_id
        ee_pos_w = _to_torch(ee_frame.data.target_pos_w)[:, 0, :]
        ee_target_pos_w, _ = _target_ee_pos_w(unwrapped, 0)
        initial_ee_pos_w = ee_pos_w.clone()
        prev_ee_pos_w = ee_pos_w.clone()
        prev_ee_target_pos_w = initial_ee_pos_w.clone() if args_cli.approach_steps > 0 else ee_target_pos_w.clone()
        joint_pos_command = _to_torch(robot.data.joint_pos)[:, arm_joint_ids].clone()

        action = torch.zeros(env.action_space.shape, device=device)
        max_steps = args_cli.max_steps
        if (args_cli.record_video or not sim.visualizers) and max_steps is None:
            max_steps = unwrapped.max_episode_length

        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")
        print("[INFO]: Tracking end-effector position with PD control and binary gripper commands.")

        step_count = 0
        try:
            if args_cli.record_video:
                frames_dir, video_path = _prepare_recording_paths()
                record_capture = _create_recording_camera()
                sim.set_camera_view(
                    _as_float3(args_cli.record_camera_position), _as_float3(args_cli.record_camera_target)
                )
                for _ in range(max(args_cli.record_warmup, 0)):
                    record_capture.render_rgb_array()
                print(f"[INFO]: Recording frames to: {frames_dir}")

            while True:
                if max_steps is not None and step_count >= max_steps:
                    break
                if sim.visualizers and not any(v.is_running() and not v.is_closed for v in sim.visualizers):
                    break

                with torch.inference_mode():
                    ee_pos_w = _to_torch(ee_frame.data.target_pos_w)[:, 0, :]
                    ee_target_pos_w, gripper_action = _target_ee_pos_w(unwrapped, step_count)
                    if step_count < args_cli.approach_steps:
                        alpha = torch.tensor(step_count / max(args_cli.approach_steps, 1), device=device)
                        ee_target_pos_w = torch.lerp(initial_ee_pos_w, ee_target_pos_w, _smoothstep(alpha))
                        gripper_action = torch.ones_like(gripper_action)

                    ee_vel_w = (ee_pos_w - prev_ee_pos_w) / unwrapped.step_dt
                    ee_target_vel_w = (ee_target_pos_w - prev_ee_target_pos_w) / unwrapped.step_dt
                    ee_pos_error_w = ee_target_pos_w - ee_pos_w
                    ee_vel_des_w = args_cli.kp * ee_pos_error_w + args_cli.kd * (ee_target_vel_w - ee_vel_w)

                    jacobian = _ee_position_jacobian(robot, hand_body_id, hand_jacobian_id, arm_joint_ids, ee_pos_w)
                    joint_vel_target = _solve_damped_least_squares(jacobian, ee_vel_des_w, args_cli.damping)
                    joint_step = torch.clamp(
                        joint_vel_target * unwrapped.step_dt,
                        min=-args_cli.max_joint_step,
                        max=args_cli.max_joint_step,
                    )
                    joint_pos_limits = _to_torch(robot.data.soft_joint_pos_limits)[:, arm_joint_ids, :]
                    joint_pos_command = torch.clamp(
                        joint_pos_command + joint_step, joint_pos_limits[..., 0], joint_pos_limits[..., 1]
                    )

                    action.zero_()
                    action[:, arm_action_slice] = _joint_position_to_raw_action(arm_action, joint_pos_command)
                    action[:, gripper_action_slice] = gripper_action

                    _, _, terminated, truncated, _ = env.step(action)

                    if args_cli.print_interval > 0 and step_count % args_cli.print_interval == 0:
                        mean_error = torch.linalg.norm(ee_pos_error_w, dim=-1).mean().item()
                        gripper = "close" if torch.mean(gripper_action) < 0 else "open"
                        print(f"[INFO]: step={step_count:05d} ee_pos_error={mean_error:.4f} m gripper={gripper}")

                    prev_ee_pos_w = ee_pos_w.clone()
                    prev_ee_target_pos_w = ee_target_pos_w.clone()
                    reset_env_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
                    if reset_env_ids.numel() > 0:
                        fresh_ee_pos_w = _to_torch(ee_frame.data.target_pos_w)[:, 0, :]
                        fresh_ee_target_pos_w, _ = _target_ee_pos_w(unwrapped, step_count + 1)
                        fresh_joint_pos = _to_torch(robot.data.joint_pos)[:, arm_joint_ids]
                        initial_ee_pos_w[reset_env_ids] = fresh_ee_pos_w[reset_env_ids]
                        prev_ee_pos_w[reset_env_ids] = fresh_ee_pos_w[reset_env_ids]
                        prev_ee_target_pos_w[reset_env_ids] = fresh_ee_target_pos_w[reset_env_ids]
                        joint_pos_command[reset_env_ids] = fresh_joint_pos[reset_env_ids]

                step_count += 1

                if record_capture is not None and step_count % max(args_cli.record_every, 1) == 0:
                    frame = record_capture.render_rgb_array()
                    _save_rgb_frame(frame, frames_dir / f"frame_{frame_count:06d}.png")
                    frame_count += 1
        except KeyboardInterrupt:
            print("[INFO]: Interrupted by user.")
        finally:
            env.close()
            if args_cli.record_video and frames_dir is not None and video_path is not None:
                _run_ffmpeg(frames_dir, video_path, frame_count)


if __name__ == "__main__":
    main()
