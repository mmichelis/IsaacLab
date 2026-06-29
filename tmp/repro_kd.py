import argparse, contextlib, sys
import gymnasium as gym, numpy as np, torch
import isaaclab_tasks  # noqa: F401
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401
from isaaclab_tasks.utils import (
    add_launcher_args, fold_preset_tokens, launch_simulation, resolve_task_config, setup_preset_cli,
)

p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("--arm_damping", type=float, default=None)  # override mjwarp damping
p.add_argument("--steps", type=int, default=60)
add_launcher_args(p)
args, hydra = setup_preset_cli(p)
sys.argv = [sys.argv[0]] + fold_preset_tokens(hydra)


def main():
    cfg, _ = resolve_task_config(args.task, "")
    with launch_simulation(cfg, args):
        cfg.scene.num_envs = 1
        if args.arm_damping is not None:
            for a in ("panda_shoulder", "panda_forearm"):
                cfg.scene.robot.actuators[a].damping = args.arm_damping
        env = gym.make(args.task, cfg=cfg)
        dev = env.unwrapped.device
        ee = env.unwrapped.scene["ee_frame"]
        # identical seeded action sequence -> identical joint-position targets in both backends
        g = torch.Generator().manual_seed(0)
        acts = (2 * torch.rand((args.steps, 1, env.action_space.shape[-1]), generator=g) - 1).to(dev)
        env.reset()
        traj = []
        with torch.inference_mode():
            for t in range(args.steps):
                env.step(acts[t])
                traj.append(ee.data.target_pos_source[0, 0, :].cpu().numpy().copy())
        ee_pos = np.stack(traj)
        rng = np.linalg.norm(ee_pos.max(0) - ee_pos.min(0))
        smooth = np.linalg.norm(np.diff(ee_pos, axis=0), axis=1).mean()
        print(f"RESULT  task={args.task}  kd={args.arm_damping}  |  EE_range_L2={rng:.4f} m  mean_step={smooth:.4f} m")
        env.close()


main()
