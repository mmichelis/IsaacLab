# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates a quadruped robot standing on a deformable sphere
and receiving random joint position targets.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/quadruped_on_deformable.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates a quadruped on a deformable sphere with random joint targets."
)
parser.add_argument(
    "--total_time",
    type=float,
    default=4.0,
    help="Total simulation time in seconds.",
)
parser.add_argument(
    "--dt",
    type=float,
    default=0.01,
    help="Simulation timestep.",
)
parser.add_argument(
    "--video_fps",
    type=int,
    default=60,
    help="FPS for the output video if --save is enabled.",
)
parser.add_argument(
    "--save",
    action="store_true",
    default=False,
    help="Save the data from camera.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# demos should open Kit visualizer by default
parser.set_defaults(visualizer=["kit"])
# parse the arguments
args_cli = parser.parse_args()
if args_cli.save:
    args_cli.enable_cameras = True
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import subprocess
import torch
import warp as wp

# deformables supported in PhysX
from isaaclab_physx.assets import DeformableObject, DeformableObjectCfg
from isaaclab_physx.sim import DeformableBodyMaterialCfg, DeformableBodyPropertiesCfg, SurfaceDeformableBodyMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.utils import convert_dict_to_backend

# import only used for rendering frames
if args_cli.save:
    import omni.replicator.core as rep

##
# Pre-defined configs
##
from isaaclab_assets.robots.anymal import ANYMAL_D_CFG  # isort:skip


def define_sensor() -> Camera:
    """Defines the camera sensor to add to the scene for rendering frames."""
    # Setup camera sensor
    sim_utils.create_prim("/World/CameraOrigin", "Xform", translation=[0.0, 0.0, 0.0])
    camera_cfg = CameraCfg(
        prim_path="/World/CameraOrigin/CameraSensor",
        update_period=1.0 / args_cli.video_fps,
        height=800,
        width=800,
        data_types=[
            "rgb",
        ],
        spawn=sim_utils.PinholeCameraCfg(),
    )
    # Create camera
    camera = Camera(cfg=camera_cfg)

    return camera


def design_scene() -> tuple[dict, list[float]]:
    """Designs the scene with a deformable sphere and a quadruped on top."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    scene_entities = {}
    # -- Deformable sphere beneath the robot
    origin = [0.0, 0.0, 0.76]
    cfg_sphere = sim_utils.UsdFileCfg(
        usd_path="/home/mmichelis/Documents/IsaacLab/scripts/demos/icosphere_3.usda",
        scale=[0.75, 0.75, 0.75],
        deformable_props=DeformableBodyPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 0.9)),
        physics_material=SurfaceDeformableBodyMaterialCfg(
            density=10.0,
            youngs_modulus=0.5e4,
            poissons_ratio=0.3,
            surface_thickness=0.1,
            surface_bend_stiffness=5e4,
            surface_shear_stiffness=5e4,
            surface_stretch_stiffness=5e4,
            static_friction=0.75,
            dynamic_friction=0.75,
        ),
    )
    cfg_sphere.func("/World/DeformableSphere", cfg_sphere, translation=origin)

    # Create a deformable object view
    deformable_cfg = DeformableObjectCfg(
        prim_path="/World/DeformableSphere",
        spawn=None,
    )
    deformable_sphere = DeformableObject(cfg=deformable_cfg)
    scene_entities["deformable_sphere"] = deformable_sphere

    # -- Quadruped robot placed above the sphere
    robot = Articulation(ANYMAL_D_CFG.replace(prim_path="/World/Robot"))
    scene_entities["robot"] = robot

    # -- Create camera if saving frames
    if args_cli.save:
        camera = define_sensor()
        scene_entities["camera"] = camera

    return scene_entities, origin


def run_simulator(sim: sim_utils.SimulationContext, entities: dict, sphere_origin: torch.Tensor, output_dir: str = "outputs"):
    """Runs the simulation loop."""
    robot: Articulation = entities["robot"]
    deformable_sphere: DeformableObject = entities["deformable_sphere"]

    # Write camera outputs
    if args_cli.save:
        camera: Camera = entities["camera"]

        # Create replicator writer
        rep_writer = rep.BasicWriter(
            output_dir=output_dir,
            frame_padding=0,
            rgb=True,
        )
        # Camera positions, targets, orientations
        camera_positions = torch.tensor([[2.25, 2.25, 2.0]], device=sim.device)
        camera_targets = torch.tensor([[0.0, 0.0, 0.75]], device=sim.device)
        camera.set_world_poses_from_view(camera_positions, camera_targets)


    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    assert sim_dt <= 1.0 / args_cli.video_fps, (
        "Simulation timestep must be smaller than the inverse of the video FPS to save frames properly."
    )
    num_steps = int(args_cli.total_time / sim_dt)
    sim_time = 0.0
    count = 0
    
    dof = robot.num_joints + deformable_sphere.max_sim_vertices_per_body * 3
    robot_mass = wp.to_torch(robot.data.body_mass).sum().item()
    sphere_radius = 0.75
    sphere_mass = 10.0 * 4.0/3.0 * torch.pi * (sphere_radius**3 - (sphere_radius - 0.05)**3)

    print(
        f"[INFO]: Starting simulation for {args_cli.total_time}s, dt={sim_dt}, steps={num_steps}, "
        f"DOFs={dof}, robot mass={robot_mass:.2f} kg, sphere mass~={sphere_mass:.2f} kg"
    )

    # Simulate physics
    for t in range(num_steps):
        # reset
        if sim_time == 0.0 or sim_time > 3.0:
            sim_time = 0.0
            count = 0
            # reset robot
            root_pose = wp.to_torch(robot.data.default_root_pose).clone()
            root_pose[:, :3] += sphere_origin
            # place robot above the sphere
            root_pose[:, 2] += 0.6
            robot.write_root_pose_to_sim_index(root_pose=root_pose)
            root_vel = wp.to_torch(robot.data.default_root_vel).clone()
            robot.write_root_velocity_to_sim_index(root_velocity=root_vel)
            joint_pos = wp.to_torch(robot.data.default_joint_pos).clone()
            joint_vel = wp.to_torch(robot.data.default_joint_vel).clone()
            robot.write_joint_position_to_sim_index(position=joint_pos)
            robot.write_joint_velocity_to_sim_index(velocity=joint_vel)
            robot.reset()
            # reset deformable sphere
            nodal_state = wp.to_torch(deformable_sphere.data.default_nodal_state_w).clone()
            deformable_sphere.write_nodal_state_to_sim_index(nodal_state)
            deformable_sphere.reset()
            print("[INFO]: Resetting scene...")

        # apply random joint position targets
        joint_pos_target = (
            wp.to_torch(robot.data.default_joint_pos) + torch.randn_like(wp.to_torch(robot.data.joint_pos)) * 0.5
        )
        robot.set_joint_position_target_index(target=joint_pos_target)
        robot.write_data_to_sim()

        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        for entity in entities.values():
            entity.update(sim_dt)

        # Extract camera data
        if args_cli.save:
            if camera.data.output["rgb"] is not None:
                cam_data = convert_dict_to_backend(camera.data.output, backend="numpy")
                rep_writer.write(
                    {
                        "annotators": {"rgb": {"render_product": {"data": cam_data["rgb"][0]}}},
                        "trigger_outputs": {"on_time": camera.frame[0]},
                    }
                )



def main():
    """Main function."""
    # Initialize the simulation context
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=args_cli.dt))
    # Set main camera
    sim.set_camera_view(eye=[2.25, 2.25, 2.0], target=[0.0, 0.0, 0.75])
    # design scene
    scene_entities, sphere_origin = design_scene()
    sphere_origin = torch.tensor([sphere_origin], device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    camera_output = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "camera")
    run_simulator(sim, scene_entities, sphere_origin, camera_output)
    print("[INFO]: Simulation complete...")

    # Store video if saving frames
    if args_cli.save:
        video_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "output.mp4")
        fps = args_cli.video_fps
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(fps),
                "-i",
                os.path.join(camera_output, "rgb_%d_0.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                video_path,
            ],
            check=True,
        )
        # Also generate gif for quick preview
        gif_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "output.gif")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                video_path,
                "-vf",
                "fps=15,scale=320:-1:flags=lanczos",
                gif_path,
            ],
            check=True,
        )
        print(f"[INFO]: Video saved to {video_path}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()