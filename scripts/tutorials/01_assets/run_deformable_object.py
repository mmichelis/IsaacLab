# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to work with the deformable object and interact with it.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/01_assets/run_deformable_object.py

"""

"""Launch Isaac Sim Simulator first."""

import os
import argparse
import subprocess

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on interacting with a deformable object.")
parser.add_argument(
    "--total_time",
    type=float,
    default=4.0,
    help="Total simulation time in seconds.",
)
parser.add_argument(
    "--dt",
    type=float,
    default=1.0/60,
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
# parse the arguments
args_cli = parser.parse_args()
if args_cli.save:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import warp as wp
from isaaclab_physx.assets import DeformableObject, DeformableObjectCfg

import matplotlib.pyplot as plt
import numpy as np

import omni.replicator.core as rep

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.sim import SimulationContext
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.utils import convert_dict_to_backend

# deformables supported in PhysX
from isaaclab_physx.assets import DeformableObject, DeformableObjectCfg
from isaaclab_physx.sim import DeformableBodyMaterialCfg, SurfaceDeformableBodyMaterialCfg


def define_sensor() -> Camera:
    """Defines the camera sensor to add to the scene."""
    sim_utils.create_prim("/World/OriginCamera", "Xform", translation=[0.0, 0.0, 0.0])
    camera_cfg = CameraCfg(
        prim_path="/World/OriginCamera/CameraSensor",
        update_period=1.0/args_cli.video_fps,
        height=800,
        width=800,
        data_types=["rgb",],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    )
    camera = Camera(cfg=camera_cfg)

    return camera


def design_scene():
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # Create a dictionary for the scene entities
    scene_entities = {}

    # Create separate groups called "Origin0", "Origin1", ...
    # Each group will have a robot in it
    origins = [[0.25, 0.25, 0.0], [-0.25, 0.25, 0.0], [0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]]
    for i, origin in enumerate(origins):
        sim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)

    # 3D Deformable Object
    cfg = DeformableObjectCfg(
        prim_path="/World/Origin.*/Cube",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.2, 0.2, 0.2),
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
            physics_material=DeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=1e5),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        debug_vis=True,
    )
    
    cube_object = DeformableObject(cfg=cfg)
    scene_entities["cube_object"] = cube_object

    # 2D Cloth Object
    sim_utils.create_prim(f"/World/OriginCloth", "Xform", translation=[0,0,1.5])
    cfg = DeformableObjectCfg(
        prim_path="/World/OriginCloth/Cloth",
        spawn=sim_utils.MeshSquareCfg(
            size=1.5,
            resolution=(21, 21),
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.01, contact_offset=0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.5, 0.1)),
            physics_material=SurfaceDeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=1e5),
        ),
    )
    cloth_object = DeformableObject(cfg=cfg)
    scene_entities["cloth_object"] = cloth_object

    # Sensors
    if args_cli.save:
        camera = define_sensor()
        scene_entities["camera"] = camera

    # return the scene information
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict, origins: torch.Tensor, output_dir: str):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    cube_object: DeformableObject = entities["cube_object"]
    cloth_object: DeformableObject = entities["cloth_object"]

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
        camera_positions = torch.tensor([[2., 2., 2.]], device=sim.device)
        camera_targets = torch.tensor([[0.0, 0.0, 0.75]], device=sim.device)
        camera.set_world_poses_from_view(camera_positions, camera_targets)


    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    assert sim_dt <= 1.0 / args_cli.video_fps, "Simulation timestep must be smaller than the inverse of the video FPS to save frames properly."
    num_steps = int(args_cli.total_time / sim_dt)
    sim_time = 0.0
    count = 0

    # Nodal kinematic targets of the deformable bodies
    nodal_kinematic_target = wp.to_torch(cube_object.data.nodal_kinematic_target).clone()

    # Simulate physics
    com_traj = []
    for t in range(num_steps):
        # reset at start and after N seconds
        if sim_time == 0.0 or sim_time > 3.0:
            # reset counters
            sim_time = 0.0
            count = 0

            # reset the nodal state of the object
            nodal_state = wp.to_torch(cube_object.data.default_nodal_state_w).clone()
            # apply random pose to the object
            pos_w = torch.rand(cube_object.num_instances, 3, device=sim.device) * 0.1 + origins
            quat_w = math_utils.random_orientation(cube_object.num_instances, device=sim.device)
            nodal_state[..., :3] = cube_object.transform_nodal_pos(nodal_state[..., :3], pos_w, quat_w)

            # write nodal state to simulation
            cube_object.write_nodal_state_to_sim_index(nodal_state)

            # Write the nodal state to the kinematic target and free all vertices
            nodal_kinematic_target[..., :3] = nodal_state[..., :3]
            nodal_kinematic_target[..., 3] = 1.0
            cube_object.write_nodal_kinematic_target_to_sim_index(nodal_kinematic_target)

            # reset buffers
            cube_object.reset()

            # reset the cloth object as well
            nodal_state = wp.to_torch(cloth_object.data.default_nodal_state_w).clone()
            cloth_object.write_nodal_state_to_sim(nodal_state)
            cloth_object.reset()

            print("----------------------------------------")
            print("[INFO]: Resetting object state...")

        # update the kinematic target for cubes at index 0 and 3
        kinematic_cubes = [0, 3]
        # we slightly move the cube in the z-direction by picking the vertex at index 0
        nodal_kinematic_target[kinematic_cubes, 0, 2] += 0.2 * sim_dt
        # set vertex at index 0 to be kinematically constrained
        # 0: constrained, 1: free
        nodal_kinematic_target[kinematic_cubes, 0, 3] = 0.0
        # write kinematic target to simulation
        cube_object.write_nodal_kinematic_target_to_sim_index(nodal_kinematic_target)

        # write internal data to simulation
        cube_object.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        for entity in entities.values():
            entity.update(sim_dt)

        com_traj.append(wp.to_torch(cube_object.data.nodal_pos_w).mean(1).cpu().numpy())
        # print the root position
        if t % args_cli.video_fps == 0:
            print(f"Time {t*sim_dt:.2f}s: \tRoot position (in world): {wp.to_torch(cube_object.data.root_pos_w)[:, :3]}")
            print(f"Cube 0 COM: {wp.to_torch(cube_object.data.nodal_pos_w)[0].mean(0)}")
            print(f"Cube 1 COM: {wp.to_torch(cube_object.data.nodal_pos_w)[1].mean(0)}")
            print(f"Cube 2 COM: {wp.to_torch(cube_object.data.nodal_pos_w)[2].mean(0)}")
            print(f"Cube 3 COM: {wp.to_torch(cube_object.data.nodal_pos_w)[3].mean(0)}")
            print(f"Cloth COM: {wp.to_torch(cloth_object.data.nodal_pos_w)[0].mean(0)}")

            trajectories = np.stack(com_traj, axis=1)
            time_axis = np.arange(trajectories.shape[1]) * sim_dt
            fig, ax = plt.subplots(figsize=(4, 3))
            for i in range(4):
                ax.plot(time_axis, trajectories[i, :, 2], label=f"Cube {i}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Z Position (m)")
            ax.legend()
            ax.grid()
            fig.savefig(os.path.join(os.path.dirname(output_dir), f"com_trajectory.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)

        # Extract camera data
        if args_cli.save:
            if camera.data.output["rgb"] is not None:
                cam_data = convert_dict_to_backend(camera.data.output, backend="numpy")
                rep_writer.write({
                    "annotators": {"rgb": {"render_product": {"data": cam_data["rgb"][0]}}},
                    "trigger_outputs": {"on_time": camera.frame[0]}
                })


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=args_cli.dt, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[3.0, 0.0, 1.0], target=[0.0, 0.0, 0.5])
    # Design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    camera_output = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "camera")
    run_simulator(sim, scene_entities, scene_origins, camera_output)
    # Store video if saving frames
    if args_cli.save:
        video_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "output.mp4")
        fps = args_cli.video_fps
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", os.path.join(camera_output, "rgb_%d_0.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            video_path,
        ], check=True)
        # Also generate gif for quick preview
        gif_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "output.gif")
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", video_path,
            "-vf", "fps=15,scale=320:-1:flags=lanczos",
            gif_path,
        ], check=True)
        print(f"[INFO]: Video saved to {video_path}")

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
