# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn deformable prims into the scene.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/deformables.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="This script demonstrates how to spawn deformable prims into the scene.")
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
import random
import subprocess

import numpy as np
import torch
import tqdm
import warp as wp

import omni.replicator.core as rep

import isaaclab.sim as sim_utils
from isaaclab.utils import convert_dict_to_backend
from isaaclab.sensors.camera import Camera, CameraCfg
# from isaaclab.assets import DeformableObject, DeformableObjectCfg
from isaaclab_physx.assets import DeformableObject, DeformableObjectCfg


def define_origins(num_origins: int, spacing: float) -> list[list[float]]:
    """Defines the origins of the the scene."""
    # create tensor based on number of environments
    env_origins = torch.zeros(num_origins, 3)
    # create a grid of origins
    num_cols = np.floor(np.sqrt(num_origins))
    num_rows = np.ceil(num_origins / num_cols)
    xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="xy")
    env_origins[:, 0] = spacing * xx.flatten()[:num_origins] - spacing * (num_rows - 1) / 2
    env_origins[:, 1] = spacing * yy.flatten()[:num_origins] - spacing * (num_cols - 1) / 2
    env_origins[:, 2] = torch.rand(num_origins) + 1.0
    # return the origins
    return env_origins.tolist()


def define_sensor() -> Camera:
    """Defines the camera sensor to add to the scene."""
    # Setup camera sensor
    # In contrast to the ray-cast camera, we spawn the prim at these locations.
    # This means the camera sensor will be attached to these prims.
    sim_utils.create_prim("/World/OriginCamera", "Xform", translation=[0.0, 0.0, 0.0])
    camera_cfg = CameraCfg(
        prim_path="/World/OriginCamera/CameraSensor",
        update_period=1.0/args_cli.video_fps,
        height=600,
        width=800,
        data_types=["rgb",],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    )
    # Create camera
    camera = Camera(cfg=camera_cfg)

    return camera


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # spawn distant light
    cfg_light = sim_utils.DomeLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light.func("/World/light", cfg_light)

    # spawn a red cone
    cfg_sphere = sim_utils.MeshSphereCfg(
        radius=0.5,
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(),
        physics_material=sim_utils.DeformableBodyMaterialCfg(),
    )
    cfg_cuboid = sim_utils.MeshCuboidCfg(
        size=(0.2, 0.2, 0.2),
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(),
        physics_material=sim_utils.DeformableBodyMaterialCfg(),
    )
    cfg_cylinder = sim_utils.MeshCylinderCfg(
        radius=0.15,
        height=0.5,
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(),
        physics_material=sim_utils.DeformableBodyMaterialCfg(),
    )
    cfg_capsule = sim_utils.MeshCapsuleCfg(
        radius=0.35,
        height=0.5,
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(),
        physics_material=sim_utils.DeformableBodyMaterialCfg(),
    )
    cfg_cone = sim_utils.MeshConeCfg(
        radius=0.15,
        height=0.5,
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(),
        physics_material=sim_utils.DeformableBodyMaterialCfg(),
    )
    # create a dictionary of all the objects to be spawned
    objects_cfg = {
        "sphere": cfg_sphere,
        "cuboid": cfg_cuboid,
        "cylinder": cfg_cylinder,
        "capsule": cfg_capsule,
        "cone": cfg_cone,
    }

    # Create separate groups of deformable objects
    origins = define_origins(num_origins=6, spacing=0.6)
    print("[INFO]: Spawning objects...")
    # Iterate over all the origins and randomly spawn objects
    for idx, origin in tqdm.tqdm(enumerate(origins), total=len(origins)):
        # randomly select an object to spawn
        obj_name = random.choice(list(objects_cfg.keys()))
        obj_cfg = objects_cfg[obj_name]
        # randomize the young modulus (somewhere between a Silicone 30 and Silicone 70)
        obj_cfg.physics_material.youngs_modulus = random.uniform(0.7e6, 3.3e6)
        # randomize the poisson's ratio
        obj_cfg.physics_material.poissons_ratio = random.uniform(0.25, 0.45)
        # randomize the color
        obj_cfg.visual_material.diffuse_color = (random.random(), random.random(), random.random())
        # spawn the object
        obj_cfg.func(f"/World/Origin/Object{idx:02d}", obj_cfg, translation=origin)

    # create a view for all the deformables
    # note: since we manually spawned random deformable meshes above, we don't need to
    #   specify the spawn configuration for the deformable object
    cfg = DeformableObjectCfg(
        prim_path="/World/Origin/Object.*",
        spawn=None,
        init_state=DeformableObjectCfg.InitialStateCfg(),
    )
    deformable_object = DeformableObject(cfg=cfg)

    scene_entities = {"deformable_object": deformable_object}
    if args_cli.save:
        camera = define_sensor()
        scene_entities["camera"] = camera

    # return the scene information
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, DeformableObject], origins: torch.Tensor, output_dir: str = "outputs"):
    """Runs the simulation loop."""

    objects: DeformableObject = entities["deformable_object"]
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
        camera_positions = torch.tensor([[2.5, 2.5, 2.5]], device=sim.device)
        camera_targets = torch.tensor([[0.0, 0.0, 0.25]], device=sim.device)
        camera.set_world_poses_from_view(camera_positions, camera_targets)

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    assert sim_dt <= 1.0 / args_cli.video_fps, "Simulation timestep must be smaller than the inverse of the video FPS to save frames properly."
    num_steps = int(args_cli.total_time / sim_dt)
    sim_time = 0.0
    count = 0

    # Simulate physics
    for t in range(num_steps):
        # reset
        if sim_time > 4.0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset deformable object state
            for _, deform_body in enumerate(entities.values()):
                # root state
                nodal_state = wp.to_torch(deform_body.data.default_nodal_state_w).clone()
                deform_body.write_nodal_state_to_sim(nodal_state)
                # reset the internal state
                deform_body.reset()
            print("[INFO]: Resetting deformable object state...")
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        for deform_body in entities.values():
            deform_body.update(sim_dt)

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
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([4.0, 4.0, 3.0], [0.5, 0.5, 0.0])

    # Design scene by adding assets to it
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

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
