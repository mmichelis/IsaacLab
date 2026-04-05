# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates a raycaster sensor measuring height above cube platforms.

Two cube platforms are spawned: one deformable and one rigid. A rigid ball at each origin falls
onto its cube, and a raycaster attached to each ball casts rays downward against its corresponding
cube mesh. The sensor readings (height above cube) are printed periodically so you can compare
how the ball settles differently on the deformable vs. rigid platform.

.. note::
    The raycaster currently only supports static meshes. The warp mesh is built once at
    initialization, so deformation is not reflected in the ray-hit results.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/raycaster_deformable.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates a raycaster sensor above deformable cube platforms."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# demos should open Kit visualizer by default
parser.set_defaults(visualizer=["kit"])
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import warp as wp

# deformables supported in PhysX
from isaaclab_physx.assets import DeformableObject, DeformableObjectCfg
from isaaclab_physx.sim import DeformableBodyMaterialCfg, DeformableBodyPropertiesCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors.ray_caster import RayCaster, RayCasterCfg, patterns


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # Two origins: one for the deformable cube, one for the rigid cube
    origins = [[-0.5, 0.0, 0.25], [0.5, 0.0, 0.25]]
    for i, origin in enumerate(origins):
        sim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)

    # -- Origin 0: Deformable cube platform (resting on ground)
    deformable_cube_cfg = DeformableObjectCfg(
        prim_path="/World/Origin0/Cube",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.4, 0.4, 0.4),
            deformable_props=DeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
            physics_material=DeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=1e5),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.2)),
        debug_vis=True,
    )
    deformable_cube = DeformableObject(cfg=deformable_cube_cfg)

    # -- Origin 1: Rigid cube platform (MeshCuboidCfg so the raycaster can target its mesh)
    rigid_cube_cfg = RigidObjectCfg(
        prim_path="/World/Origin1/Cube",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.4, 0.4, 0.4),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.2)),
    )
    rigid_cube = RigidObject(cfg=rigid_cube_cfg)

    # Rigid balls above both cubes (raycaster parent prims)
    ball_cfg = RigidObjectCfg(
        prim_path="/World/Origin.*/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.1,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    )
    balls = RigidObject(cfg=ball_cfg)

    # Raycaster for ball above deformable cube -- casts against the deformable cube mesh
    ray_caster_deformable = RayCaster(
        cfg=RayCasterCfg(
            prim_path="/World/Origin0/Ball",
            mesh_prim_paths=["/World/Origin0/Cube"],
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(0.4, 0.4)),
            ray_alignment="world",
            debug_vis=not args_cli.headless,
        )
    )

    # Raycaster for ball above rigid cube -- casts against the rigid cube mesh
    ray_caster_rigid = RayCaster(
        cfg=RayCasterCfg(
            prim_path="/World/Origin1/Ball",
            mesh_prim_paths=["/World/Origin1/Cube"],
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(0.4, 0.4)),
            ray_alignment="world",
            debug_vis=not args_cli.headless,
        )
    )

    scene_entities = {
        "deformable_cube": deformable_cube,
        "rigid_cube": rigid_cube,
        "balls": balls,
        "ray_caster_deformable": ray_caster_deformable,
        "ray_caster_rigid": ray_caster_rigid,
    }
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict, origins: torch.Tensor):
    """Runs the simulation loop."""
    deformable_cube: DeformableObject = entities["deformable_cube"]
    rigid_cube: RigidObject = entities["rigid_cube"]
    balls: RigidObject = entities["balls"]
    rc_deformable: RayCaster = entities["ray_caster_deformable"]
    rc_rigid: RayCaster = entities["ray_caster_rigid"]

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Simulate physics
    while simulation_app.is_running():
        # reset
        if sim_time == 0.0 or sim_time > 3.0:
            sim_time = 0.0
            count = 0

            # reset deformable cube
            nodal_state = wp.to_torch(deformable_cube.data.default_nodal_state_w).clone()
            deformable_cube.write_nodal_state_to_sim_index(nodal_state)
            deformable_cube.reset()

            # reset rigid cube
            rigid_pose = wp.to_torch(rigid_cube.data.default_root_pose).clone()
            rigid_pose[:, :3] += origins[1:2]
            rigid_cube.write_root_pose_to_sim_index(root_pose=rigid_pose)
            rigid_vel = wp.to_torch(rigid_cube.data.default_root_vel).clone()
            rigid_cube.write_root_velocity_to_sim_index(root_velocity=rigid_vel)
            rigid_cube.reset()

            # reset rigid balls
            ball_pose = wp.to_torch(balls.data.default_root_pose).clone()
            ball_pose[:, :3] += origins
            balls.write_root_pose_to_sim_index(root_pose=ball_pose)
            ball_vel = wp.to_torch(balls.data.default_root_vel).clone()
            balls.write_root_velocity_to_sim_index(root_velocity=ball_vel)
            balls.reset()

            # reset raycasters
            rc_deformable.reset()
            rc_rigid.reset()

            print("----------------------------------------")
            print("[INFO]: Resetting scene...")

        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        deformable_cube.update(sim_dt)
        rigid_cube.update(sim_dt)
        balls.update(sim_dt)
        rc_deformable.update(dt=sim_dt, force_recompute=True)
        rc_rigid.update(dt=sim_dt, force_recompute=True)

        # print sensor readings periodically
        if count % int(0.1 / sim_dt) == 0:
            # height above deformable cube (ball 0)
            pos_def = rc_deformable.data.pos_w
            hits_def = rc_deformable.data.ray_hits_w
            height_def = (pos_def[:, 2].unsqueeze(1) - hits_def[..., 2]).mean(dim=1)
            # height above rigid cube (ball 1)
            pos_rig = rc_rigid.data.pos_w
            hits_rig = rc_rigid.data.ray_hits_w
            height_rig = (pos_rig[:, 2].unsqueeze(1) - hits_rig[..., 2]).mean(dim=1)

            print(f"  Time: {sim_time:.2f}s")
            print(f"  [Deformable] ball z={pos_def[0, 2]:.3f}  height above cube={height_def[0]:.3f}")
            print(f"  [Rigid]      ball z={pos_rig[0, 2]:.3f}  height above cube={height_rig[0]:.3f}")


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[2.0, 2.0, 2.0], target=[0.0, 0.0, 0.75])
    # Design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
