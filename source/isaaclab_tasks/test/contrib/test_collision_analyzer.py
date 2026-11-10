# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Live-node collision queries share the rigid SDF path, without stepping physics."""

from isaaclab_newton.physics import NewtonCfg, VBDSolverCfg

from isaaclab.sim import SimulationCfg
from isaaclab.test.utils import launch_test_simulation

SIM_CFG = SimulationCfg(device="cpu", physics=NewtonCfg(solver_cfg=VBDSolverCfg(iterations=1), use_cuda_graph=False))
launch_test_simulation(SIM_CFG, device="cpu")

from types import SimpleNamespace

import pytest
import torch
from isaaclab_newton.sim.spawners.materials import NewtonDeformableBodyMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import build_simulation_context
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.nist.utils.collision_analyzer import CollisionAnalyzer
from isaaclab_tasks.contrib.nist.utils.collision_analyzer_cfg import CollisionAnalyzerCfg

pytestmark = pytest.mark.kitless


@configclass
class QuerySceneCfg(InteractiveSceneCfg):
    soft = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Soft",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.1, 0.1, 0.1),
            volume_deformable_props=[],
            physics_material=NewtonDeformableBodyMaterialCfg(density=500.0, k_mu=1e4, k_lambda=1e4),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.4, -0.2, 1.0)),
    )
    rigid = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Rigid",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            mass_props=sim_utils.MassCfg(mass=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, -0.2, 1.0)),
    )
    obstacle = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle",
        spawn=sim_utils.CuboidCfg(size=(0.6, 0.6, 0.4), collision_props=sim_utils.UsdPhysicsCollisionCfg()),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.4, -0.2, 0.3), rot=(0.0, 0.0, 0.3826834, 0.9238795)),
    )


def test_live_nodes_subset_and_rigid_queries():
    with build_simulation_context(sim_cfg=SIM_CFG, device="cpu") as sim:
        scene = InteractiveScene(QuerySceneCfg(num_envs=3, env_spacing=2.0))
        sim.register_interactive_scene(scene)
        sim.reset()
        env = SimpleNamespace(scene=scene, device="cpu", num_envs=3)
        cfg = CollisionAnalyzerCfg(
            asset_cfg=SceneEntityCfg("soft"), obstacle_cfgs=[SceneEntityCfg("obstacle")], min_dist=0.01
        )
        checker = CollisionAnalyzer(cfg, env)
        ids = torch.tensor([2, 0])
        assert checker(env, ids).tolist() == [True, True]
        nodes = scene["soft"].data.nodal_pos_w.torch.clone()
        # Only one live node enters the obstacle; the rest shape stays above it.
        nodes[2, -1] = scene.env_origins[2] + torch.tensor([0.4, -0.2, 0.3])
        scene["soft"].write_nodal_pos_to_sim_index(nodes, env_ids=torch.arange(3))
        assert checker(env, ids).tolist() == [False, True]
        nodes[2, -1, 2] = 1.0
        nodes[0, -1] = scene.env_origins[0] + torch.tensor([0.4, -0.2, 0.3])
        scene["soft"].write_nodal_pos_to_sim_index(nodes, env_ids=torch.arange(3))
        assert checker(env, ids).tolist() == [True, False]

        cfg.asset_cfg = SceneEntityCfg("rigid")
        rigid_checker = CollisionAnalyzer(cfg, env)
        assert rigid_checker(env, ids).tolist() == [True, True]
        pose = scene["rigid"].data.root_pose_w.torch[0:1].clone()
        pose[0, :3] = scene.env_origins[0] + torch.tensor([0.4, -0.2, 0.3])
        scene["rigid"].write_root_pose_to_sim_index(root_pose=pose, env_ids=torch.tensor([0]))
        assert rigid_checker(env, ids).tolist() == [True, False]
