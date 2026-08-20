# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.sim.spawners.materials import NewtonMaterialCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_visualizers.newton import NewtonVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ManagerTermBaseCfg, SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg
from isaaclab_contrib.deformable.newton_manager_cfg import VBDSolverCfg

from isaaclab_tasks.core.peg_in_hole import mdp
from isaaclab_tasks.utils import PresetCfg

TABLE_SPAWN_CFG = sim_utils.CuboidCfg(
    size=(1.3, 0.9, 1.05),
    collision_props=sim_utils.CollisionPropertiesCfg(),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    visible=False,
)


_HOLE_PART_NAMES = ("hole_left", "hole_right", "hole_front", "hole_back", "hole_bottom")
_HOLE_DEFAULT_ANCHOR = (0.5, 0.0, 0.7)
_HOLE_BOTTOM_THICKNESS = 0.005
_HOLE_TOP_OFFSET = 0.005


@configclass
class HoleStructureCfg:
    """Dimensions of the hole structure [m]."""

    width: float = 0.05
    height: float = 0.05
    depth: float = 0.06
    opening_width: float = 0.03
    opening_height: float = 0.03


def _hole_parts(cfg: HoleStructureCfg) -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """Return hole-part sizes and target-relative offsets [m]."""
    values = (cfg.width, cfg.height, cfg.depth, cfg.opening_width, cfg.opening_height)
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError(f"Hole dimensions must be finite and positive, got {values}.")
    if cfg.opening_width >= cfg.width or cfg.opening_height >= cfg.height:
        raise ValueError("Hole opening dimensions must be smaller than the structure dimensions.")
    if cfg.depth <= _HOLE_BOTTOM_THICKNESS:
        raise ValueError(f"Hole depth must exceed the {_HOLE_BOTTOM_THICKNESS} m bottom thickness.")

    side_width = 0.5 * (cfg.width - cfg.opening_width)
    end_height = 0.5 * (cfg.height - cfg.opening_height)
    wall_depth = cfg.depth - _HOLE_BOTTOM_THICKNESS
    wall_z = _HOLE_TOP_OFFSET - 0.5 * wall_depth
    bottom_z = _HOLE_TOP_OFFSET - wall_depth - 0.5 * _HOLE_BOTTOM_THICKNESS
    side_x = 0.5 * (cfg.opening_width + side_width)
    end_y = 0.5 * (cfg.opening_height + end_height)
    return {
        "hole_left": ((side_width, cfg.height, wall_depth), (-side_x, 0.0, wall_z)),
        "hole_right": ((side_width, cfg.height, wall_depth), (side_x, 0.0, wall_z)),
        "hole_front": ((cfg.opening_width, end_height, wall_depth), (0.0, -end_y, wall_z)),
        "hole_back": ((cfg.opening_width, end_height, wall_depth), (0.0, end_y, wall_z)),
        "hole_bottom": ((cfg.width, cfg.height, _HOLE_BOTTOM_THICKNESS), (0.0, 0.0, bottom_z)),
    }


_DEFAULT_HOLE_PARTS = _hole_parts(HoleStructureCfg())


def _hole_part_cfg(name: str) -> RigidObjectCfg:
    """Create a fixed hole part."""
    size, offset = _DEFAULT_HOLE_PARTS[name]
    position = tuple(anchor + delta for anchor, delta in zip(_HOLE_DEFAULT_ANCHOR, offset, strict=True))
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name.title().replace('_', '')}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=position),
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.25, 0.65)),
        ),
    )


##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for a peg-in-hole scene with a robot and an object."""

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # task object
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.0, 0.04]),
        spawn=sim_utils.CuboidCfg(
            size=(0.02, 0.02, 0.07),
            physics_material=[
                UsdPhysicsRigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
                NewtonMaterialCfg(contact_stiffness=1.0e4, contact_damping=100.0),
            ],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        ),
    )

    # visual target
    target: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.0, 0.0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.02, 0.02, 0.07),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), opacity=0.35),
            visible=False,
        ),
    )

    hole_left = _hole_part_cfg("hole_left")
    hole_right = _hole_part_cfg("hole_right")
    hole_front = _hole_part_cfg("hole_front")
    hole_back = _hole_part_cfg("hole_back")
    hole_bottom = _hole_part_cfg("hole_bottom")

    # static table collider with its top surface at z = 0
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, -0.525]),
        spawn=TABLE_SPAWN_CFG,
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    pass


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        target_corners = ObsTerm(
            func=mdp.cuboid_corners_b,
            params={
                "asset_cfg": SceneEntityCfg("target"),
                "reference_cfg": SceneEntityCfg("robot"),
                "flatten": True,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class ProprioObsCfg(ObsGroup):
        """Observations for proprioception group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PerceptionObsCfg(ObsGroup):
        """Observations for perception group."""

        object_corners = ObsTerm(
            func=mdp.cuboid_corners_b,
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "reference_cfg": SceneEntityCfg("robot"),
                "flatten": True,
            },
        )

        hole_structure_corners = ObsTerm(
            func=mdp.hole_structure_corners_b,
            params={
                "part_cfgs": [SceneEntityCfg(name) for name in _HOLE_PART_NAMES],
                "reference_cfg": SceneEntityCfg("robot"),
                "flatten": True,
                "visualize": True,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_dim = 0
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    proprio: ProprioObsCfg = ProprioObsCfg()
    perception: PerceptionObsCfg = PerceptionObsCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    conditional_reset = EventTerm(
        func="isaaclab_tasks.core.lift.mdp.events:conditional_reset",
        mode="reset",
        params={
            "terms": {
                "reset_target_position": EventTerm(
                    func=mdp.reset_root_state_uniform,
                    mode="reset",
                    params={
                        "pose_range": {
                            "x": (-0.1, 0.1),
                            "y": (-0.25, 0.25),
                            "z": (0.05, 0.06),
                            "yaw": (-0.5, 0.5),
                        },
                        "velocity_range": {},
                        "asset_cfg": SceneEntityCfg("target"),
                    },
                ),
                "reset_robot_arm_joints": EventTerm(
                    func=mdp.reset_joints_within_limits_range,
                    mode="reset",
                    params={
                        "position_range": {
                            "panda_joint[1357]": (-0.2, 0.2),
                            "panda_joint[24]": (-0.2, 0.5),
                            "panda_joint6": (-0.3, 0.2),
                        },
                        "velocity_range": {"panda_joint.*": (0.0, 0.0)},
                        "use_default_offset": True,
                        "asset_cfg": SceneEntityCfg("robot"),
                    },
                ),
                "reset_robot_gripper_joints": EventTerm(
                    func=mdp.reset_joints_shared_offset,
                    mode="reset",
                    params={
                        "position_range": (-0.02, 0.0),
                        "asset_cfg": SceneEntityCfg("robot", joint_names="panda_finger_joint.*"),
                    },
                ),
                "reset_object_position": EventTerm(
                    func=mdp.reset_root_state_uniform,
                    mode="reset",
                    params={
                        "pose_range": {"x": (-0.2, 0.2), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
                        "velocity_range": {},
                        "asset_cfg": SceneEntityCfg("object", body_names="Object"),
                    },
                ),
                "reset_hole_from_target": EventTerm(
                    func=mdp.reset_hole_from_target,
                    mode="reset",
                    params={
                        "part_offsets": {name: offset for name, (_, offset) in _DEFAULT_HOLE_PARTS.items()},
                        "depth_range": (0.0, 0.0),
                        "target_cfg": SceneEntityCfg("target"),
                    },
                ),
            },
            "buffer_size_per_group": 32768,
            "oversample_factor": 2.0,
            "diversity_feature": mdp.GraspTravelDistanceCfg(
                asset_name="robot",
                body_names="panda_hand",
                object_name="object",
                target_name="target",
                log_scale=True,
            ),
            "valid_criteria": {
                "object_robot_clearance": mdp.MeshClearanceCfg(
                    asset_name="robot",
                    body_names=".*",
                    object_name="object",
                    num_object_points=32,
                    min_clearance=0.001,
                ),
                **{
                    f"robot_{name}_clearance": mdp.MeshClearanceCfg(
                        asset_name="robot",
                        body_names=".*",
                        object_name=name,
                        num_object_points=32,
                        min_clearance=0.001,
                    )
                    for name in _HOLE_PART_NAMES
                },
                "object_hole_clearance": ManagerTermBaseCfg(
                    func=mdp.rigid_object_box_clearance,
                    params={
                        "object_name": "object",
                        "obstacle_names": list(_HOLE_PART_NAMES),
                        "min_clearance": 0.001,
                    },
                ),
                "hole_table_clearance": ManagerTermBaseCfg(
                    func=mdp.rigid_objects_above_plane,
                    params={
                        "object_names": list(_HOLE_PART_NAMES),
                        "height": 0.0,
                        "min_clearance": 0.001,
                    },
                ),
                "robot_table_clearance": mdp.SlabClearanceCfg(
                    asset_name="robot",
                    body_names=["panda_link[1-7]", "panda_hand", ".*finger"],
                    object_name="object",
                    obstacle_slabs=[((-0.15, 1.15), (-0.45, 0.45), 0.0)],
                    num_object_points=32,
                    min_clearance=0.001,
                ),
            },
            "success_monitor": mdp.SuccessMonitorCfg(target_success_rate=0.5),
            "uniform_eval_interval_steps": 2400,
            "uniform_eval_num_episodes": 1024,
        },
    )

    reset_target_depth = EventTerm(
        func=mdp.reset_target_depth,
        mode="reset",
        params={
            "depth_range": (0.05, 0.07),
            "asset_cfg": SceneEntityCfg("target"),
        },
    )

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": [0.5, 1.0],
            "dynamic_friction_range": [0.5, 1.0],
            "restitution_range": [0.0, 0.0],
            "num_buckets": 250,
        },
    )

    joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": [0.8, 1.2],
            "damping_distribution_params": [0.8, 1.2],
            "operation": "scale",
        },
    )

    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "static_friction_range": [0.5, 1.0],
            "dynamic_friction_range": [0.5, 1.0],
            "restitution_range": [0.0, 0.0],
            "num_buckets": 250,
        },
    )

    object_physics_inertia = EventTerm(
        func=mdp.randomize_rigid_body_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "inertia_distribution_params": [0.9, 1.1],
            "operation": "scale",
            "diagonal_only": True,
        },
    )

    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": [0.5, 2.0],
            "operation": "scale",
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.2, "object_cfg": SceneEntityCfg("object", body_names="Object")},
        weight=1.0,
    )

    goal_distance = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.3,
            "object_cfg": SceneEntityCfg("object", body_names="Object"),
            "target_cfg": SceneEntityCfg("target"),
        },
        weight=1.0,
    )

    success = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.05,
            "success_threshold": 0.05,
            "object_cfg": SceneEntityCfg("object", body_names="Object"),
            "target_cfg": SceneEntityCfg("target"),
        },
        weight=1.0,
    )

    success_bonus = RewTerm(
        func=mdp.object_target_point_cloud_reached,
        params={
            "success_threshold": 0.05,
            "object_cfg": SceneEntityCfg("object", body_names="Object"),
            "target_cfg": SceneEntityCfg("target"),
            "success_vis_asset_name": "table",
            "success_visualizer_cfg": VisualizationMarkersCfg(
                prim_path="/Visuals/SuccessMarkers",
                markers={
                    "failure": TABLE_SPAWN_CFG.replace(
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.5, 0.5)), visible=True
                    ),
                    "success": TABLE_SPAWN_CFG.replace(
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.8, 0.5)), visible=True
                    ),
                },
            ),
        },
        weight=1.0,
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-3)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_out_of_bounds = DoneTerm(
        func=mdp.object_outside_bounds,
        params={
            "x_bounds": (0.0, 1.0),
            "y_bounds": (-0.5, 0.5),
            "z_bounds": (-0.02, 1.0),
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    joint_vel_out_of_limit = DoneTerm(
        func=mdp.joint_vel_out_of_sim_limit,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    adr = CurrTerm(
        func=mdp.DifficultyScheduler,
        params={
            "init_difficulty": 0,
            "min_difficulty": 0,
            "max_difficulty": 10,
        },
    )

    target_depth_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "events.reset_target_depth.params.depth_range",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {
                "initial_value": (0.05, 0.07),
                "final_value": (-0.015, -0.015),
                "difficulty_term_str": "adr",
            },
        },
    )

    success_threshold_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.success.params.success_threshold",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {
                "initial_value": 0.05,
                "final_value": 0.005,
                "difficulty_term_str": "adr",
            },
        },
    )

    success_bonus_threshold_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.success_bonus.params.success_threshold",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {
                "initial_value": 0.05,
                "final_value": 0.005,
                "difficulty_term_str": "adr",
            },
        },
    )

    action_rate = CurrTerm(
        func=mdp.reward_weight_linear,
        params={
            "term_name": "action_rate",
            "start_weight": -1e-3,
            "end_weight": -1e-1,
            "start_step": 50000,
            "end_step": 60000,
        },
    )

    episode_length = CurrTerm(
        func=mdp.modify_env_param,
        params={
            "address": "cfg.episode_length_s",
            "modify_fn": mdp.linear_interpolate,
            "modify_params": {
                "start_value": 2.0,
                "end_value": 8.0,
                "start_step": 0,
                "end_step": 10000,
            },
        },
    )


##
# Environment configuration
##


@configclass
class PegInHolePhysicsCfg(PresetCfg):
    isaacsim_physx: PhysxCfg = PhysxCfg(
        bounce_threshold_velocity=0.01,
        gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
        gpu_total_aggregate_pairs_capacity=16 * 1024,
        friction_correlation_distance=0.00625,
    )

    physx: PhysxAutoCfg = PhysxAutoCfg(isaacsim_physx=isaacsim_physx)

    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            cone="pyramidal",
            integrator="implicitfast",
            use_mujoco_contacts=False,
            nconmax=200,
        ),
        num_substeps=2,
        collision_decimation=1,
    )

    newton_mjwarp_vbd_proxy: NewtonCfg = NewtonCfg(
        solver_cfg=CouplerProxyCfg(
            entries=[
                CouplerEntryCfg(
                    name="rigid",
                    solver_cfg=MJWarpSolverCfg(
                        cone="elliptic", ls_iterations=20, integrator="implicitfast", nconmax=200
                    ),
                    bodies=[
                        r"/World/envs/env_.*/Robot",
                        r"/World/envs/env_.*/Target",
                        r"/World/envs/env_.*/Hole(Left|Right|Front|Back|Bottom)",
                        r"/World/envs/env_.*/Table",
                    ],
                ),
                CouplerEntryCfg(
                    name="object",
                    solver_cfg=VBDSolverCfg(iterations=10),
                    bodies=[r"/World/envs/env_.*/Object"],
                    include_static_shapes=True,
                ),
            ],
            proxies=[
                CouplerProxyMappingCfg(
                    source="rigid",
                    destination="object",
                    bodies=[
                        r"/World/envs/env_.*/Robot/panda_hand",
                        r"/World/envs/env_.*/Robot/panda_(left|right)finger",
                        r"/World/envs/env_.*/Hole(Left|Right|Front|Back|Bottom)",
                        r"/World/envs/env_.*/Table",
                    ],
                )
            ],
            iterations=1,
        ),
        num_substeps=2,
    )

    default = newton_mjwarp


@configclass
class PegInHoleEnvCfg(ManagerBasedRLEnvCfg):
    """Initial peg-in-hole configuration copied from the lift task."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=8192, env_spacing=2.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    hole_structure: HoleStructureCfg = HoleStructureCfg()

    def __post_init__(self):
        """Post initialization."""
        self._sync_hole_structure()

        # general settings
        self.decimation = 4
        self.episode_length_s = 2.0
        # simulation settings
        self.sim.dt = 1.0 / 120
        self.sim.render_interval = self.decimation

        self.sim.physics = PegInHolePhysicsCfg()

        # Camera pose (lowered eye framing the tabletop workspace).
        eye, lookat = (2.75, -2.0, 1.0), (0.5, 0.0, -0.5)
        self.viewer.eye = eye
        self.viewer.lookat = lookat
        self.sim.visualizer_cfgs = [NewtonVisualizerCfg(eye=eye, lookat=lookat, window_width=1920, window_height=1080)]

    def validate_config(self):
        """Synchronize derived hole geometry before environment creation."""
        self._sync_hole_structure()

    def _sync_hole_structure(self):
        hole_parts = _hole_parts(self.hole_structure)
        for name, (size, offset) in hole_parts.items():
            part_cfg = getattr(self.scene, name)
            part_cfg.spawn.size = size
            part_cfg.init_state.pos = tuple(
                anchor + delta for anchor, delta in zip(_HOLE_DEFAULT_ANCHOR, offset, strict=True)
            )
        reset_terms = self.events.conditional_reset.params["terms"]
        reset_terms["reset_hole_from_target"].params["part_offsets"] = {
            name: offset for name, (_, offset) in hole_parts.items()
        }

    def play_mode(self):
        super().play_mode()
        self.episode_length_s = 8.0
        reset_params = self.events.conditional_reset.params
        reset_params["buffer_size_per_group"] = 64
        reset_params["oversample_factor"] = 1.0
        reset_params["diversity_feature"] = None
        reset_params["uniform_eval_interval_steps"] = None
        if self.curriculum is not None:
            self.curriculum.adr.params["init_difficulty"] = self.curriculum.adr.params["max_difficulty"]
            self.curriculum.adr.params["promotion_only"] = True
            self.curriculum.episode_length = None
