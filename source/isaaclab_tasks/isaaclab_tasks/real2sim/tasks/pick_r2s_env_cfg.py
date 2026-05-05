import sys
import numpy as np
from pathlib import Path
from pxr import Usd, UsdPhysics
from typing import Dict, Optional, List
from dataclasses import MISSING, field
import isaaclab.sim as sim_utils
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sensors import CameraCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.utils import configclass
from isaaclab.sensors import FrameTransformerCfg, CameraCfg, TiledCameraCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from . import mdp
from isaaclab_tasks.real2sim.assets.franka import *
from isaaclab_tasks.real2sim.assets.ur5e import UR5E_ROBOTIQ85_CFG
from isaaclab_tasks.real2sim.assets.sawyer import SAWYER_ROBOTIQ85_CFG
from isaaclab_tasks.real2sim.assets.xarm7 import XARM7_CFG
from isaaclab.envs.mdp.commands.commands_cfg import ObjectRelativePoseCommandCfg

from isaaclab_tasks.real2sim.utils.task_config_loader import load_task_config, TaskConfig, ObjectConfig, ArticulationObjectConfig, print_task_config

ROBOT_CONFIGS = {
    "franka_fr3_default": FRANKA_FR3_CFG,
    "franka_fr3_ourgripper": FRANKA_FR3_OURGRIPPER_CFG,
    "franka_panda_ourgripper": FRANKA_PANDA_OURGRIPPER_CFG,
    "ur5e_robotiq85": UR5E_ROBOTIQ85_CFG,
    "sawyer_robotiq85": SAWYER_ROBOTIQ85_CFG,
    "xarm7_ufactory": XARM7_CFG,
}

USE_BACKGROUND_REPLACE = True

_CURRENT_TASK_CONFIG: Optional[TaskConfig] = None


def _get_project_root() -> Path:
    """Find the project root that contains the models directory."""
    root = Path(__file__).resolve()
    while root != root.parent:
        if (root / "models").is_dir():
            return root
        root = root.parent
    return Path.cwd()


def _get_use_background_replace() -> bool:
    """Read background replacement switch from the active task config."""
    if _CURRENT_TASK_CONFIG is not None:
        return _CURRENT_TASK_CONFIG.runtime.use_background_replace
    return False


def _get_scene_usd_path() -> str:
    """Read the selected static scene USD from the active task config."""
    project_root = _get_project_root()
    if _CURRENT_TASK_CONFIG is not None:
        raw_path = str(_CURRENT_TASK_CONFIG.scene.usd_path or "").strip()
        if raw_path:
            scene_path = Path(raw_path).expanduser()
            if not scene_path.is_absolute():
                scene_path = project_root / scene_path
            return str(scene_path)
    return str(project_root / "models/table.usd")


def _get_sim_camera_resolution() -> tuple[int, int]:
    """Read the simulation camera resolution (width, height) from the active task config."""
    if _CURRENT_TASK_CONFIG is not None:
        return _CURRENT_TASK_CONFIG.runtime.sim_camera_resolution
    return (640, 480)

@configclass
class SceneCfg(InteractiveSceneCfg):
    """Scene configuration."""

    light_sphere = AssetBaseCfg(
        prim_path="/World/light_sphere",
        spawn=sim_utils.SphereLightCfg(intensity=0.0, color=(1.0, 1.0, 1.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.0, 0.0, 2.0)),
    )
    light_disk = AssetBaseCfg(
        prim_path="/World/light_disk",
        spawn=sim_utils.DiskLightCfg(intensity=0.0, color=(1.0, 1.0, 1.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-1.0, 0.0, 2.5)),
    )
    light_distant = AssetBaseCfg(
        prim_path="/World/light_distant",
        spawn=sim_utils.DistantLightCfg(intensity=2500.0, color=(1.0, 1.0, 1.0)),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.9238795, 0.3826834, 0.0, 0.0)),
    )

    robot: ArticulationCfg = FRANKA_FR3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(color=(0.5, 0.5, 0.5), size=(1500, 1500)),
    )
    
    external_cam_side: CameraCfg = MISSING
    
    wrist_cam: CameraCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    top_rgb_camera: CameraCfg = MISSING
    top_depth_camera: CameraCfg = MISSING

@configclass
class ActionCfg:
    body: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionZeroToOneActionCfg | mdp.BinaryJointPositionActionCfg = MISSING

@configclass
class ObservationCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations."""
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        arm_joint_pos = ObsTerm(func=mdp.arm_joint_pos)
        arm_joint_vel = ObsTerm(func=mdp.arm_joint_vel)
        arm_ee_pose = ObsTerm(func=mdp.arm_ee_pose)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)
        arm_ee_vel = ObsTerm(func=mdp.arm_ee_vel)
        
        if USE_BACKGROUND_REPLACE:
            semantic_segmentation_side = ObsTerm(
                func=mdp.image,
                params={
                    "sensor_cfg": SceneEntityCfg("external_cam_side"),
                    "data_type": "semantic_segmentation",
                    "normalize": False
                },
            )
            semantic_segmentation_wrist = ObsTerm(
                func=mdp.image,
                params={
                    "sensor_cfg": SceneEntityCfg("wrist_cam"),
                    "data_type": "semantic_segmentation",
                    "normalize": False
                },
            )

        external_cam_side = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("external_cam_side"),
                "data_type": "rgb",
                "normalize": True,
            }
        )
        
        wrist_cam = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_cam"),
                "data_type": "rgb",
                "normalize": True,
            }
        )

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True
            # self.enable_corruption = False
            # self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()

LIGHT_RANDOMIZATION_CONFIGS = [
    {
        "prim_path": "/World/light_sphere",
        "intensity_range": (500.0, 14000.0),
        "color_range": ((0.35, 0.30, 0.25), (1.0, 0.95, 0.85)),
        "position_range": ((-2.8, -2.8, 0.8), (2.8, 2.8, 4.2)),
    },
    {
        "prim_path": "/World/light_disk",
        "intensity_range": (300.0, 12000.0),
        "color_range": ((0.30, 0.35, 0.45), (1.0, 0.95, 1.0)),
        "position_range": ((-3.5, -3.0, 0.7), (3.5, 3.0, 4.5)),
    },
    {
        "prim_path": "/World/light_distant",
        "intensity_range": (100.0, 5000.0),
        "color_range": ((0.45, 0.40, 0.35), (1.0, 1.0, 0.95)),
        "direction_range": ((-85.0, -85.0, -180.0), (85.0, 85.0, 180.0)),  # rotateXYZ in degrees
    },
]

@configclass
class EventCfg:
    """
    Default event configuration.

    Note: When using YAML configuration, this class is replaced by a dynamically generated class from create_event_cfg().
    """
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # Default event (will be overridden by configuration)
    event_term = EventTerm(
        func=mdp.set_rigid_object_pose,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("basic_block_red"),
            "position_range": {
                'x': (0.45, 0.55),
                'y': (-0.2, -0.1),
                'z': (0.05, 0.05),
            },
            "orientation_range": {'yaw': (0, 0)},
        }
    )

    randomize_lights = EventTerm(
        func=mdp.randomize_lights,
        mode="reset",
        params={"light_configs": LIGHT_RANDOMIZATION_CONFIGS},
    )

@configclass
class CommandsCfg:
    """MDP commands configuration."""
    object_pose = mdp.ObjectRelativePoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        object_name="basic_block_red",  # Will be overridden by task configuration
        lift_height=0.3,
        resampling_time_range=(5.0, 5.0),
    )

@configclass
class RewardsCfg:
    """MDP rewards configuration."""
    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.1, "object_cfg": SceneEntityCfg("basic_block_red")},
        weight=0.5
    )
    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        params={
            "minimal_height": 1e-3,
            "maximum_height": 0.35,
            "object_cfg": SceneEntityCfg("basic_block_red")
        },
        weight=12.0
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    ee_orientation_penalty = RewTerm(func=mdp.ee_orientation_penalty, weight=1)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-5e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-25.0)
    object_stable_at_height = RewTerm(
        func=mdp.object_stable_at_height,
        params={
            "minimal_height": 0.15,
            "maximum_height": 0.3,
            "velocity_threshold": 1.0,
            "command_name": "object_pose",
            "reward_value": 30.0,
            "object_cfg": SceneEntityCfg("basic_block_red"),
        },
        weight=1.0,
    )
    object_stable_at_height_fine_grained = RewTerm(
        func=mdp.object_stable_at_height,
        params={
            "minimal_height": 0.15,
            "maximum_height": 0.3,
            "velocity_threshold": 0.3,
            "command_name": "object_pose",
            "reward_value": 20,
            "object_cfg": SceneEntityCfg("basic_block_red"),
        },
        weight=1.0,
    )
    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.3,
            "minimal_height": 0.15,
            "command_name": "object_pose",
            "object_cfg": SceneEntityCfg("basic_block_red")
        },
        weight=0.0,
    )
    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.05,
            "minimal_height": 0.15,
            "command_name": "object_pose",
            "object_cfg": SceneEntityCfg("basic_block_red")
        },
        weight=0.0,
    )

@configclass
class TerminationsCfg:
    """MDP termination conditions configuration."""
    time_out = DoneTerm(func=mdp.time_out, time_out=False)
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -20.0, "asset_cfg": SceneEntityCfg("basic_block_red")}
    )

@configclass
class CurriculumCfg:
    """Curriculum learning configuration."""
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )
    ee_orientation_penalty = CurrTerm(
        mdp.gradual_reward_weight_modification,
        params={
            "term_name": "ee_orientation_penalty",
            "initial_weight": 1,
            "final_weight": 10,
            "start_it": 1000,
            "end_it": 2000
        }
    )
    reaching_object = CurrTerm(
        mdp.gradual_reward_weight_modification,
        params={
            "term_name": "reaching_object",
            "initial_weight": 1,
            "final_weight": 8,
            "start_it": 0,
            "end_it": 1000
        }
    )
    action_rate = CurrTerm(
        mdp.gradual_reward_weight_modification,
        params={
            "term_name": "action_rate",
            "initial_weight": -0.0,
            "final_weight": -0.1,
            "start_it": 0,
            "end_it": 1000
        }
    )
    joint_vel = CurrTerm(
        mdp.gradual_reward_weight_modification,
        params={
            "term_name": "joint_vel",
            "initial_weight": -0.0,
            "final_weight": -0.1,
            "start_it": 0,
            "end_it": 1000
        }
    )
    dof_pos_limits = CurrTerm(
        mdp.gradual_reward_weight_modification,
        params={
            "term_name": "dof_pos_limits",
            "initial_weight": -5.0,
            "final_weight": -100.0,
            "start_it": 0,
            "end_it": 1000
        }
    )
    object_goal_tracking = CurrTerm(
        mdp.gradual_reward_weight_modification,
        params={
            "term_name": "object_goal_tracking",
            "initial_weight": 0,
            "final_weight": 16,
            "start_it": 100,
            "end_it": 101
        }
    )
    object_goal_tracking_fine_grained = CurrTerm(
        mdp.gradual_reward_weight_modification,
        params={
            "term_name": "object_goal_tracking_fine_grained",
            "initial_weight": 0,
            "final_weight": 5,
            "start_it": 100,
            "end_it": 101
        }
    )

@configclass
class EnvCfg(ManagerBasedRLEnvCfg):
    # Task configuration (set when loaded from YAML)
    task_config: Optional[TaskConfig] = None
    config_path: Optional[str] = None
    
    # Sub-configurations
    scene = SceneCfg(num_envs=512, env_spacing=4)
    observations = ObservationCfg()
    actions = ActionCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    commands = CommandsCfg()
    events = EventCfg()
    curriculum = CurriculumCfg()

    def __post_init__(self):
        """Post-initialization processing."""
        self.episode_length_s = 5
        self.viewer.eye = (4.5, 0.0, 6.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.decimation = 30
        self.sim.dt = 1 / (60 * 2)
        self.sim.render_interval = 10
        self.sim.physx.enable_ccd = True
        self.sim.physx.gpu_temp_buffer_capacity = 2**20
        self.sim.physx.gpu_heap_capacity = 2**20
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**20
        self.sim.physx.gpu_collision_stack_size = 2**22
        self.sim.physx.gpu_max_rigid_contact_count = 2**20
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**20
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**20
        self.sim.physx.gpu_max_num_partitions = 16
        self.rerender_on_reset = False

    def _apply_yaml_config(self, task_config: TaskConfig):

        # Apply simulation configuration
        self.decimation = task_config.simulation.decimation
        self.sim.dt = task_config.simulation.dt
        self.sim.render_interval = task_config.simulation.render_interval
        self.episode_length_s = task_config.simulation.episode_length_s
        # Apply scene configuration
        self.scene.env_spacing = task_config.scene.env_spacing
        self.scene.num_envs = task_config.scene.num_envs

        # Create scene object configurations (read from YAML)
        self._create_scene_objects(task_config)
        
        # Dynamically create event configuration (set object position randomization and curriculum from YAML)
        DynamicEventCfg = create_event_cfg(task_config)
        self.events = DynamicEventCfg()
        
        # Update target object references
        target_name = task_config.get_target_name()
        if target_name:
            self._update_target_object_references(target_name)
    
    def _create_scene_objects(self, task_config: TaskConfig):

        # Iterate over all object configs and create corresponding RigidObjectCfg
        for obj_name, obj_config in task_config.objects.items():
            rigid_obj_cfg = create_rigid_object_cfg(obj_config)
            setattr(self.scene, obj_name, rigid_obj_cfg)

        # Iterate over all articulation object configs and create corresponding ArticulationCfg
        for art_name, art_config in task_config.articulations.items():
            art_cfg = create_articulation_cfg(art_config)
            setattr(self.scene, art_name, art_cfg)
    
    def _update_target_object_references(self, target_name: str):
        
        # Update command configuration
        self.commands.object_pose.object_name = target_name
        
        # Update object references in reward configuration
        target_cfg = SceneEntityCfg(target_name)
        self.rewards.reaching_object.params["object_cfg"] = target_cfg
        self.rewards.lifting_object.params["object_cfg"] = target_cfg
        self.rewards.object_stable_at_height.params["object_cfg"] = target_cfg
        self.rewards.object_stable_at_height_fine_grained.params["object_cfg"] = target_cfg
        self.rewards.object_goal_tracking.params["object_cfg"] = target_cfg
        self.rewards.object_goal_tracking_fine_grained.params["object_cfg"] = target_cfg
        
        # Update termination conditions
        self.terminations.object_dropping.params["asset_cfg"] = target_cfg
    
    def get_grasp_offset(self) -> np.ndarray:
        """Get grasp offset."""
        if self.task_config:
            return self.task_config.grasp_offset
        return np.array([0, 0, 0])
    
    def get_place_offset(self) -> np.ndarray:
        """Get place offset."""
        if self.task_config:
            return self.task_config.place_offset
        return np.array([0, 0, 0])
    
    def get_task_name(self) -> str:
        """Get task name."""
        if self.task_config:
            return self.task_config.task.name
        return "unknown"
    
    def get_data_dir(self) -> str:
        """Get data save directory."""
        if self.task_config:
            return self.task_config.runtime.data_dir
        return "/data"

def create_rigid_object_cfg(obj_config: ObjectConfig) -> RigidObjectCfg:

    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{obj_config.name}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=obj_config.usd_path,
            semantic_tags=obj_config.semantic_tags,
            scale=obj_config.scale,
            # visual_material=sim_utils.PreviewSurfaceCfg(
            #     diffuse_color=obj_config.diffuse_color
            # ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=obj_config.init_pos,
            rot=obj_config.init_rot,
        ),
    )

def create_articulation_cfg(art_config: ArticulationObjectConfig) -> ArticulationCfg:
    """Create ArticulationCfg from ArticulationObjectConfig."""
    # Build actuators dict
    actuators = {}
    for act_name, act_cfg in art_config.actuators.items():
        actuators[act_name] = ImplicitActuatorCfg(
            joint_names_expr=act_cfg.joint_names_expr,
            effort_limit_sim=act_cfg.effort_limit,
            velocity_limit_sim=act_cfg.velocity_limit,
            stiffness=act_cfg.stiffness,
            damping=act_cfg.damping,
        )

    return ArticulationCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{art_config.name}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=art_config.usd_path,
            semantic_tags=art_config.semantic_tags,
            scale=art_config.scale,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=art_config.disable_gravity,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=art_config.enabled_self_collisions,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=art_config.init_pos,
            rot=art_config.init_rot,
            joint_pos=art_config.joint_pos,
        ),
        actuators=actuators,
        soft_joint_pos_limit_factor=1.0,
    )


def create_event_cfg(task_config: TaskConfig) -> type:

    @configclass
    class DynamicEventCfg:
        reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    if task_config.enable_random_lights:
        setattr(DynamicEventCfg, "randomize_lights", EventTerm(
            func=mdp.randomize_lights,
            mode="reset",
            params={"light_configs": LIGHT_RANDOMIZATION_CONFIGS},
        ))

    event_params = task_config.get_event_params()
    
    for obj_name, params in event_params.items():
        event_name = f"reset_{obj_name.replace('-', '_').replace('.', '_')}"

        if not task_config.enable_objects_randomization:
            event_term_params = {
                "asset_cfg": SceneEntityCfg(obj_name),
                "position": params["position"],
                "orientation": params["orientation"],
        }
            event_term = EventTerm(
                func=mdp.set_rigid_object_pose,
                mode="reset",
                params=event_term_params
        )
            setattr(DynamicEventCfg, event_name, event_term)

            # BUGFIX NOTE:
            # Previously this would directly `return DynamicEventCfg`, causing only
            # the first object's reset event to be registered when
            # enable_objects_randomization=False; subsequent object/articulation reset events would not take effect.
            # return DynamicEventCfg
            continue
        
        # Build EventTerm parameters
        event_term_params = {
            "asset_cfg": SceneEntityCfg(obj_name),
            "position_range": params["position_range"],
            "orientation_range": params["orientation_range"],
        }
        
        # If curriculum config exists, add curriculum_config
        if task_config.enable_objects_randomization_curriculum and params.get("curriculum_config"):
            event_term_params["curriculum_config"] = params["curriculum_config"]
        
        # Create EventTerm
        event_term = EventTerm(
            func=mdp.set_rigid_object_pose,
            mode="reset",
            params=event_term_params
        )
        
        # Add to class attributes
        setattr(DynamicEventCfg, event_name, event_term)

    # Handle articulation object randomization
    art_event_params = task_config.get_articulation_event_params()
    for art_name, params in art_event_params.items():
        event_name = f"reset_{art_name.replace('-', '_').replace('.', '_')}"

        if not task_config.enable_objects_randomization:
            event_term_params = {
                "asset_cfg": SceneEntityCfg(art_name),
                "position": params["position"],
                "orientation": params["orientation"],
            }
            event_term = EventTerm(
                func=mdp.set_articulation_pose,
                mode="reset",
                params=event_term_params,
            )
            setattr(DynamicEventCfg, event_name, event_term)
        else:
            event_term_params = {
                "asset_cfg": SceneEntityCfg(art_name),
                "position_range": params["position_range"],
                "orientation_range": params["orientation_range"],
            }
            event_term = EventTerm(
                func=mdp.set_articulation_pose,
                mode="reset",
                params=event_term_params,
            )
            setattr(DynamicEventCfg, event_name, event_term)

    return DynamicEventCfg

def create_pick_env_cfg_from_yaml(
    yaml_path: str,
    env_cfg_class: type = None,
    num_envs: int = None,
    device: str = None,
    seed: int = None,
) -> EnvCfg:
    
    global _CURRENT_TASK_CONFIG
    
    # Load YAML configuration
    task_config = load_task_config(yaml_path)
    _CURRENT_TASK_CONFIG = task_config
    
    # Determine which environment configuration class to use
    if env_cfg_class is None:
        # Default to EnvCfg base class; subclasses need explicit specification
        env_cfg_class = EnvCfg
    
    # Create environment configuration
    env_cfg = env_cfg_class()

    # Fix: @configclass instantiates the default PolicyCfg() at module import time,
    # when _CURRENT_TASK_CONFIG is still None, which may cause semantic segmentation
    # observations to be cleared prematurely. Here we restore them after instantiation
    # based on runtime configuration.
    if task_config.runtime.use_background_replace:
        policy_cfg = env_cfg.observations.policy
        if getattr(policy_cfg, "semantic_segmentation_side", None) is None:
            policy_cfg.semantic_segmentation_side = ObsTerm(
                func=mdp.image,
                params={
                    "sensor_cfg": SceneEntityCfg("external_cam_side"),
                    "data_type": "semantic_segmentation",
                    "normalize": False,
                },
            )
        if getattr(policy_cfg, "semantic_segmentation_wrist", None) is None:
            policy_cfg.semantic_segmentation_wrist = ObsTerm(
                func=mdp.image,
                params={
                    "sensor_cfg": SceneEntityCfg("wrist_cam"),
                    "data_type": "semantic_segmentation",
                    "normalize": False,
                },
            )

    env_cfg.config_path = yaml_path
    env_cfg.task_config = task_config
    
    # Apply YAML configuration
    env_cfg._apply_yaml_config(task_config)
    
    # Apply command-line argument overrides
    if num_envs is not None:
        env_cfg.scene.num_envs = num_envs
        task_config.scene.num_envs = num_envs
    if device is not None:
        env_cfg.sim.device = device
    if seed is not None:
        env_cfg.seed = seed

    print("=" * 60)
    print(f"[INFO] Loading task config from: {yaml_path}")
    print("=" * 60)
    print_task_config(task_config)
    
    return env_cfg

def get_pick_env_cfg_class_by_name(task_name: str) -> type:
    if "Kinova" in task_name and "Robotiq85" in task_name:
        from isaaclab_tasks.real2sim.tasks.config.kinova.robotiq85.joint_ee_vis_env_cfg import KinovaRobotiq85JointEEVisPickEnvCfg
        return KinovaRobotiq85JointEEVisPickEnvCfg
    elif "Sawyer" in task_name and "Robotiq85" in task_name:
        from isaaclab_tasks.real2sim.tasks.config.sawyer.robotiq85.joint_ee_vis_env_cfg import SawyerRobotiq85JointEEVisPickEnvCfg
        return SawyerRobotiq85JointEEVisPickEnvCfg
    elif "UR5e" in task_name and "Robotiq85" in task_name:
        from isaaclab_tasks.real2sim.tasks.config.ur5e.robotiq85.ik_rel_ee_vis_env_cfg import UR5eRobotiq85IKRelEEVisPickEnvCfg
        return UR5eRobotiq85IKRelEEVisPickEnvCfg
    elif "XArm7" in task_name and "UFactory" in task_name:
        from isaaclab_tasks.real2sim.tasks.config.xarm7.ufactory.joint_ee_vis_env_cfg import XArm7UFactoryJointEEVisPickEnvCfg
        return XArm7UFactoryJointEEVisPickEnvCfg
    elif "IKRel" in task_name and "OurGripper" in task_name:
        from isaaclab_tasks.real2sim.tasks.config.franka_fr3.ourgripper.ik_rel_vis_env_cfg import FrankaFr3OurGripperIKRelVisPickEnvCfg
        return FrankaFr3OurGripperIKRelVisPickEnvCfg
    elif "IKRel-EE" in task_name and "OurGripper" in task_name:
        from isaaclab_tasks.real2sim.tasks.config.franka_fr3.ourgripper.ik_rel_ee_vis_env_cfg import FrankaFr3OurGripperIKRelEEVisPickEnvCfg
        return FrankaFr3OurGripperIKRelEEVisPickEnvCfg
    elif "Joint" in task_name and "OurGripper" in task_name:
        from isaaclab_tasks.real2sim.tasks.config.franka_fr3.ourgripper.joint_vis_env_cfg import FrankaFr3OurGripperJointVisPickEnvCfg
        return FrankaFr3OurGripperJointVisPickEnvCfg
    elif "IKRel" in task_name and "DefaultGripper" in task_name:
        from isaaclab_tasks.real2sim.tasks.config.franka_fr3.defaultgripper.ik_rel_vis_env_cfg import FrankaFr3DefaultGripperIKRelVisPickEnvCfg
        return FrankaFr3DefaultGripperIKRelVisPickEnvCfg
    elif "Joint" in task_name and "DefaultGripper" in task_name:
        from isaaclab_tasks.real2sim.tasks.config.franka_fr3.defaultgripper.joint_vis_env_cfg import FrankaFr3DefaultGripperJointVisPickEnvCfg
        return FrankaFr3DefaultGripperJointVisPickEnvCfg
    else:
        return EnvCfg
