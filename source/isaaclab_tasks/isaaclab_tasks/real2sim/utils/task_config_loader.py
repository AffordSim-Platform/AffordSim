"""
Task Config Loader

This module loads and parses task configurations from YAML files.
Each task corresponds to an independent YAML configuration file.
"""

import yaml
import numpy as np
import math
from pathlib import Path
from typing import Dict, Optional, Any, Tuple, List, Union
from dataclasses import dataclass, field


def parse_phase_depenetration(data: Optional[Dict[str, Any]], objects: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Parse an opt-in policy restricted to named task rigid objects."""
    if data is None:
        return None
    if not isinstance(data, dict) or set(data) != {"objects", "reset", "primitives"}:
        raise ValueError("phase_depenetration requires exactly objects, reset and primitives")
    names = data["objects"]
    if (not isinstance(names, list) or not names or any(not isinstance(n, str) or not n or not all(c.isascii() and (c.isalnum() or c == '_') for c in n) for n in names)
            or len(names) != len(set(names)) or not set(names) <= set(objects) or "robot" in names):
        raise ValueError("phase_depenetration objects must be unique names from task objects")
    phases = data["primitives"]
    if not isinstance(phases, dict) or set(phases) not in ({"pick", "place"}, {"pick", "release"}):
        raise ValueError("phase_depenetration primitives must specify pick and exactly one of place or release")

    def value(number):
        if (isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number)
                or number <= 0 or number > 3.4028234663852886e38):
            raise ValueError("phase_depenetration values must be in (0, float32 max]")
        return float(number)

    return {"objects": list(names), "reset": value(data["reset"]),
            "primitives": {name: value(number) for name, number in phases.items()}}

@dataclass
class PositionCurriculumConfig:
    """
    Object position curriculum learning configuration.

    Attributes:
        initial_position_range: Initial position range for curriculum (smaller/easier).
        final_position_range: Final position range for curriculum (larger/harder), uses object's position_range.
        start_iteration: Iteration at which the curriculum starts.
        end_iteration: Iteration at which the curriculum ends.
    """
    initial_position_range: Dict[str, Tuple[float, float]]
    final_position_range: Dict[str, Tuple[float, float]]
    start_iteration: int
    end_iteration: int
    
    @classmethod
    def from_dict(cls, data: Dict) -> Optional["PositionCurriculumConfig"]:
        """Create PositionCurriculumConfig from dict. Returns None if data is None."""
        if data is None:
            return None
        
        # Parse initial position range
        initial_position_range = {}
        final_position_range = {}
        if data.get("initial_position_range"):
            for axis, values in data["initial_position_range"].items():
                initial_position_range[axis] = tuple(values)
        if data.get("final_position_range"):
            for axis, values in data["final_position_range"].items():
                final_position_range[axis] = tuple(values)
        
        return cls(
            initial_position_range=initial_position_range,
            final_position_range=final_position_range,
            start_iteration=data.get("start_iteration", 1000),
            end_iteration=data.get("end_iteration", 1500)
        )
    
@dataclass
class SimulationConfig:
    """Simulation configuration."""
    decimation: int
    dt: float
    render_interval: int
    episode_length_s: float
    
    @classmethod
    def from_dict(cls, data: Dict) -> "SimulationConfig":
        return cls(
            decimation=data.get("decimation", 30),
            dt=data.get("dt", 1 / 120),  # default 1 / (60 * 2)
            render_interval=data.get("render_interval", 10),
            episode_length_s=data.get("episode_length_s", 5.0)
        )
    
@dataclass
class SceneConfig:
    """Scene configuration."""
    env_spacing: Union[float, Tuple[float, float]]
    num_envs: int
    name: str
    usd_path: str
    prim_path: str
    table_bounds: Dict[str, Any]
    
    @classmethod
    def from_dict(cls, data: Dict) -> "SceneConfig":
        env_spacing = data.get("env_spacing", 4.0)
        if isinstance(env_spacing, dict):
            default = env_spacing.get("xy", 4.0)
            env_spacing = (float(env_spacing.get("x", default)), float(env_spacing.get("y", default)))
        elif isinstance(env_spacing, (list, tuple)):
            if len(env_spacing) != 2:
                raise ValueError("scene.env_spacing list/tuple must contain exactly two values: [x, y].")
            env_spacing = (float(env_spacing[0]), float(env_spacing[1]))
        else:
            env_spacing = float(env_spacing)
        return cls(
            env_spacing=env_spacing,
            num_envs=data.get("num_envs", 1),
            name=data.get("name", "table"),
            usd_path=data.get("usd_path", "models/table.usd"),
            prim_path=data.get("prim_path", "{ENV_REGEX_NS}/table"),
            table_bounds=data.get("table_bounds", {}) or {},
        )
    
@dataclass
class TaskInfo:
    """Basic task information."""
    type: str
    name: str
    # scene_tag: int
    description: str
    
    @classmethod
    def from_dict(cls, data: Dict) -> "TaskInfo":
        return cls(
            type=data.get("type", ""),
            name=data.get("name", ""),
            # scene_tag=data.get("scene_tag", 0),
            description=data.get("description", "")
        )

@dataclass
class RuntimeConfig:
    """Runtime configuration."""
    headless: bool
    episodes: int
    save_npy: bool
    save_camera_images: bool
    save_composite_images: bool
    save_mask_images: bool
    save_original_images: bool
    save_depth_images: bool
    enable_motion_planning: bool
    data_dir: str
    task_name: str
    use_background_replace: bool
    sim_camera_resolution: Tuple[int, int]  # (width, height) simulation camera resolution
    background_image_size: Tuple[int, int]  # (width, height) background image / target output size
    cam_side_background: str
    cam_wrist_background: str
    da3_background: Optional[Dict[str, Any]] = None
    
    @staticmethod
    def _find_project_root(start_dir: Path) -> Path:
        """Return the project root.

        Derived from this file's fixed location at
        source/isaaclab_tasks/isaaclab_tasks/real2sim/utils/task_config_loader.py
        so resolution does not depend on ``.git`` (which may be missing for zip
        installs or shadowed by a stray ``.git`` higher up the tree).
        """
        return Path(__file__).resolve().parents[5]

    @classmethod
    def from_dict(cls, data: Dict, config_dir: Optional[Path] = None) -> "RuntimeConfig":
        data_dir = data.get("data_dir", "/data")
        # If relative path, resolve to absolute path based on project root
        if config_dir is not None and not Path(data_dir).is_absolute():
            project_root = cls._find_project_root(config_dir)
            data_dir = str((project_root / data_dir).resolve())

        # Parse background image paths (relative paths resolved from project root)
        cam_side_bg = data.get("cam_side_background", "")
        cam_wrist_bg = data.get("cam_wrist_background", "")
        if config_dir is not None:
            project_root = cls._find_project_root(config_dir)
            if cam_side_bg and not Path(cam_side_bg).is_absolute():
                cam_side_bg = str((project_root / cam_side_bg).resolve())
            if cam_wrist_bg and not Path(cam_wrist_bg).is_absolute():
                cam_wrist_bg = str((project_root / cam_wrist_bg).resolve())

        # Keep the rendered background and its camera calibration together.
        # The export's USER frame is already the metric robot frame.
        da3_background = None
        da3_settings = data.get("da3_background")
        if da3_settings is not None:
            from .da3_background import load_da3_background_image

            if not isinstance(da3_settings, dict) or not da3_settings.get("image_path"):
                raise ValueError(
                    "runtime.da3_background requires image_path; specify the background image "
                    "path instead of scene_dir/view_index"
                )
            project_root = cls._find_project_root(Path(__file__).parent)
            image_path = Path(da3_settings["image_path"]).expanduser()
            if not image_path.is_absolute():
                image_path = project_root / image_path
            calibration_path = da3_settings.get("calibration_path")
            if calibration_path is not None:
                calibration_path = Path(calibration_path).expanduser()
                if not calibration_path.is_absolute():
                    calibration_path = project_root / calibration_path
            da3_background = load_da3_background_image(
                image_path,
                calibration_path=calibration_path,
                output_size=tuple(data.get("sim_camera_resolution", [640, 480])),
            )
            da3_background["keep_table"] = bool(da3_settings.get("keep_table", True))
            cam_side_bg = da3_background["image_path"]

        return cls(
            headless=data.get("headless", False),
            episodes=data.get("episodes", 50),
            save_npy=data.get("save_npy", True),
            save_camera_images=data.get("save_camera_images", True),
            save_composite_images=data.get("save_composite_images", False),
            save_mask_images=data.get("save_mask_images", False),
            save_original_images=data.get("save_original_images", False),
            save_depth_images=data.get("save_depth_images", False),
            enable_motion_planning=data.get("enable_motion_planning", True),
            data_dir=data_dir,
            task_name=data.get("task_name", ""),
            use_background_replace=da3_background is not None or data.get("use_background_replace", False),
            sim_camera_resolution=tuple(data.get("sim_camera_resolution", [640, 480])),
            background_image_size=tuple(data.get("background_image_size", [480, 480])),
            cam_side_background=cam_side_bg,
            cam_wrist_background=cam_wrist_bg,
            da3_background=da3_background,
        )

@dataclass
class PrimitiveConfig:
    """Primitive step configuration."""

    type: str
    mode: Optional[str]
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["PrimitiveConfig"]:
        """Create PrimitiveConfig from dict. Returns None if data is None."""
        if data is None:
            return None

        return cls(
            type=data.get("type", ""),
            mode=data.get("mode"),
            params=dict(data.get("params", {}) or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Export as a plain dict for easy consumption by downstream modules."""
        primitive_dict = {
            "type": self.type,
            "params": self.params,
        }
        if self.mode is not None:
            primitive_dict["mode"] = self.mode
        return primitive_dict

@dataclass
class RobotConfig:
    """Robot configuration."""
    cfg_name: str
    type: str
    
    @classmethod
    def from_dict(cls, data: Dict) -> "RobotConfig":
        return cls(
            cfg_name=data.get("cfg_name", "franka_fr3.yml"),
            type=data.get("type", "franka_fr3_default")
        )

@dataclass
class ObjectConfig:
    """
    Object configuration class.

    Attributes:
        name: Object name (corresponds to the asset name in the scene).
        usd_path: USD file path.
        semantic_tags: Semantic tag list, e.g. [("class", "cube")].
        diffuse_color: Diffuse color (RGB, 0-1 range). If None, uses the original USD color.
        init_pos: Initial position [x, y, z].
        init_rot: Initial rotation [w, x, y, z] quaternion.
        position_range: Position randomization range {'x': (min, max), 'y': (min, max), 'z': (min, max)}.
        orientation_range: Orientation randomization range {'yaw': (min, max), ...}.
        curriculum: Position curriculum learning configuration (optional).
        mass: Optional rigid-body mass in kg; omitted preserves the USD value.
        max_depenetration_velocity: Optional per-object override in m/s.
    """
    name: str
    usd_path: str
    semantic_tags: List[Tuple[str, str]]
    scale: Tuple[float, float, float]
    diffuse_color: Optional[Tuple[float, float, float]]
    init_pos: Tuple[float, float, float]
    init_rot: Tuple[float, float, float, float]
    position_range: Dict[str, Tuple[float, float]]
    orientation_range: Dict[str, Tuple[float, float]]
    curriculum: Optional[PositionCurriculumConfig] = None
    mass: Optional[float] = None
    max_depenetration_velocity: Optional[float] = None
    
    @classmethod
    def from_dict(cls, data: Dict) -> Optional["ObjectConfig"]:
        """Create ObjectConfig from dict. Returns None if data is None."""
        if data is None:
            return None
        
        # Parse semantic tags
        semantic_tags = []
        if data.get("semantic_tags"):
            for tag in data["semantic_tags"]:
                semantic_tags.append(tuple(tag))
        
        # Parse color - None if not configured
        diffuse_color = None
        if data.get("diffuse_color") is not None:
            diffuse_color = tuple(data.get("diffuse_color"))

        scale = tuple(data.get("scale", [1.0, 1.0, 1.0]))
        
        # Parse initial position and rotation
        init_pos = tuple(data.get("init_pos", [0.5, 0.0, 0.05]))
        init_rot = tuple(data.get("init_rot", [1.0, 0.0, 0.0, 0.0]))
        
        # Parse position range
        position_range = {}
        if data.get("position_range"):
            for axis, values in data["position_range"].items():
                position_range[axis] = tuple(values)
        
        # Parse orientation range
        orientation_range = {}
        if data.get("orientation_range"):
            for axis, values in data["orientation_range"].items():
                orientation_range[axis] = tuple(values)
        
        # Parse curriculum configuration
        curriculum = PositionCurriculumConfig.from_dict(data.get("curriculum"))
        mass = data.get("mass")
        if mass is not None:
            mass = float(mass)
            if not np.isfinite(mass) or mass <= 0:
                raise ValueError(f"Object {data['name']} mass must be a finite positive value in kg")
        depenetration = data.get("max_depenetration_velocity")
        if depenetration is not None:
            depenetration = float(depenetration)
            if not np.isfinite(depenetration) or depenetration <= 0:
                raise ValueError(f"Object {data['name']} max_depenetration_velocity must be finite and positive")
        
        return cls(
            name=data["name"],
            usd_path=data.get("usd_path", ""),
            scale=scale,
            semantic_tags=semantic_tags,
            diffuse_color=diffuse_color,
            init_pos=init_pos,
            init_rot=init_rot,
            position_range=position_range,
            orientation_range=orientation_range,
            curriculum=curriculum,
            mass=mass,
            max_depenetration_velocity=depenetration,
        )
    
    def has_curriculum(self) -> bool:
        """Check if curriculum learning configuration exists."""
        return self.curriculum is not None 
    
    def get_curriculum_config(self) -> Optional[Dict[str, Any]]:

        if not self.has_curriculum():
            return None

        return {
            "initial_ranges": self.curriculum.initial_position_range,
            "final_ranges": self.curriculum.final_position_range,
            "start_it": self.curriculum.start_iteration,
            "end_it": self.curriculum.end_iteration,
        }

@dataclass
class ActuatorConfig:
    """Actuator configuration."""
    joint_names_expr: List[str]
    effort_limit: float
    velocity_limit: float
    stiffness: float
    damping: float

    @classmethod
    def from_dict(cls, data: Dict) -> "ActuatorConfig":
        return cls(
            joint_names_expr=data.get("joint_names_expr", []),
            effort_limit=data.get("effort_limit", 1.0),
            velocity_limit=data.get("velocity_limit", 1.0),
            stiffness=data.get("stiffness", 1.0),
            damping=data.get("damping", 1.0),
        )

@dataclass
class ArticulationObjectConfig:
    """
    Articulation object configuration class (e.g., microwave and other jointed objects).

    Attributes:
        name: Object name.
        usd_path: USD file path.
        semantic_tags: Semantic tag list, e.g. [("class", "microwave")].
        scale: Scale factor.
        disable_gravity: Whether to disable gravity.
        enabled_self_collisions: Whether to enable self-collisions.
        init_pos: Initial position [x, y, z].
        init_rot: Initial rotation [w, x, y, z].
        joint_pos: Initial joint positions {joint_name: value}.
        actuators: Actuator configuration {actuator_name: ActuatorConfig}.
        position_range: Position randomization range.
        orientation_range: Orientation randomization range.
    """
    name: str
    usd_path: str
    semantic_tags: List[Tuple[str, str]]
    scale: Tuple[float, float, float]
    disable_gravity: bool
    enabled_self_collisions: bool
    init_pos: Tuple[float, float, float]
    init_rot: Tuple[float, float, float, float]
    joint_pos: Dict[str, float]
    actuators: Dict[str, ActuatorConfig]
    position_range: Dict[str, Tuple[float, float]]
    orientation_range: Dict[str, Tuple[float, float]]

    @classmethod
    def from_dict(cls, data: Dict) -> Optional["ArticulationObjectConfig"]:
        if data is None:
            return None

        semantic_tags = []
        if data.get("semantic_tags"):
            for tag in data["semantic_tags"]:
                semantic_tags.append(tuple(tag))

        scale = tuple(data.get("scale", [1.0, 1.0, 1.0]))
        init_pos = tuple(data.get("init_pos", [0.5, 0.0, 0.05]))
        init_rot = tuple(data.get("init_rot", [1.0, 0.0, 0.0, 0.0]))
        joint_pos = data.get("joint_pos", {})

        actuators = {}
        for act_name, act_data in data.get("actuators", {}).items():
            actuators[act_name] = ActuatorConfig.from_dict(act_data)

        position_range = {}
        if data.get("position_range"):
            for axis, values in data["position_range"].items():
                position_range[axis] = tuple(values)

        orientation_range = {}
        if data.get("orientation_range"):
            for axis, values in data["orientation_range"].items():
                orientation_range[axis] = tuple(values)

        return cls(
            name=data["name"],
            usd_path=data.get("usd_path", ""),
            semantic_tags=semantic_tags,
            scale=scale,
            disable_gravity=data.get("disable_gravity", False),
            enabled_self_collisions=data.get("enabled_self_collisions", True),
            init_pos=init_pos,
            init_rot=init_rot,
            joint_pos=joint_pos,
            actuators=actuators,
            position_range=position_range,
            orientation_range=orientation_range,
        )

@dataclass
class TaskConfig:
    """
    Complete task configuration class.

    This is the full task configuration loaded from a single YAML file.

    Current YAML structure:
    - objects: Defines all objects (key is the object name).
    - primitives: Defines the primitive sequence.
    """
    task: TaskInfo
    simulation: SimulationConfig
    scene: SceneConfig
    runtime: RuntimeConfig
    robot: RobotConfig
    grasp_offset: np.ndarray
    place_offset: np.ndarray
    objects: Dict[str, ObjectConfig]  # All objects, key is the object name
    articulations: Dict[str, ArticulationObjectConfig]  # All articulation objects
    target_name: Optional[str]        # Target object name
    container_name: Optional[str]     # Container object name
    primitives: List[PrimitiveConfig] # Primitive sequence definition
    enable_random_lights: bool
    enable_objects_randomization: bool
    enable_objects_randomization_curriculum: bool
    image_noise_std: float
    image_speckle_noise_std: float
    phase_depenetration: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, data: Dict, config_dir: Optional[Path] = None) -> "TaskConfig":
        """Create TaskConfig from dict."""
        # Parse all object configurations
        objects_data = data.get("objects", {})
        objects = {}
        for obj_name, obj_data in objects_data.items():
            if obj_data is not None:
                # Pass object name into configuration
                obj_data_with_name = obj_data.copy()
                obj_data_with_name["name"] = obj_name
                objects[obj_name] = ObjectConfig.from_dict(obj_data_with_name)
        
        # Parse articulation object configuration
        articulations_data = data.get("articulations", {})
        articulations = {}
        for art_name, art_data in articulations_data.items():
            if art_data is not None:
                art_data_with_name = art_data.copy()
                art_data_with_name["name"] = art_name
                articulations[art_name] = ArticulationObjectConfig.from_dict(art_data_with_name)

        # Parse primitive sequence
        primitives_data = data.get("primitives", [])
        primitives = []
        for primitive_data in primitives_data:
            primitive = PrimitiveConfig.from_dict(primitive_data)
            if primitive is not None:
                primitives.append(primitive)

        # Auto-infer target/container from primitives
        target_name = None
        container_name = None

        for primitive in primitives:
            primitive_target = primitive.params.get("target")
            if primitive.type in {"pick", "open", "push"} and primitive_target:
                target_name = primitive_target
                break
        if target_name is None:
            for primitive in primitives:
                primitive_target = primitive.params.get("target")
                if primitive_target:
                    target_name = primitive_target
                    break

        for primitive in primitives:
            primitive_target = primitive.params.get("target")
            if primitive.type in {"place", "pour", "hang"} and primitive_target:
                container_name = primitive_target
                break

        # Parse offsets
        offsets = data.get("offsets", {})
        grasp_offset = np.array(offsets.get("grasp", [0, 0, 0.12]))
        place_offset = np.array(offsets.get("place", [0, 0, 0.12]))
        
        return cls(
            task=TaskInfo.from_dict(data.get("task", {})),
            simulation=SimulationConfig.from_dict(data.get("simulation", {})),
            scene=SceneConfig.from_dict(data.get("scene", {})),
            runtime=RuntimeConfig.from_dict(data.get("runtime", {}), config_dir=config_dir),
            robot=RobotConfig.from_dict(data.get("robot", {})),
            grasp_offset=grasp_offset,
            place_offset=place_offset,
            objects=objects,
            articulations=articulations,
            target_name=target_name,
            container_name=container_name,
            primitives=primitives,
            enable_random_lights=data.get("enable_random_lights", True),
            enable_objects_randomization=data.get("enable_objects_randomization", True),
            enable_objects_randomization_curriculum=data.get("enable_objects_randomization_curriculum", True),
            image_noise_std=data.get("image_noise_std", 0.5),
            image_speckle_noise_std=data.get("image_speckle_noise_std", 0.0),
            phase_depenetration=parse_phase_depenetration(data.get("phase_depenetration"), objects),
        )
    
    def get_all_objects(self) -> List[ObjectConfig]:
        """Get list of all object configurations."""
        return list(self.objects.values())
    
    def get_object_by_name(self, name: str) -> Optional[ObjectConfig]:
        """Get object configuration by name."""
        return self.objects.get(name)
    
    def get_target(self) -> Optional[ObjectConfig]:
        """Get target object configuration."""
        if self.target_name:
            return self.objects.get(self.target_name)
        return None
    
    def get_container(self) -> Optional[ObjectConfig]:
        """Get container object configuration."""
        if self.container_name:
            return self.objects.get(self.container_name)
        return None
    
    def get_target_name(self) -> Optional[str]:
        """Get target object name."""
        return self.target_name
    
    def get_container_name(self) -> Optional[str]:
        """Get container object name."""
        return self.container_name

    def has_primitives(self) -> bool:
        """Check whether the current task has a primitive sequence configured."""
        return len(self.primitives) > 0

    def get_primitives(self, as_dict: bool = False) -> List[Any]:
        """Get primitive sequence, optionally exported as a list of dicts."""
        if as_dict:
            return [primitive.to_dict() for primitive in self.primitives]
        return list(self.primitives)
    
    def get_event_params(self) -> Dict[str, Dict[str, Any]]:

        params = {}
        
        for obj in self.get_all_objects():
            if self.enable_objects_randomization:
                # Missing axes stay at their initial positions. Keep this fallback
                # local to reset events: writing it into obj.position_range would
                # also introduce a new bound in the physical success evaluator.
                obj_params = {
                    "position_range": {
                        axis: obj.position_range.get(axis, (obj.init_pos[i], obj.init_pos[i]))
                        for i, axis in enumerate(("x", "y", "z"))
                    },
                    "orientation_range": obj.orientation_range or {"yaw": (0.0, 0.0)}
                }
                
                # If curriculum config exists, add curriculum_config
                if obj.has_curriculum():
                    obj_params["curriculum_config"] = obj.get_curriculum_config()
                
                params[obj.name] = obj_params
            else:   
                # Randomization disabled: use init_pos and init_rot as fixed values
                params[obj.name] = {
                    "position": obj.init_pos,
                    "orientation": {"yaw": 0.0}  # init_rot is already applied at USD load time, keep default here
                }
        
        return params

    def get_articulation_event_params(self) -> Dict[str, Dict[str, Any]]:
        """Get event parameters for articulation objects (used for position randomization)."""
        params = {}

        for art in self.articulations.values():
            if self.enable_objects_randomization:
                params[art.name] = {
                    "position_range": {
                        axis: art.position_range.get(axis, (art.init_pos[i], art.init_pos[i]))
                        for i, axis in enumerate(("x", "y", "z"))
                    },
                    "orientation_range": art.orientation_range or {"yaw": (0.0, 0.0)},
                }
            else:
                params[art.name] = {
                    "position": art.init_pos,
                    "orientation": {"yaw": 0.0},
                }

        return params

    def to_datacollector_dict(self) -> Dict[str, Any]:
        """
        Convert TaskConfig to a parameter dict required by DataCollector.

        Returns:
            Dict: Keyword argument dict for DataCollector.
        """
        # Determine randomize mode
        if self.enable_objects_randomization:
            randomize = "Random"
        else:
            randomize = "None"
        
        return {
            "scene": self.task.name,
            "randomize": randomize,
            "robot_type": self.robot.type,
            "robot_cfg_name": self.robot.cfg_name,
            "headless": self.runtime.headless,
            "enable_cameras": True,  # Cameras enabled by default
            "enable_motion_planning": self.runtime.enable_motion_planning,
            "device": "cuda:0",  # Default device
            "save_npy": self.runtime.save_npy,
            "data_dir": self.runtime.data_dir,
            "save_camera_images": self.runtime.save_camera_images,
            "save_composite_images": self.runtime.save_composite_images,
            "save_mask_images": self.runtime.save_mask_images,
            "save_original_images": self.runtime.save_original_images,
            "save_depth_images": self.runtime.save_depth_images,
            "episodes": self.runtime.episodes,
            "num_envs": self.scene.num_envs,
            "task_name": self.runtime.task_name,
        }

def load_task_config(config_path: str) -> TaskConfig:

    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        raw_config = yaml.safe_load(f)
    
    return TaskConfig.from_dict(raw_config, config_dir=config_path.parent.resolve())

def print_task_config(config: TaskConfig):
    """Print task configuration info (for debugging)."""
    print("=" * 60)
    print(f"decimation: {config.simulation.decimation}")
    print(f"dt: {config.simulation.dt}")
    print(f"render_interval: {config.simulation.render_interval}")
    print(f"episode_length_s: {config.simulation.episode_length_s}")
    print("-" * 60)
    print(f"Num envs: {config.scene.num_envs}")
    print(f"Env spacing: {config.scene.env_spacing}")
    print("-" * 60)
    print(f"Task name: {config.task.name}")
    print(f"Task type: {config.task.type}")
    # print(f"Scene tag: {config.task.scene_tag}")
    print(f"Description: {config.task.description}")
    print("-" * 60)
    print(f"Robot type: {config.robot.type}")
    print(f"Num envs: {config.scene.num_envs}")
    print(f"Enable random lights: {config.enable_random_lights}")
    print(f"Image noise std: {config.image_noise_std}")
    print(f"Image speckle noise std: {config.image_speckle_noise_std}")
    print(f"Enable objects randomization: {config.enable_objects_randomization}")
    print(f"Enable objects randomization curriculum: {config.enable_objects_randomization_curriculum}")
    print("-" * 60)
    print(f"Target object: {config.target_name}")
    print(f"Container object: {config.container_name}")
    print(f"Has primitives: {config.has_primitives()}")
    print("-" * 60)
    print("Object configurations:")
    
    def print_object_config(obj: ObjectConfig):
        print(f"  [{obj.name}]")
        print(f"    USD path: {obj.usd_path}")
        print(f"    Scale: {obj.scale}")
        print(f"    Semantic tags: {obj.semantic_tags}")
        print(f"    Color: {obj.diffuse_color}")
        print(f"    Init pos: {obj.init_pos}")
        if config.enable_objects_randomization:
            print(f"    Position range: {obj.position_range}")
            print(f"    Orientation range: {obj.orientation_range}")
        if config.enable_objects_randomization_curriculum and obj.has_curriculum():
            print(f"    Curriculum: enabled")
            print(f"      Initial range: {obj.curriculum.initial_position_range}")
            print(f"      Final range: {obj.curriculum.final_position_range}")
            print(f"      Iteration range: [{obj.curriculum.start_iteration}, {obj.curriculum.end_iteration}]")
    
    for obj in config.get_all_objects():
        print_object_config(obj)

    if config.articulations:
        print("-" * 60)
        print("Articulation object configurations:")
        for art in config.articulations.values():
            print(f"  [{art.name}]")
            print(f"    USD path: {art.usd_path}")
            print(f"    Semantic tags: {art.semantic_tags}")
            print(f"    Scale: {art.scale}")
            print(f"    Init pos: {art.init_pos}")
            print(f"    Init rot: {art.init_rot}")
            print(f"    Joint init pos: {art.joint_pos}")
            if config.enable_objects_randomization and art.position_range:
                print(f"    Position range: {art.position_range}")
                print(f"    Orientation range: {art.orientation_range}")

    if config.has_primitives():
        print("-" * 60)
        print("Primitive configurations:")
        for index, primitive in enumerate(config.get_primitives(as_dict=True), start=1):
            print(f"  [{index}] {primitive['type']}")
            if "mode" in primitive:
                print(f"    mode: {primitive['mode']}")
            print(f"    params: {primitive.get('params', {})}")

    print("=" * 60)
