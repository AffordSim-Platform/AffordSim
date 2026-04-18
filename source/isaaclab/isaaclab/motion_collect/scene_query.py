"""
Scene query module

Provides object pose queries, environment origin retrieval, scene description parsing, etc.
All functions are standalone functions (not mixin methods), accepting scene / env parameters.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


def find_object_in_scene(label: str, scene: Any) -> Optional[str]:
    """
    Find an object in scene assets by label.

    Based on YAML format where labels are simple names like "banana", "bowl", "can".
    The rigid_objects keys match these labels directly.

    Args:
        label: Object label from YAML (e.g., "banana")
        scene: Isaac Lab scene object with rigid_objects / articulations

    Returns:
        The key in rigid_objects/articulations if found, None otherwise
    """
    if not label:
        return None

    asset_groups = []
    if hasattr(scene, "rigid_objects"):
        asset_groups.append(scene.rigid_objects)
    if hasattr(scene, "articulations"):
        asset_groups.append(scene.articulations)

    for assets in asset_groups:
        if label in assets:
            return label

    return None


def get_object_entry_from_scene_desc(label: str, scene_desc: List[dict]) -> Optional[dict]:
    """
    Find object entry in scene description by label.

    Args:
        label: Object label to search for
        scene_desc: List of object dictionaries from YAML

    Returns:
        Object dictionary if found, None otherwise
    """
    if not label:
        return None

    for obj in scene_desc:
        if obj.get('label') == label:
            return obj
    return None


def get_object_pose_global(label: str, env_idx: int, scene: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Retrieve global pose of an object from simulation.

    Args:
        label: Object label (e.g., "banana", "bowl")
        env_idx: Environment index
        scene: Isaac Lab scene object

    Returns:
        Tuple of (position, quaternion_wxyz) or (None, None) if not found
    """
    object_key = find_object_in_scene(label, scene)
    if object_key is None:
        return None, None

    if hasattr(scene, "rigid_objects") and object_key in scene.rigid_objects:
        obj_prim = scene.rigid_objects[object_key]
    elif hasattr(scene, "articulations") and object_key in scene.articulations:
        obj_prim = scene.articulations[object_key]
    else:
        return None, None

    pose = obj_prim.data.root_state_w[env_idx, :7].cpu().numpy()
    return pose[:3], pose[3:]


def get_object_pose(object_name: str, scene: Any, env_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get object pose relative to environment origin.

    Args:
        object_name: Name of the object in scene
        scene: Isaac Lab scene object
        env_idx: Environment index

    Returns:
        Tuple of (position_relative, quaternion_wxyz)
    """
    rigid_obj = scene[object_name]
    root_state = rigid_obj.data.root_state_w[env_idx]

    position_world = root_state[:3].cpu().numpy()
    quaternion = root_state[3:7].cpu().numpy()

    env_origin = scene.env_origins[env_idx].cpu().numpy()
    position_relative = position_world - env_origin

    return position_relative, quaternion


def get_env_origins(scene: Any) -> np.ndarray:
    """
    Safely retrieve environment origins.

    Args:
        scene: Isaac Lab scene object (or env.env.scene)

    Returns:
        numpy array of env origins [num_envs, 3]
    """
    return scene.env_origins.cpu().numpy()


def get_object_poses_for_env(env_idx: int, scene: Any) -> Dict[str, np.ndarray]:
    """
    Get current positions of all rigid objects for a specific environment,
    relative to the environment origin.

    Uses rigid_obj.data.root_state_w to get actual simulation positions.

    Args:
        env_idx: Environment index
        scene: Isaac Lab scene object

    Returns:
        Dictionary mapping object names to their positions relative to env origin
    """
    object_poses = {}

    if not hasattr(scene, "rigid_objects"):
        return object_poses

    env_origin = scene.env_origins[env_idx].cpu().numpy()

    for name, obj in scene.rigid_objects.items():
        if hasattr(obj.data, "root_state_w"):
            root_state = obj.data.root_state_w[env_idx]
            position_world = root_state[:3].cpu().numpy()
            position_relative = position_world - env_origin

            if position_relative[2] > -0.1:
                object_poses[name] = position_relative

    return object_poses


def rebuild_scene_desc_from_yaml(task_config_path: str, yaml_parser: Any) -> Tuple[List[dict], str]:
    """
    Parse the YAML configuration to reconstruct scene description.

    Args:
        task_config_path: Path to task YAML config file
        yaml_parser: YAML parser instance (ruamel.yaml.YAML)

    Returns:
        Tuple of (scene_desc list, task_instruction string)
    """
    print(f"[INFO] Loading config from: {task_config_path}")
    with open(task_config_path, 'r', encoding='utf-8') as f:
        config = yaml_parser.load(f)

    scene_desc = []
    objects = config.get('objects', {})
    articulations = config.get('articulations', {})
    task_instruction = config.get('task', {}).get('description', "Unknown task")

    for obj_key, obj_data in objects.items():
        usd_path = obj_data.get('usd_path', '')
        init_pos = obj_data.get('init_pos', [0, 0, 0])
        init_rot = obj_data.get('init_rot', [1, 0, 0, 0])
        height = obj_data.get('height', None)

        obj_id = Path(usd_path).stem if usd_path else "000"

        scene_desc.append({
            "id": obj_id,
            "label": obj_key,
            "position": {"x": init_pos[0], "y": init_pos[1], "z": init_pos[2]},
            "orientation": {"w": init_rot[0], "x": init_rot[1], "y": init_rot[2], "z": init_rot[3]},
            "height": height
        })

    for art_key, art_data in articulations.items():
        usd_path = art_data.get('usd_path', '')
        init_pos = art_data.get('init_pos', [0, 0, 0])
        init_rot = art_data.get('init_rot', [1, 0, 0, 0])
        obj_id = Path(usd_path).stem if usd_path else art_key

        scene_desc.append({
            "id": obj_id,
            "label": art_key,
            "position": {"x": init_pos[0], "y": init_pos[1], "z": init_pos[2]},
            "orientation": {"w": init_rot[0], "x": init_rot[1], "y": init_rot[2], "z": init_rot[3]},
            "height": None,
        })

    print(f"[INFO] Loaded {len(scene_desc)} objects: {[obj['label'] for obj in scene_desc]}")
    return scene_desc, task_instruction
