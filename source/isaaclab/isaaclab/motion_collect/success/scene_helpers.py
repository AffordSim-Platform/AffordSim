from typing import Any

import numpy as np

from .constants import POSITION_RANGE_TOLERANCE


def attach_scene_success_meta(scene, task_yaml: dict[str, Any] | None) -> None:
    """Attach the minimal YAML metadata needed by success evaluators onto the scene."""
    objects_cfg = dict((task_yaml or {}).get("objects", {}) or {})
    articulations_cfg = dict((task_yaml or {}).get("articulations", {}) or {})
    success_objects = {}
    for label, cfg in {**objects_cfg, **articulations_cfg}.items():
        success_objects[label] = {
            "position_range": dict(cfg.get("position_range", {}) or {}),
        }
    scene.success_meta = {"objects": success_objects}


def _get_success_meta(scene) -> dict[str, Any]:
    meta = getattr(scene, "success_meta", None)
    if isinstance(meta, dict):
        return meta
    return {}


def _find_object_in_scene(scene, label: str | None) -> str | None:
    if not label:
        return None

    for attr_name in ("rigid_objects", "articulations"):
        mapping = getattr(scene, attr_name, None)
        if mapping and label in mapping:
            return label

    label_lower = str(label).lower()
    for attr_name in ("rigid_objects", "articulations"):
        mapping = getattr(scene, attr_name, None)
        if not mapping:
            continue
        for key in mapping.keys():
            if str(key).lower() == label_lower:
                return key
    return None


def _get_object_pose_world_from_scene(
    scene, label: str | None, env_idx: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    object_key = _find_object_in_scene(scene, label)
    if object_key is None:
        return None, None

    if hasattr(scene, "rigid_objects") and object_key in scene.rigid_objects:
        obj = scene.rigid_objects[object_key]
    elif hasattr(scene, "articulations") and object_key in scene.articulations:
        obj = scene.articulations[object_key]
    else:
        return None, None

    try:
        pose = obj.data.root_state_w[env_idx, :7].detach().cpu().numpy()
    except Exception:
        return None, None
    return np.array(pose[:3], dtype=float), np.array(pose[3:7], dtype=float)


def _get_object_pose_env_from_scene(
    scene, label: str | None, env_idx: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    position_world, quat = _get_object_pose_world_from_scene(scene, label, env_idx)
    if position_world is None or quat is None:
        return None, None
    try:
        env_origin = scene.env_origins[env_idx].detach().cpu().numpy()
    except Exception:
        return None, None
    return np.array(position_world - env_origin, dtype=float), quat


def _get_position_range(scene, label: str | None) -> dict[str, Any] | None:
    if not label:
        return None
    objects_meta = dict(_get_success_meta(scene).get("objects", {}) or {})
    if label in objects_meta:
        position_range = objects_meta[label].get("position_range")
        return dict(position_range or {}) if position_range is not None else None

    label_lower = str(label).lower()
    for key, meta in objects_meta.items():
        if str(key).lower() == label_lower:
            position_range = dict(meta or {}).get("position_range")
            return dict(position_range or {}) if position_range is not None else None
    return None


def _is_position_in_range(
    position_env: np.ndarray | None,
    position_range: dict[str, Any] | None,
    tolerance: float = POSITION_RANGE_TOLERANCE,
    axes: tuple[str, ...] = ("x", "y"),
) -> bool:
    if position_env is None:
        return False
    if not position_range:
        return True

    axis_to_idx = {"x": 0, "y": 1, "z": 2}
    for axis in axes:
        idx = axis_to_idx[axis]
        bounds = position_range.get(axis)
        if bounds is None:
            continue
        try:
            lower, upper = float(bounds[0]), float(bounds[1])
        except Exception:
            continue
        value = float(position_env[idx])
        if value < lower - tolerance or value > upper + tolerance:
            return False
    return True


def _get_scene_num_envs(scene) -> int:
    env_origins = getattr(scene, "env_origins", None)
    if env_origins is None:
        return 0
    try:
        return int(env_origins.shape[0])
    except Exception:
        return 0


def _get_articulation_joint_pos(
    scene, label: str, joint_name: str | None, env_idx: int
) -> float | None:
    object_key = _find_object_in_scene(scene, label)
    if object_key is None:
        return None
    articulations = getattr(scene, "articulations", None)
    if not articulations or object_key not in articulations:
        return None
    art = articulations[object_key]
    try:
        joint_pos = art.data.joint_pos[env_idx].detach().cpu().numpy()
    except Exception:
        return None
    if joint_name is not None:
        try:
            joint_names = art.joint_names
            if joint_name in joint_names:
                idx = joint_names.index(joint_name)
                return float(joint_pos[idx])
        except Exception:
            pass
        return None
    if joint_pos.size > 0:
        return float(joint_pos[0])
    return None


def _get_articulation_all_joint_pos(
    scene, label: str, env_idx: int
) -> dict[str, float] | None:
    """返回 {joint_name: joint_pos} 的全部关节位置字典."""
    object_key = _find_object_in_scene(scene, label)
    if object_key is None:
        return None
    articulations = getattr(scene, "articulations", None)
    if not articulations or object_key not in articulations:
        return None
    art = articulations[object_key]
    try:
        joint_pos = art.data.joint_pos[env_idx].detach().cpu().numpy()
        joint_names = list(art.joint_names)
    except Exception:
        return None
    return {name: float(joint_pos[idx]) for idx, name in enumerate(joint_names)}
