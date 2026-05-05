import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation

from ..success.scene_helpers import attach_scene_success_meta
from ..tools.affordance_resolver import resolve_affordance_path
from ..tools.pose_utils import (
    normalize_ee_orientation_wxyz,
    _quat_wxyz_to_matrix,
    _quat_wxyz_to_xyzw,
    _quat_xyzw_to_wxyz,
)


# =========================================================================
# Grasp helpers (moved from datacollector.py, batch 4)
# =========================================================================

def get_pick_robot_params() -> Dict[str, Any]:
    """Return default parameters for AnyGrasp/affordance pick tasks."""
    return {"lift_offset": 0.1}


def get_pick_pos_robot_params() -> Dict[str, Any]:
    """Return default parameters for GT pose pick tasks."""
    return {"lift_offset": 0.0}


def load_grasp_config(task_config_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load GT grasp offset and base quaternion from task YAML."""
    with open(task_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    offsets = cfg.get("offsets", {})
    grasp_pos_offset = np.array(offsets.get("grasp", [0.0, 0.0, 0.0]), dtype=float)
    grasp_quat = np.array(offsets.get("grasp_quat", [0.0, -1.0, 0.0, 0.0]), dtype=float)
    grasp_quat /= np.linalg.norm(grasp_quat)
    return grasp_pos_offset, grasp_quat


def compute_aligned_grasp_quat(
    obj_quat_wxyz: np.ndarray,
    base_grasp_quat_wxyz: np.ndarray,
) -> np.ndarray:
    """Align a base downward grasp quaternion with the object's world yaw."""
    w, x, y, z = obj_quat_wxyz
    obj_rot = Rotation.from_quat([x, y, z, w])
    yaw = obj_rot.as_euler("ZYX")[0]

    rot_yaw = Rotation.from_euler("z", yaw)
    bw, bx, by, bz = base_grasp_quat_wxyz
    rot_base = Rotation.from_quat([bx, by, bz, bw])
    rot_grasp = rot_yaw * rot_base

    qx, qy, qz, qw = rot_grasp.as_quat()
    return np.array([qw, qx, qy, qz], dtype=float)


# =========================================================================
# Target availability
# =========================================================================

def compute_target_availability_mask(collector, target_label: Optional[str]) -> List[bool]:
    """Return per-env availability for an explicit target label in the current scene."""
    if not target_label:
        return [True] * collector.num_envs
    availability = []
    for env_idx in range(collector.num_envs):
        pos, _ = collector._get_object_pose_global(target_label, env_idx)
        availability.append(pos is not None)
    return availability


# =========================================================================
# Dual-line success tracking helpers (planning line + task line)
# =========================================================================

# Failure reason categories (single source of truth).
FAILURE_REASON_NONE = "none"
FAILURE_REASON_INACTIVE = "inactive"
FAILURE_REASON_TARGET_MISSING = "target_missing"
FAILURE_REASON_PLANNING_FAILED = "planning_failed"
FAILURE_REASON_TASK_FAILED = "task_failed"


def _ensure_mask(value: Any, num_envs: int, default: bool) -> List[bool]:
    if value is None:
        return [default] * num_envs
    try:
        out = [bool(v) for v in list(value)]
    except Exception:
        return [default] * num_envs
    if len(out) != num_envs:
        return [default] * num_envs
    return out


def _ensure_reason_mask(value: Any, num_envs: int) -> List[str]:
    if value is None:
        return [FAILURE_REASON_NONE] * num_envs
    try:
        out = [str(v) if v is not None else FAILURE_REASON_NONE for v in list(value)]
    except Exception:
        return [FAILURE_REASON_NONE] * num_envs
    if len(out) != num_envs:
        return [FAILURE_REASON_NONE] * num_envs
    return out


def get_prev_alive_mask(context: Dict[str, Any], num_envs: int) -> List[bool]:
    """Read the env-alive mask going INTO the current primitive."""
    return _ensure_mask(context.get("env_alive_mask"), num_envs, True)


def init_local_active_mask(context: Dict[str, Any], num_envs: int) -> List[bool]:
    """Local mask tracking which envs are still planning-alive within this primitive."""
    return list(get_prev_alive_mask(context, num_envs))


def merge_segment_into_local_active(local_active: List[bool], segment_successes: List[bool]) -> None:
    """AND segment successes into the local plan-active mask (in-place)."""
    n = len(local_active)
    for i in range(n):
        if i >= len(segment_successes):
            break
        if not segment_successes[i]:
            local_active[i] = False


def mark_local_inactive_with_reason(
    context: Dict[str, Any],
    local_active: List[bool],
    env_idx: int,
    reason: str,
) -> None:
    """Mark a single env inactive locally and record a specific failure reason
    (e.g. ``target_missing``) so the subsequent plan-failure commit does not
    overwrite it with the default ``planning_failed``.
    """
    if env_idx < 0 or env_idx >= len(local_active):
        return
    local_active[env_idx] = False
    reason_mask = _ensure_reason_mask(context.get("step_failure_reason_mask"), len(local_active))
    if reason_mask[env_idx] == FAILURE_REASON_NONE:
        reason_mask[env_idx] = reason
    context["step_failure_reason_mask"] = reason_mask


def commit_plan_failures(
    context: Dict[str, Any],
    prev_alive: List[bool],
    local_active: List[bool],
    default_reason: str = FAILURE_REASON_PLANNING_FAILED,
) -> List[bool]:
    """AND ``local_active`` into ``context.plan_success_mask`` for envs that
    were alive entering this primitive. Records ``default_reason`` for envs
    that died here unless a more specific reason is already set.
    """
    n = len(prev_alive)
    plan_mask = _ensure_mask(context.get("plan_success_mask"), n, True)
    reason_mask = _ensure_reason_mask(context.get("step_failure_reason_mask"), n)
    for i in range(n):
        if not prev_alive[i]:
            continue
        if not local_active[i]:
            plan_mask[i] = False
            if reason_mask[i] == FAILURE_REASON_NONE:
                reason_mask[i] = default_reason
    context["plan_success_mask"] = plan_mask
    context["step_failure_reason_mask"] = reason_mask
    return plan_mask


def commit_task_failures(
    context: Dict[str, Any],
    prev_alive: List[bool],
    plan_active_after: List[bool],
    task_pass: List[bool],
) -> List[bool]:
    """AND ``task_pass`` into ``context.task_success_mask`` for envs that were
    alive entering the primitive AND survived its planning chain.
    """
    n = len(prev_alive)
    task_mask = _ensure_mask(context.get("task_success_mask"), n, True)
    reason_mask = _ensure_reason_mask(context.get("step_failure_reason_mask"), n)
    for i in range(n):
        if not prev_alive[i] or not plan_active_after[i]:
            continue
        if i >= len(task_pass) or not task_pass[i]:
            task_mask[i] = False
            if reason_mask[i] == FAILURE_REASON_NONE:
                reason_mask[i] = FAILURE_REASON_TASK_FAILED
    context["task_success_mask"] = task_mask
    context["step_failure_reason_mask"] = reason_mask
    return task_mask


def compute_env_alive_mask(context: Dict[str, Any], num_envs: int) -> List[bool]:
    """env_alive = plan_success ∧ task_success."""
    plan_mask = _ensure_mask(context.get("plan_success_mask"), num_envs, True)
    task_mask = _ensure_mask(context.get("task_success_mask"), num_envs, True)
    return [bool(p and t) for p, t in zip(plan_mask, task_mask)]


def log_target_missing_envs(
    primitive_name: str,
    target_label: Optional[str],
    target_available_mask: List[bool],
    prev_success_mask: Optional[List[bool]] = None,
):
    """Print a concise warning for envs whose explicit target is missing."""
    if not target_label:
        return
    if prev_success_mask is None:
        active_missing = [not available for available in target_available_mask]
    else:
        active_missing = [
            bool(prev_success_mask[idx]) and not bool(target_available_mask[idx])
            for idx in range(len(target_available_mask))
        ]
    missing_count = sum(active_missing)
    if missing_count > 0:
        print(
            f"  ⚠️ [Primitive] Target '{target_label}' missing for "
            f"{missing_count}/{len(target_available_mask)} envs in {primitive_name}; "
            "those envs will fail with reason=target_missing"
        )


def prepare_primitive_episode(collector, context: Dict[str, Any]) -> bool:
    """Shared episode bootstrap for primitive execution."""
    if context.get("_prepared", False):
        return True

    task_yaml = {}
    if collector.task_config_path:
        with open(collector.task_config_path, "r", encoding="utf-8") as f:
            task_yaml = yaml.safe_load(f) or {}
        context["task_yaml"] = task_yaml
        attach_scene_success_meta(collector.scene, task_yaml)
        tex_cfg = task_yaml.get("texture_randomization", {})
        if hasattr(collector, "clear_texture_randomization_record"):
            collector.clear_texture_randomization_record()
        if tex_cfg.get("enabled", False):
            collector.apply_random_textures(
                texture_dir=tex_cfg["texture_dir"],
                targets=tex_cfg.get("targets", {}),
            )

    action_dim = collector.env.env.action_manager.total_action_dim
    for _ in range(10):
        action = torch.zeros(collector.num_envs, action_dim, device=collector.device)
        action[:, :] = collector.robot.data.joint_pos[: collector.num_envs, :action_dim]
        collector.env.step(action)

    collector._settle_sim_and_update_obstacles()
    init_ee_pos, init_ee_quat = collector.get_current_ee_pose()
    context["initial_ee_pos"] = np.atleast_2d(np.array(init_ee_pos, dtype=float))
    context["initial_ee_quat"] = np.atleast_2d(np.array(init_ee_quat, dtype=float))

    if collector.recorder:
        metadata = {
            "scene_desc": json.dumps(context["scene_desc"]),
            "episode_num": context["episode_num"],
            "grasp_instruction": context["task_desc"],
            "task_config": task_yaml,
        }
        if hasattr(collector, "build_episode_initial_metadata"):
            metadata.update(collector.build_episode_initial_metadata(task_yaml))
        collector.recorder.start_episode(context["episode_num"], metadata)

    context["_prepared"] = True
    return True


def compute_pick_hang_below(
    collector,
    target_label: str,
    all_pick_pos: List[np.ndarray],
    target_height: float,
) -> List[float]:
    """Compute height difference between pick point and object bottom."""
    env_origins = collector._get_env_origins()
    all_hang_below = []
    for env_idx in range(collector.num_envs):
        target_pos_global, _ = collector._get_object_pose_global(target_label, env_idx)
        if target_pos_global is not None:
            target_bottom_z = target_pos_global[2] - target_height / 2.0
            grasp_z = all_pick_pos[env_idx][2] + env_origins[env_idx][2]
            all_hang_below.append(grasp_z - target_bottom_z)
        else:
            all_hang_below.append(target_height / 2.0)
    return all_hang_below


def parse_position_offset(value: Any, default: Any) -> np.ndarray:
    """Normalize scalar/vector config into xyz offset."""
    if value is None:
        value = default
    offset = np.array(value, dtype=float)
    if offset.ndim == 0:
        return np.array([0.0, 0.0, float(offset)], dtype=float)
    if offset.shape == (3,):
        return offset
    raise ValueError(f"Invalid position offset: {value}")


def normalize_primitive_mode(value: Any) -> Any:
    """Handle YAML 1.1 bool coercion for on/off style strings."""
    if isinstance(value, bool):
        return "on" if value else "off"
    return value


def resolve_scene_entry_height(entry: Optional[Dict[str, Any]], default: float) -> float:
    """Safely read height from scene entry."""
    if not entry:
        return float(default)
    raw_height = entry.get("height", default)
    if raw_height is None:
        return float(default)
    try:
        return float(raw_height)
    except (TypeError, ValueError):
        return float(default)


def store_pick_grasp_context(
    context: Dict[str, Any],
    *,
    mode: str,
    target_label: str,
    all_pick_pos: List[np.ndarray],
    all_pick_quat: List[np.ndarray],
    valid_env_indices: List[int],
    target_height: float,
    all_hang_below: List[float],
    target_available_mask: List[bool],
    robot_params: Dict[str, Any],
    pre_pick_offset: Any,
    pick_interpolation_steps: int,
    lift_interpolation_steps: int,
) -> bool:
    """Store grasp computation results into primitive context."""
    if not valid_env_indices:
        if any(target_available_mask):
            print("  ❌ No valid grasp poses found.")
            return False

    all_pick_pos_np = np.array(all_pick_pos)
    all_pick_quat_np = np.array(all_pick_quat)

    context["grasp_mode"] = mode
    context["pick_target_label"] = target_label
    context["grasp_pose"] = {
        "mode": mode,
        "target_label": target_label,
        "positions": all_pick_pos_np,
        "orientations": all_pick_quat_np,
        "target_available_mask": list(target_available_mask),
        "valid_env_indices": list(valid_env_indices),
    }
    context["all_pick_pos"] = all_pick_pos_np
    context["all_pick_quat"] = all_pick_quat_np
    context["valid_env_indices"] = list(valid_env_indices)
    context["target_height"] = target_height
    context["all_hang_below"] = list(all_hang_below)
    context["pick_robot_params"] = dict(robot_params)
    context["pick_pre_pick_offset"] = np.array(pre_pick_offset, dtype=float)
    context["pick_interpolation_steps"] = int(pick_interpolation_steps)
    context["lift_interpolation_steps"] = int(lift_interpolation_steps)
    context["held_object"] = target_label
    context["selected_grasp_centers_local"] = [None] * len(all_pick_pos_np)
    return True


def get_selected_multidir_grasp_center_locals(collector, context: Dict[str, Any]) -> List[Optional[np.ndarray]]:
    """Return selected local grasp centers from multidir selection info."""
    selection_infos = list(context.get("affordance_selection_info", []))
    grasp_centers_local: List[Optional[np.ndarray]] = []
    for env_idx in range(collector.num_envs):
        info = selection_infos[env_idx] if env_idx < len(selection_infos) else None
        best_local = None
        if info and info.get("candidates"):
            best_candidate = info["candidates"][0]
            grasp_cfg = best_candidate.get("grasp", {})
            if grasp_cfg.get("grasp_center") is not None:
                best_local = np.array(grasp_cfg["grasp_center"], dtype=np.float64)
        grasp_centers_local.append(best_local)
    return grasp_centers_local


def build_per_env_grasp_centers_local(
    num_envs: int,
    grasp_center_local: np.ndarray,
    valid_env_indices: List[int],
) -> List[Optional[np.ndarray]]:
    """Broadcast a single selected local grasp center to valid environments only."""
    per_env_centers: List[Optional[np.ndarray]] = [None] * num_envs
    grasp_center_local = np.asarray(grasp_center_local, dtype=np.float64)
    for env_idx in valid_env_indices:
        per_env_centers[env_idx] = grasp_center_local.copy()
    return per_env_centers


def load_affordance_grasp(npz_path: str) -> dict:
    """Load single-direction affordance grasp result."""
    data = np.load(npz_path)
    return {
        "grasp_center": data["grasp_center"],
        "approach_dir": data["approach_dir"],
        "grip_axis": data["grip_axis"],
        "binormal": data["binormal"],
        "rotation_matrix": data["rotation_matrix"],
        "score": float(data["score"]),
    }


def transform_grasp_to_world(
    grasp_center_local: np.ndarray,
    rotation_matrix_local: np.ndarray,
    obj_pos_world: np.ndarray,
    obj_quat_world: np.ndarray,
    env_origin: np.ndarray,
    reference_quat_wxyz: np.ndarray = None,
    object_scale: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform local grasp pose into robot frame relative to env origin."""
    if object_scale is None:
        object_scale = np.ones(3)

    rot_obj = Rotation.from_quat(
        [obj_quat_world[1], obj_quat_world[2], obj_quat_world[3], obj_quat_world[0]]
    ).as_matrix()

    grasp_center_world = rot_obj @ (grasp_center_local * object_scale) + obj_pos_world
    rot_grasp_world = rot_obj @ rotation_matrix_local

    approach_w = rot_grasp_world[:, 2]
    approach_w = approach_w / np.linalg.norm(approach_w)
    grip_w = rot_grasp_world[:, 1]
    grip_w = grip_w - np.dot(grip_w, approach_w) * approach_w
    grip_w = grip_w / np.linalg.norm(grip_w)
    binormal_w = np.cross(grip_w, approach_w)
    binormal_w = binormal_w / np.linalg.norm(binormal_w)
    rot_grasp_world = np.column_stack([binormal_w, grip_w, approach_w])

    grasp_pos_robot = grasp_center_world - env_origin
    quat_xyzw = Rotation.from_matrix(rot_grasp_world).as_quat()
    grasp_quat_robot = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    grasp_quat_robot = normalize_ee_orientation_wxyz(grasp_quat_robot, reference_quat_wxyz)
    return grasp_pos_robot, grasp_quat_robot


def load_affordance_grasp_multidir(npz_path: str) -> List[dict]:
    """Load multi-direction affordance grasp results."""
    data = np.load(npz_path)
    n_directions = int(data["n_directions"]) if "n_directions" in data else 4

    grasps = []
    for i in range(n_directions):
        prefix = f"q{i}_"
        key = prefix + "grasp_center"
        if key not in data:
            continue
        grasps.append(
            {
                "grasp_center": data[prefix + "grasp_center"],
                "approach_dir": data[prefix + "approach_dir"],
                "grip_axis": data[prefix + "grip_axis"],
                "binormal": data[prefix + "binormal"],
                "rotation_matrix": data[prefix + "rotation_matrix"],
                "score": float(data[prefix + "score"]),
                "azimuth": float(data[prefix + "azimuth"]),
                "ee_position": data[prefix + "ee_position"],
                "direction_index": i,
            }
        )

    print(f"  [INFO] Loaded {len(grasps)} grasp directions from {npz_path}")
    return grasps


DIRECTION_NAMES = ["Q0(0-90°)", "Q1(90-180°)", "Q2(180-270°)", "Q3(270-360°)", "Vertical(Z-)"]


def rotate_quat_wxyz_about_z(quat_wxyz: np.ndarray, angle_deg: float = 180.0) -> np.ndarray:
    rot = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    rot_z = Rotation.from_euler("z", angle_deg, degrees=True)
    rotated = rot * rot_z
    quat_xyzw = rotated.as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def select_best_grasp_direction(
    grasps: List[dict],
    obj_pos_world: np.ndarray,
    obj_quat_world: np.ndarray,
    env_origin: np.ndarray,
    ee_pos: np.ndarray,
    ee_quat: np.ndarray,
    motion_planner,
    current_joint_pos: np.ndarray,
    current_joint_vel: np.ndarray,
    joint_names: List[str],
    env_idx: int = 0,
    affordance_weight: float = 0.05,
    reachability_weight: float = 0.35,
    collision_weight: float = 0.65,
) -> Tuple[Optional[dict], Dict]:
    """Score candidate multi-direction grasps and select the best one."""
    candidates = []

    for grasp in grasps:
        dir_idx = grasp["direction_index"]
        dir_name = DIRECTION_NAMES[dir_idx] if dir_idx < len(DIRECTION_NAMES) else f"Dir{dir_idx}"

        grasp_pos, grasp_quat = transform_grasp_to_world(
            grasp_center_local=grasp["grasp_center"],
            rotation_matrix_local=grasp["rotation_matrix"],
            obj_pos_world=obj_pos_world,
            obj_quat_world=obj_quat_world,
            env_origin=env_origin,
            reference_quat_wxyz=ee_quat,
        )

        pos_dist = np.linalg.norm(grasp_pos - ee_pos)
        q1 = np.array([ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]])
        q2 = np.array([grasp_quat[1], grasp_quat[2], grasp_quat[3], grasp_quat[0]])
        if np.dot(q1, q2) < 0:
            q2 = -q2
        dot = np.clip(abs(np.dot(q1, q2)), 0.0, 1.0)
        angle_diff = 2.0 * np.arccos(dot)

        dist_score = max(0.0, 1.0 - pos_dist / 0.5)
        angle_score = max(0.0, 1.0 - angle_diff / np.pi)
        reachability_score = dist_score * 0.6 + angle_score * 0.4

        pre_pick_pos = grasp_pos.copy()
        pre_pick_pos[2] += 0.05
        pre_pick_pos[0] -= 0.05

        collision_score = 0.0
        plan_success = False
        if motion_planner is not None:
            try:
                _, success = motion_planner.plan_to_pose(
                    current_joint_pos=current_joint_pos,
                    current_joint_vel=current_joint_vel,
                    joint_names=joint_names,
                    target_pos=pre_pick_pos,
                    target_quat=grasp_quat,
                    max_attempts=2,
                )
                plan_success = success
                collision_score = 1.0 if success else 0.0
            except Exception as exc:
                print(f"    [WARN] Planning failed for {dir_name}: {exc}")

        total_score = (
            affordance_weight * grasp["score"]
            + reachability_weight * reachability_score
            + collision_weight * collision_score
        )

        candidates.append(
            {
                "direction_index": dir_idx,
                "direction_name": dir_name,
                "grasp": grasp,
                "grasp_pos": grasp_pos,
                "grasp_quat": grasp_quat,
                "affordance_score": float(grasp["score"]),
                "reachability_score": float(reachability_score),
                "collision_score": float(collision_score),
                "plan_success": bool(plan_success),
                "pos_dist": float(pos_dist),
                "angle_diff_deg": float(np.degrees(angle_diff)),
                "total_score": float(total_score),
            }
        )

    candidates.sort(key=lambda item: item["total_score"], reverse=True)
    best_candidate = candidates[0] if candidates else None

    selection_info = {
        "env_idx": env_idx,
        "candidates": candidates,
        "best_direction_index": best_candidate["direction_index"] if best_candidate else None,
        "best_direction_name": best_candidate["direction_name"] if best_candidate else None,
        "forced_vertical": False,
        "weights": {
            "affordance": affordance_weight,
            "reachability": reachability_weight,
            "collision": collision_weight,
        },
    }
    return best_candidate, selection_info


def load_affordance_centroid(npz_path: str) -> np.ndarray:
    data = np.load(npz_path)
    if "centroids" not in data:
        raise KeyError(f"No 'centroids' found in {npz_path}")
    centroids = np.asarray(data["centroids"], dtype=np.float64)
    if centroids.ndim != 2 or len(centroids) == 0:
        raise ValueError(f"'centroids' in {npz_path} must have shape (N, 3) with N >= 1")
    return centroids[0].copy()


def compute_pour_position(
    container_center_world: np.ndarray,
) -> np.ndarray:
    return np.array(container_center_world, dtype=np.float64).copy()


def compute_pour_quaternion(
    current_quat_wxyz: np.ndarray,
    cup_pos: np.ndarray,
    container_pos: np.ndarray,
    tilt_angle_deg: float = 90.0,
) -> np.ndarray:
    delta = container_pos.copy() - cup_pos.copy()
    delta[2] = 0
    dist = np.linalg.norm(delta)

    if dist < 1e-6:
        pour_dir = np.array([1.0, 0.0, 0.0])
    else:
        pour_dir = delta / dist

    tilt_axis = np.array([-pour_dir[1], pour_dir[0], 0.0])
    tilt_axis = tilt_axis / np.linalg.norm(tilt_axis)

    q = current_quat_wxyz
    current_rot = Rotation.from_quat([q[1], q[2], q[3], q[0]])
    tilt_rot = Rotation.from_rotvec(np.radians(tilt_angle_deg) * tilt_axis)
    final_rot = tilt_rot * current_rot

    quat_xyzw = final_rot.as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def load_affordance_clusters(npz_path: str) -> np.ndarray:
    data = np.load(npz_path)
    if "centroids" not in data:
        raise KeyError(f"No 'centroids' found in {npz_path}")
    return np.asarray(data["centroids"], dtype=np.float64).copy()


def select_pour_cluster(
    cluster_centroids_local: np.ndarray,
    basket_pos_world: np.ndarray,
    basket_quat_world: np.ndarray,
    container_pos_world: np.ndarray,
) -> Tuple[np.ndarray, int]:
    rot_obj = Rotation.from_quat(
        [basket_quat_world[1], basket_quat_world[2], basket_quat_world[3], basket_quat_world[0]]
    ).as_matrix()

    delta = container_pos_world[:2] - basket_pos_world[:2]
    dist = np.linalg.norm(delta)
    pour_dir_xy = np.array([1.0, 0.0]) if dist < 1e-6 else delta / dist

    best_idx = 0
    best_proj = -np.inf
    for i, center_local in enumerate(cluster_centroids_local):
        center_world = rot_obj @ center_local + basket_pos_world
        offset_xy = center_world[:2] - basket_pos_world[:2]
        proj = np.dot(offset_xy, pour_dir_xy)
        if proj > best_proj:
            best_proj = proj
            best_idx = i

    return cluster_centroids_local[best_idx].copy(), best_idx


_POUR_GEOMETRY_CACHE: dict[tuple[str, float], dict[str, Any]] = {}


def _load_labeled_points_for_pour(npz_path: str) -> np.ndarray:
    base_dir = Path(npz_path).resolve().parent
    pt_path = base_dir / "labeled_result.pt"
    npz_labeled_path = base_dir / "labeled_result.npz"

    if pt_path.exists():
        data = torch.load(pt_path, map_location="cpu")
        points = data["points"]
        return points.numpy() if hasattr(points, "numpy") else np.asarray(points, dtype=np.float64)

    if npz_labeled_path.exists():
        data = np.load(npz_labeled_path)
        return data["points"].astype(np.float64)

    raise FileNotFoundError(f"labeled_result.pt/.npz not found: {base_dir}")


def _normalize_xy(vec: np.ndarray) -> tuple[np.ndarray, float]:
    vec_xy = np.asarray(vec[:2], dtype=np.float64)
    norm = float(np.linalg.norm(vec_xy))
    if norm < 1e-8:
        return np.zeros(2, dtype=np.float64), 0.0
    return vec_xy / norm, norm




def compute_tilt_quaternion_from_pour_direction(
    current_quat_wxyz: np.ndarray,
    pour_direction_world: np.ndarray,
    tilt_angle_deg: float,
) -> np.ndarray:
    dir_xy = np.asarray(pour_direction_world[:2], dtype=np.float64)
    norm = float(np.linalg.norm(dir_xy))
    dir_xy = np.array([1.0, 0.0], dtype=np.float64) if norm < 1e-6 else dir_xy / norm
    tilt_axis = np.array([-dir_xy[1], dir_xy[0], 0.0], dtype=np.float64)

    current_rot = Rotation.from_quat(_quat_wxyz_to_xyzw(current_quat_wxyz))
    tilt_rot = Rotation.from_rotvec(np.radians(tilt_angle_deg) * tilt_axis)
    final_rot = tilt_rot * current_rot
    return _quat_xyzw_to_wxyz(final_rot.as_quat())


def prepare_pour_geometry(npz_path: str, directional_threshold: float = 0.35) -> dict[str, Any]:
    cache_key = (str(Path(npz_path).resolve()), float(directional_threshold))
    cached = _POUR_GEOMETRY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    centroids = load_affordance_clusters(npz_path)
    all_points = _load_labeled_points_for_pour(npz_path)

    aff_data = np.load(npz_path)
    aff_points = aff_data["aff_points"].astype(np.float64).copy()
    default_pour_centroid = np.asarray(centroids[0], dtype=np.float64)

    object_center_xy = np.median(all_points[:, :2], axis=0)
    object_center_z = float(np.median(all_points[:, 2]))
    object_center_local = np.array([object_center_xy[0], object_center_xy[1], object_center_z], dtype=np.float64)

    if len(aff_points) > 0:
        rim_radii = np.linalg.norm(aff_points[:, :2] - object_center_xy, axis=1)
        opening_radius = float(np.median(rim_radii))
        rim_height = float(np.median(aff_points[:, 2]))
    else:
        opening_radius = float(np.linalg.norm(default_pour_centroid[:2] - object_center_xy))
        rim_height = float(default_pour_centroid[2])

    if opening_radius < 1e-4:
        spread_xy = np.ptp(all_points[:, :2], axis=0)
        opening_radius = max(float(np.max(spread_xy) * 0.25), 0.01)

    result = {
        "default_pour_centroid": default_pour_centroid,
        "aff_points": aff_points,
        "object_center_local": object_center_local,
        "opening_radius": opening_radius,
        "rim_height": rim_height,
        "directional_threshold": float(directional_threshold),
    }
    _POUR_GEOMETRY_CACHE[cache_key] = result
    return result


def select_effective_pour_point_local(
    geometry: dict[str, Any],
    pour_centroid_local: np.ndarray,
    grasp_center_local: np.ndarray,
    ee_z_axis_local: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    center_xy = geometry["object_center_local"][:2]
    pour_centroid_local = np.asarray(pour_centroid_local, dtype=np.float64)
    offset_xy = pour_centroid_local[:2] - center_xy
    offset_xy_norm = float(np.linalg.norm(offset_xy))
    offset_ratio = offset_xy_norm / max(float(geometry["opening_radius"]), 1e-6)

    if offset_ratio >= float(geometry["directional_threshold"]):
        return pour_centroid_local.copy(), {
            "mode": "explicit",
            "offset_ratio": offset_ratio,
        }

    radius = geometry["opening_radius"]

    if ee_z_axis_local is not None:
        z_xy = np.asarray(ee_z_axis_local[:2], dtype=np.float64)
        z_xy_norm = float(np.linalg.norm(z_xy))
        if z_xy_norm > 1e-4:
            direction_xy = z_xy / z_xy_norm
            angle_deg = float(np.degrees(np.arctan2(direction_xy[1], direction_xy[0])))
            pour_center_local = np.array(
                [
                    center_xy[0] + direction_xy[0] * radius,
                    center_xy[1] + direction_xy[1] * radius,
                    geometry["rim_height"],
                ],
                dtype=np.float64,
            )
            return pour_center_local, {
                "mode": "ee_z_rim",
                "angle_deg": angle_deg,
                "offset_ratio": offset_ratio,
            }

    grasp_xy = np.asarray(grasp_center_local[:2], dtype=np.float64)
    grasp_dir = grasp_xy - center_xy
    grasp_dir_norm = float(np.linalg.norm(grasp_dir))
    direction_xy = -grasp_dir / grasp_dir_norm if grasp_dir_norm > 1e-4 else np.array([1.0, 0.0], dtype=np.float64)

    pour_center_local = np.array(
        [
            center_xy[0] + direction_xy[0] * radius,
            center_xy[1] + direction_xy[1] * radius,
            geometry["rim_height"],
        ],
        dtype=np.float64,
    )
    angle_deg = float(np.degrees(np.arctan2(direction_xy[1], direction_xy[0])))
    return pour_center_local, {
        "mode": "opposite_grasp",
        "angle_deg": angle_deg,
        "offset_ratio": offset_ratio,
    }


def search_best_pour_rotation(
    container_center_world: np.ndarray,
    effective_pour_world: np.ndarray,
    current_ee_world: np.ndarray,
    object_center_world: np.ndarray,
    env_origin: np.ndarray,
    lift_pos_robot: np.ndarray,
    current_grasp_quat_wxyz: np.ndarray,
    n_candidates: int = 36,
    tilt_angle_deg: float = 90.0,
    tilt_lift_height: float = 0.08,
    tilt_horizontal_offset: float = 0.05,
    hard_constraint_min_z_axis_x: float = 0.0,
    hard_constraint_max_z_axis_z: float = 0.0,
    soft_bonus_z_axis_x_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, dict[str, float]]:
    pour_target_world = np.array(
        [
            container_center_world[0],
            container_center_world[1],
            container_center_world[2],
        ],
        dtype=np.float64,
    )

    current_ee_world = np.asarray(current_ee_world, dtype=np.float64)
    current_rot = Rotation.from_quat(_quat_wxyz_to_xyzw(current_grasp_quat_wxyz))
    pour_offset_in_ee = current_rot.inv().apply(effective_pour_world - current_ee_world)
    obj_to_pour_in_ee = current_rot.inv().apply(effective_pour_world - object_center_world)
    lift_xy = np.asarray(lift_pos_robot[:2], dtype=np.float64)

    best_score = -np.inf
    best_ee_robot = None
    best_ee_quat = None
    best_pour_dir = None
    best_breakdown = None

    current_x_axis_world = current_rot.apply(np.array([1.0, 0.0, 0.0], dtype=np.float64))
    x_axis_xy, x_axis_xy_norm = _normalize_xy(current_x_axis_world)
    if x_axis_xy_norm <= 1e-6:
        base_theta_candidates = [0.0, np.pi]
    else:
        base_heading = float(np.arctan2(x_axis_xy[1], x_axis_xy[0]))
        base_theta_candidates = [-base_heading, np.pi - base_heading]

    chosen_base_theta = None
    chosen_base_x = None
    for theta_base in base_theta_candidates:
        base_rot = Rotation.from_rotvec([0.0, 0.0, theta_base]) * current_rot
        ee_world_base = pour_target_world - base_rot.apply(pour_offset_in_ee)
        if ee_world_base[0] < container_center_world[0]:
            chosen_base_theta = theta_base
            chosen_base_x = float(ee_world_base[0])
            break

    if chosen_base_theta is None:
        base_candidates_with_x = []
        for theta_base in base_theta_candidates:
            base_rot = Rotation.from_rotvec([0.0, 0.0, theta_base]) * current_rot
            ee_world_base = pour_target_world - base_rot.apply(pour_offset_in_ee)
            base_candidates_with_x.append((theta_base, float(ee_world_base[0])))
        chosen_base_theta, chosen_base_x = min(base_candidates_with_x, key=lambda item: item[1])

    candidate_offsets_deg = np.array([-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0], dtype=np.float64)

    for offset_deg in candidate_offsets_deg:
        theta = chosen_base_theta + float(np.radians(offset_deg))
        z_rot = Rotation.from_rotvec([0.0, 0.0, theta])
        new_rot = z_rot * current_rot
        new_quat_wxyz = _quat_xyzw_to_wxyz(new_rot.as_quat())
        ee_world = pour_target_world - new_rot.apply(pour_offset_in_ee)
        ee_robot = ee_world - env_origin
        gripper_z_axis_robot = new_rot.apply(np.array([0.0, 0.0, 1.0], dtype=np.float64))
        gripper_x_axis_world = new_rot.apply(np.array([1.0, 0.0, 0.0], dtype=np.float64))

        # Hard constraint: gripper local z-axis should point forward and downward in robot frame.
        if gripper_z_axis_robot[0] <= hard_constraint_min_z_axis_x or gripper_z_axis_robot[2] >= hard_constraint_max_z_axis_z:
            continue
        if ee_world[0] >= container_center_world[0]:
            continue

        pour_dir_rotated = new_rot.apply(obj_to_pour_in_ee)

        tilt_pos = ee_robot.copy()
        tilt_pos[2] += tilt_lift_height
        pour_dir_xy, pour_dir_norm = _normalize_xy(pour_dir_rotated)
        if pour_dir_norm > 1e-6:
            tilt_pos[:2] += pour_dir_xy * tilt_horizontal_offset

        container_robot = np.asarray(container_center_world - env_origin, dtype=np.float64)
        move_dist = float(np.linalg.norm(ee_robot - lift_pos_robot))
        side_penalty = abs(float(ee_robot[1] - container_robot[1]))
        delta_xy = float(np.linalg.norm(ee_robot[:2] - lift_xy))
        rotation_abs = min(float(theta), float(2.0 * np.pi - theta))
        rotation_penalty = rotation_abs / np.pi
        tilt_side_penalty = abs(float(tilt_pos[1] - container_robot[1]))

        score = (
            -0.40 * move_dist
            - 0.40 * side_penalty
            - 1.20 * delta_xy
            - 1.00 * rotation_penalty
            - 0.60 * tilt_side_penalty
            + soft_bonus_z_axis_x_weight * float(gripper_z_axis_robot[0])
        )

        breakdown = {
            "theta_deg": float(np.degrees(theta)),
            "base_theta_deg": float(np.degrees(chosen_base_theta)),
            "base_ee_x": float(chosen_base_x),
            "offset_deg": float(offset_deg),
            "move_dist": move_dist,
            "side_penalty": side_penalty,
            "delta_xy": delta_xy,
            "rotation_penalty": rotation_penalty,
            "tilt_side_penalty": tilt_side_penalty,
            "gripper_z_axis_x": float(gripper_z_axis_robot[0]),
            "gripper_z_axis_z": float(gripper_z_axis_robot[2]),
            "gripper_x_axis_world": gripper_x_axis_world.tolist(),
            "tilt_pos": tilt_pos.tolist(),
        }

        if score > best_score:
            best_score = score
            best_ee_robot = ee_robot.copy()
            best_ee_quat = new_quat_wxyz.copy()
            best_pour_dir = pour_dir_rotated.copy()
            best_breakdown = breakdown

    if best_ee_robot is None:
        raise RuntimeError(
            "No pour rotation candidate satisfies hard constraint: "
            f"z_axis_x > {hard_constraint_min_z_axis_x}, z_axis_z < {hard_constraint_max_z_axis_z}"
        )

    return pour_target_world, best_ee_robot, best_ee_quat, best_pour_dir, best_score, best_breakdown


def _print_multidir_score_table(env_idx: int, selection_info: Dict[str, Any], top_k: int = 5) -> None:
    """Pretty-print multi-direction grasp scoring details."""
    candidates = list(selection_info.get("candidates", []))
    weights = selection_info.get("weights", {})
    if not candidates:
        print(f"  [ScoreTable][env {env_idx}] No candidates.")
        return

    top_k = max(1, min(int(top_k), len(candidates)))
    print(
        f"  [ScoreTable][env {env_idx}] "
        f"weights(a/r/c)=({weights.get('affordance', 'N/A')}/{weights.get('reachability', 'N/A')}/{weights.get('collision', 'N/A')})"
    )
    print("    rank | dir | aff | reach | coll | pos_dist(m) | angle(deg) | plan | total")
    for rank, cand in enumerate(candidates[:top_k], start=1):
        print(
            "    "
            f"{rank:>4} | "
            f"{cand.get('direction_name', 'N/A'):<10} | "
            f"{cand.get('affordance_score', 0.0):>4.3f} | "
            f"{cand.get('reachability_score', 0.0):>5.3f} | "
            f"{cand.get('collision_score', 0.0):>4.1f} | "
            f"{cand.get('pos_dist', 0.0):>10.4f} | "
            f"{cand.get('angle_diff_deg', 0.0):>10.2f} | "
            f"{'Y' if cand.get('plan_success', False) else 'N':>4} | "
            f"{cand.get('total_score', 0.0):>5.3f}"
        )


def primitive_compute_grasp_pose(collector, mode: Optional[str], params: Dict[str, Any], context: Dict[str, Any]) -> bool:
    target_label = params.get("target") or context.get("held_object")
    if not target_label:
        print("[ERROR] Primitive 'compute_grasp_pose' requires a target object.")
        return False
    collector.target_prim_path = f"/World/envs/env_0/{target_label}"

    target_entry = collector._get_object_entry_from_scene_desc(target_label, context["scene_desc"])
    target_height = resolve_scene_entry_height(target_entry, 0.05)

    if mode == "gt":
        print("  📐 [Primitive] Computing grasp pose from GT pose...")
        grasp_pos_offset, grasp_quat_default = collector._load_grasp_config()
        has_explicit_grasp_quat = "grasp_quat" in params
        if has_explicit_grasp_quat:
            grasp_quat_default = np.array(params["grasp_quat"], dtype=float)
            grasp_quat_default /= np.linalg.norm(grasp_quat_default)

        env_origins = collector._get_env_origins()
        all_pick_pos = []
        all_pick_quat = []
        valid_env_indices = []
        target_available_mask = []
        _, ee_quat_w = collector.get_current_ee_pose()

        for env_idx in range(collector.num_envs):
            target_pos_global, target_quat_global = collector._get_object_pose_global(target_label, env_idx)
            env_ee_quat = ee_quat_w[env_idx] if ee_quat_w.ndim > 1 else ee_quat_w
            if target_pos_global is None:
                target_available_mask.append(False)
                print(f"  ❌ Env {env_idx}: object '{target_label}' not found in scene.")
                all_pick_pos.append(np.array([0.3, 0.0, 0.2]))
                all_pick_quat.append(normalize_ee_orientation_wxyz(grasp_quat_default.copy(), env_ee_quat))
                continue
            target_available_mask.append(True)

            pick_pos = target_pos_global - env_origins[env_idx] + grasp_pos_offset
            if has_explicit_grasp_quat:
                aligned_quat = normalize_ee_orientation_wxyz(grasp_quat_default.copy(), env_ee_quat)
            else:
                aligned_quat = collector._compute_aligned_grasp_quat(
                    obj_quat_wxyz=target_quat_global,
                    base_grasp_quat_wxyz=grasp_quat_default,
                )
                aligned_quat = normalize_ee_orientation_wxyz(aligned_quat, env_ee_quat)

            all_pick_pos.append(pick_pos)
            all_pick_quat.append(aligned_quat)
            valid_env_indices.append(env_idx)
            print(f"  ✅ Env {env_idx}: GT pick pos={pick_pos}, quat={aligned_quat}")

        all_hang_below = compute_pick_hang_below(collector, target_label, all_pick_pos, target_height)
        print(f"  📏 Pre-computed hang_below: {all_hang_below}")

        return store_pick_grasp_context(
            context,
            mode=mode,
            target_label=target_label,
            all_pick_pos=all_pick_pos,
            all_pick_quat=all_pick_quat,
            valid_env_indices=valid_env_indices,
            target_height=target_height,
            all_hang_below=all_hang_below,
            target_available_mask=target_available_mask,
            robot_params=collector._get_pick_pos_robot_params(),
            pre_pick_offset=params.get("pre_pick_offset", 0.1),
            pick_interpolation_steps=int(params.get("pick_interpolation_steps", 10)),
            lift_interpolation_steps=int(params.get("lift_interpolation_steps", 10)),
        )

    if mode == "anygrasp":
        print("  👁️ [Primitive] Executing AnyGrasp perception pipeline...")
        target_available_mask = compute_target_availability_mask(collector, target_label)
        log_target_missing_envs("pick(anygrasp)", target_label, target_available_mask)
        if not collector.capture_camera_data_batch():
            return False
        if not collector.generate_bbox_batch(target_label):
            return False
        if not collector.vision_pipeline.run_anygrasp(collector.num_envs):
            return False

        _, ee_quat_w = collector.get_current_ee_pose()
        all_pick_pos, all_pick_quat, valid_env_indices = collector.vision_pipeline.parse_grasp_results(
            collector.num_envs, collector.device, ee_quat_w=ee_quat_w
        )
        all_hang_below = compute_pick_hang_below(collector, target_label, all_pick_pos, target_height)
        print(f"  📏 Pre-computed hang_below: {all_hang_below}")

        return store_pick_grasp_context(
            context,
            mode=mode,
            target_label=target_label,
            all_pick_pos=all_pick_pos,
            all_pick_quat=all_pick_quat,
            valid_env_indices=valid_env_indices,
            target_height=target_height,
            all_hang_below=all_hang_below,
            target_available_mask=target_available_mask,
            robot_params=collector._get_pick_robot_params(),
            pre_pick_offset=params.get("pre_pick_offset", 0.1),
            pick_interpolation_steps=int(params.get("pick_interpolation_steps", 5)),
            lift_interpolation_steps=int(params.get("lift_interpolation_steps", 5)),
        )

    if mode in ("affordance", "hang", "affordance_open", "affordance_pull"):
        if mode == "affordance_open":
            resolve_action = "open"
        elif mode == "affordance_pull":
            resolve_action = "pull"
        elif mode == "hang":
            resolve_action = "pick_hang"
        else:
            resolve_action = "pick"
        affordance_path = (
            params.get("affordance_npz_path")
            or resolve_affordance_path(target_label, resolve_action)
        )
        if not affordance_path:
            print(
                f"[ERROR] Primitive 'compute_grasp_pose({mode})' could not resolve affordance data. "
                "Set params.affordance_npz_path or provide the convention file under "
                f"pc_dataset_generate/pc/{{object}}_{'pick' if resolve_action == 'pick_hang' else resolve_action}/."
            )
            return False

        print(f"  🤏 [Primitive] Loading affordance grasp from: {affordance_path}")
        try:
            aff_grasp = load_affordance_grasp(affordance_path)
        except Exception as exc:
            print(f"  [ERROR] Failed to load affordance data: {exc}")
            return False

        env_origins = collector._get_env_origins()
        all_pick_pos = []
        all_pick_quat = []
        valid_env_indices = []
        target_available_mask = []
        _, ee_quat_w = collector.get_current_ee_pose()

        object_config = None
        if collector.task_config:
            object_config = collector.task_config.get_object_by_name(target_label)
            if object_config is None:
                object_config = collector.task_config.articulations.get(target_label)
        object_scale = np.array([1.0, 1.0, 1.0], dtype=float)
        if object_config is not None and getattr(object_config, "scale", None) is not None:
            object_scale = np.array(object_config.scale, dtype=float)

        for env_idx in range(collector.num_envs):
            obj_pos, obj_quat = collector._get_object_pose_global(target_label, env_idx)
            if obj_pos is None:
                target_available_mask.append(False)
                print(f"  ❌ Env {env_idx}: target '{target_label}' not found in scene.")
                all_pick_pos.append(np.array([0.5, 0.0, 0.5]))
                all_pick_quat.append(np.array([0.0, 1.0, 0.0, 0.0]))
                continue
            target_available_mask.append(True)

            env_ee_quat = ee_quat_w[env_idx] if ee_quat_w.ndim > 1 else ee_quat_w
            grasp_pos, grasp_quat = transform_grasp_to_world(
                grasp_center_local=aff_grasp["grasp_center"],
                rotation_matrix_local=aff_grasp["rotation_matrix"],
                obj_pos_world=obj_pos,
                obj_quat_world=obj_quat,
                env_origin=env_origins[env_idx],
                reference_quat_wxyz=env_ee_quat,
                object_scale=object_scale,
            )

            all_pick_pos.append(grasp_pos)
            all_pick_quat.append(grasp_quat)
            valid_env_indices.append(env_idx)
            print(f"  ✅ Env {env_idx}: affordance pick pos={grasp_pos}, quat={grasp_quat}")

        all_hang_below = compute_pick_hang_below(collector, target_label, all_pick_pos, target_height)
        print(f"  📏 Pre-computed hang_below: {all_hang_below}")

        result = store_pick_grasp_context(
            context,
            mode=mode,
            target_label=target_label,
            all_pick_pos=all_pick_pos,
            all_pick_quat=all_pick_quat,
            valid_env_indices=valid_env_indices,
            target_height=target_height,
            all_hang_below=all_hang_below,
            target_available_mask=target_available_mask,
            robot_params=collector._get_pick_robot_params(),
            pre_pick_offset=params.get("pre_pick_offset", 0.05),
            pick_interpolation_steps=int(params.get("pick_interpolation_steps", 5)),
            lift_interpolation_steps=int(params.get("lift_interpolation_steps", 5)),
        )
        if result:
            context["pick_mode"] = "pick_hang" if mode == "hang" else "pick"
            context["selected_grasp_centers_local"] = build_per_env_grasp_centers_local(
                collector.num_envs,
                aff_grasp["grasp_center"],
                valid_env_indices,
            )
        return result

    if mode == "affordance_multidir":
        affordance_path = (
            params.get("affordance_npz_path")
            or resolve_affordance_path(target_label, "pick_4dir")
        )
        if not affordance_path:
            print(
                "[ERROR] Primitive 'compute_grasp_pose(affordance_multidir)' could not resolve affordance data. "
                "Set params.affordance_npz_path or provide "
                "pc_dataset_generate/pc/{object}_pick/grasp_result_4dir.npz."
            )
            return False

        affordance_weight = 0.05
        reachability_weight = 0.35
        collision_weight = 0.65

        print(f"  🤏 [Primitive] Loading multi-direction affordance grasps from: {affordance_path}")
        try:
            all_grasps = load_affordance_grasp_multidir(affordance_path)
        except Exception as exc:
            print(f"  [ERROR] Failed to load multidir affordance data: {exc}")
            return False

        if not all_grasps:
            print("  [ERROR] No valid multi-direction grasp candidates found.")
            return False

        ee_pos, ee_quat = collector.get_current_ee_pose()
        env_origins = collector._get_env_origins()
        all_pick_pos = []
        all_pick_quat = []
        valid_env_indices = []
        target_available_mask = []
        all_selection_info = []

        print_score_table = True
        score_table_top_k = 5

        for env_idx in range(collector.num_envs):
            obj_pos, obj_quat = collector._get_object_pose_global(target_label, env_idx)
            if obj_pos is None:
                target_available_mask.append(False)
                print(f"  ❌ Env {env_idx}: target '{target_label}' not found in scene.")
                all_pick_pos.append(np.array([0.5, 0.0, 0.5]))
                all_pick_quat.append(np.array([0.0, 1.0, 0.0, 0.0]))
                all_selection_info.append(None)
                continue
            target_available_mask.append(True)

            env_ee_pos = ee_pos[env_idx] if ee_pos.ndim > 1 else ee_pos
            env_ee_quat = ee_quat[env_idx] if ee_quat.ndim > 1 else ee_quat

            collector._update_obstacles_for_env(env_idx)
            current_joint_pos = collector.robot.data.joint_pos[env_idx].cpu().numpy()
            current_joint_vel = collector.robot.data.joint_vel[env_idx].cpu().numpy()

            best_candidate, selection_info = select_best_grasp_direction(
                grasps=all_grasps,
                obj_pos_world=obj_pos,
                obj_quat_world=obj_quat,
                env_origin=env_origins[env_idx],
                ee_pos=env_ee_pos,
                ee_quat=env_ee_quat,
                motion_planner=collector.motion_planner,
                current_joint_pos=current_joint_pos,
                current_joint_vel=current_joint_vel,
                joint_names=collector.robot.joint_names,
                env_idx=env_idx,
                affordance_weight=affordance_weight,
                reachability_weight=reachability_weight,
                collision_weight=collision_weight,
            )
            all_selection_info.append(selection_info)
            if print_score_table:
                _print_multidir_score_table(env_idx, selection_info, top_k=score_table_top_k)

            if best_candidate is None:
                all_pick_pos.append(np.array([0.5, 0.0, 0.5]))
                all_pick_quat.append(np.array([0.0, 1.0, 0.0, 0.0]))
                continue

            all_pick_pos.append(best_candidate["grasp_pos"])
            all_pick_quat.append(best_candidate["grasp_quat"])
            valid_env_indices.append(env_idx)
            print(
                f"  ✅ Env {env_idx}: multidir selected {best_candidate['direction_name']}, "
                f"pos={best_candidate['grasp_pos']}, quat={best_candidate['grasp_quat']}"
            )

        all_hang_below = compute_pick_hang_below(collector, target_label, all_pick_pos, target_height)
        print(f"  📏 Pre-computed hang_below: {all_hang_below}")
        context["affordance_selection_info"] = all_selection_info

        result = store_pick_grasp_context(
            context,
            mode=mode,
            target_label=target_label,
            all_pick_pos=all_pick_pos,
            all_pick_quat=all_pick_quat,
            valid_env_indices=valid_env_indices,
            target_height=target_height,
            all_hang_below=all_hang_below,
            target_available_mask=target_available_mask,
            robot_params=collector._get_pick_robot_params(),
            pre_pick_offset=params.get("pre_pick_offset", [-0.05, 0.0, 0.05]),
            pick_interpolation_steps=int(params.get("pick_interpolation_steps", 15)),
            lift_interpolation_steps=int(params.get("lift_interpolation_steps", 15)),
        )
        if result:
            context["pick_mode"] = "pick_4dir"
            context["selected_grasp_centers_local"] = get_selected_multidir_grasp_center_locals(collector, context)
        return result

    raise NotImplementedError(f"Primitive 'compute_grasp_pose' mode '{mode}' is not implemented yet.")
