from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from ..success.constants import HANG_SUCCESS_MIN_Z, HANG_SUCCESS_XY_THRESHOLD
from ..success.evaluators import _evaluate_hang_success_with_metrics
from ..tools.affordance_resolver import resolve_affordance_path
from ..tools.pose_utils import _quat_wxyz_to_matrix, _quat_xyzw_to_wxyz
from .utils import (
    compute_target_availability_mask,
    log_target_missing_envs,
)

FIXED_GRIPPER_STEPS = 10


def _normalize(vec: np.ndarray, eps: float = 1e-8) -> Optional[np.ndarray]:
    vec = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return None
    return vec / norm


def _resolve_optional_path(raw_path: Any, collector) -> Optional[Path]:
    if raw_path is None:
        return None
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    if getattr(collector, "task_config_path", None):
        return (Path(collector.task_config_path).parent / path).resolve()
    return path.resolve()


def _load_mount_point_from_npz(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    if "centroids" in data:
        centroids = np.asarray(data["centroids"], dtype=np.float64)
        if centroids.ndim == 2 and len(centroids) > 0:
            return np.array(centroids[0], dtype=np.float64)
    if "grasp_center" in data:
        return np.asarray(data["grasp_center"], dtype=np.float64)
    raise KeyError(f"Unsupported affordance npz keys in {npz_path}")


def _load_object_center_from_affordance_dir(npz_path: Path) -> np.ndarray:
    base_dir = npz_path.resolve().parent
    pt_path = base_dir / "labeled_result.pt"
    npz_labeled_path = base_dir / "labeled_result.npz"

    if pt_path.exists():
        data = torch.load(pt_path, map_location="cpu")
        points = data["points"]
        points = points.numpy() if hasattr(points, "numpy") else np.asarray(points, dtype=np.float64)
        return np.median(points, axis=0).astype(np.float64)

    if npz_labeled_path.exists():
        data = np.load(npz_labeled_path, allow_pickle=True)
        points = np.asarray(data["points"], dtype=np.float64)
        return np.median(points, axis=0).astype(np.float64)

    raise FileNotFoundError(f"Missing labeled_result.pt/.npz under affordance dir: {base_dir}")


def _resolve_target_mount_local(
    collector,
    params: Dict[str, Any],
    target_label: str,
) -> Optional[np.ndarray]:
    raw_npz_path = params.get("target_mount_affordance_npz_path") or resolve_affordance_path(target_label, "hang")
    npz_path = _resolve_optional_path(raw_npz_path, collector)
    if npz_path is None or not npz_path.exists():
        return None
    return _load_mount_point_from_npz(npz_path)


def _resolve_container_mount_local(
    collector,
    params: Dict[str, Any],
    container_label: str,
) -> Optional[np.ndarray]:
    raw_npz_path = params.get("container_mount_affordance_npz_path") or resolve_affordance_path(container_label, "hang")
    npz_path = _resolve_optional_path(raw_npz_path, collector)
    if npz_path is None or not npz_path.exists():
        return None
    return _load_mount_point_from_npz(npz_path)


def _resolve_container_center_local(
    collector,
    params: Dict[str, Any],
    container_label: str,
) -> Optional[np.ndarray]:
    raw_npz_path = params.get("container_mount_affordance_npz_path") or resolve_affordance_path(container_label, "hang")
    npz_path = _resolve_optional_path(raw_npz_path, collector)
    if npz_path is None or not npz_path.exists():
        return None
    return _load_object_center_from_affordance_dir(npz_path)


def _build_hang_orientation(
    x_axis_world: np.ndarray,
    y_reference_world: np.ndarray,
    reference_quat_wxyz: np.ndarray,
) -> np.ndarray:
    x_axis = _normalize(x_axis_world)
    if x_axis is None:
        raise ValueError("Zero-length hang direction.")

    y_reference_world = np.asarray(y_reference_world, dtype=np.float64)
    y_axis = y_reference_world - np.dot(y_reference_world, x_axis) * x_axis
    y_axis = _normalize(y_axis)

    if y_axis is None:
        reference_rot = _quat_wxyz_to_matrix(np.asarray(reference_quat_wxyz, dtype=np.float64))
        current_y = reference_rot[:, 1]
        y_axis = current_y - np.dot(current_y, x_axis) * x_axis
        y_axis = _normalize(y_axis)

    if y_axis is None:
        fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(fallback, x_axis))) > 0.9:
            fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        y_axis = fallback - np.dot(fallback, x_axis) * x_axis
        y_axis = _normalize(y_axis)

    if y_axis is None:
        raise ValueError("Failed to construct hang y-axis.")

    z_axis = np.cross(x_axis, y_axis)
    z_axis = _normalize(z_axis)
    if z_axis is None:
        raise ValueError("Failed to construct hang z-axis.")

    y_axis = np.cross(z_axis, x_axis)
    y_axis = _normalize(y_axis)
    if y_axis is None:
        raise ValueError("Failed to re-orthogonalize hang y-axis.")

    rot_world = np.column_stack([x_axis, y_axis, z_axis])
    return _quat_xyzw_to_wxyz(Rotation.from_matrix(rot_world).as_quat())


def _adjust_mount_direction_with_angle(
    original_direction_world: np.ndarray,
    target_angle_deg: Optional[float],
    eps: float = 1e-8,
) -> np.ndarray:
    direction_unit = _normalize(original_direction_world, eps=eps)
    if direction_unit is None:
        raise ValueError("Zero-length hang direction.")

    if target_angle_deg is None:
        return direction_unit

    neg_z = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    lateral = direction_unit - np.dot(direction_unit, neg_z) * neg_z
    lateral_unit = _normalize(lateral, eps=eps)

    # If the original direction is (almost) parallel to z, the plane is ill-defined.
    # Keep the original direction in that degenerate case.
    if lateral_unit is None:
        return direction_unit

    theta = np.deg2rad(float(target_angle_deg))
    adjusted = np.cos(theta) * neg_z + np.sin(theta) * lateral_unit
    adjusted_unit = _normalize(adjusted, eps=eps)
    if adjusted_unit is None:
        raise ValueError("Failed to construct adjusted hang direction.")
    return adjusted_unit


def _roll_hang_orientation_to_make_z_down(quat_wxyz: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    base_rot = _quat_wxyz_to_matrix(np.asarray(quat_wxyz, dtype=np.float64))
    world_down = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    candidates = []

    for angle_deg in (60.0, -60.0):
        local_roll = Rotation.from_euler("x", angle_deg, degrees=True).as_matrix()
        rotated = base_rot @ local_roll
        z_axis_world = rotated[:, 2]
        score = float(np.dot(z_axis_world, world_down))
        candidates.append(
            {
                "angle_deg": angle_deg,
                "score": score,
                "quat": _quat_xyzw_to_wxyz(Rotation.from_matrix(rotated).as_quat()),
                "z_axis_world": z_axis_world.tolist(),
            }
        )

    best = max(candidates, key=lambda item: item["score"])
    return np.asarray(best["quat"], dtype=np.float64), {
        "selected_local_x_roll_deg": best["angle_deg"],
        "selected_z_down_score": best["score"],
        "candidate_local_x_roll_scores": {str(item["angle_deg"]): item["score"] for item in candidates},
        "selected_z_axis_world": best["z_axis_world"],
    }


def _compute_orientation_z_down_score(quat_wxyz: np.ndarray) -> tuple[float, list[float]]:
    z_axis_world = _quat_wxyz_to_matrix(np.asarray(quat_wxyz, dtype=np.float64))[:, 2]
    score = float(np.dot(z_axis_world, np.array([0.0, 0.0, -1.0], dtype=np.float64)))
    return score, z_axis_world.tolist()


def _compute_hang_orientation_candidates(
    inward_direction_world: np.ndarray,
    reference_quat_wxyz: np.ndarray,
) -> list[tuple[str, np.ndarray]]:
    inward_direction_world = _normalize(inward_direction_world)
    if inward_direction_world is None:
        raise ValueError("Zero-length hang direction.")

    return [
        (
            "x_along_d_y_down",
            _build_hang_orientation(
                x_axis_world=inward_direction_world,
                y_reference_world=np.array([0.0, 0.0, -1.0], dtype=np.float64),
                reference_quat_wxyz=reference_quat_wxyz,
            ),
        ),
        (
            "x_against_d_y_up",
            _build_hang_orientation(
                x_axis_world=-inward_direction_world,
                y_reference_world=np.array([0.0, 0.0, 1.0], dtype=np.float64),
                reference_quat_wxyz=reference_quat_wxyz,
            ),
        ),
    ]


def primitive_hang(collector, mode: Optional[str], params: Dict[str, Any], context: Dict[str, Any]) -> bool:
    """Composite hang primitive: compute hang pose -> execute hang."""
    if not primitive_compute_hang_pose(collector, mode, params, context):
        return False
    return primitive_execute_hang(collector, params, context)


def primitive_compute_hang_pose(collector, mode: Optional[str], params: Dict[str, Any], context: Dict[str, Any]) -> bool:
    if context.get("post_pick_success_mask") is None:
        print("[ERROR] Primitive 'compute_hang_pose' requires pick execution first.")
        return False

    target_label = context.get("held_object") or context.get("pick_target_label")
    container_label = params.get("target")
    if not target_label or not container_label:
        print("[ERROR] Primitive 'compute_hang_pose' requires held object and target rack.")
        return False

    target_mount_local = _resolve_target_mount_local(collector, params, target_label)
    if target_mount_local is None:
        print(
            "[ERROR] Missing target hang mount point. "
            "Set params.target_mount_affordance_npz_path or provide "
            "pc_dataset_generate/pc/{object}_hang/affordance_clusters.npz."
        )
        return False

    container_mount_local = _resolve_container_mount_local(collector, params, container_label)
    if container_mount_local is None:
        print(
            "[ERROR] Missing container hang mount point. "
            "Set params.container_mount_affordance_npz_path or provide "
            "pc_dataset_generate/pc/{object}_hang/affordance_clusters.npz."
        )
        return False

    container_center_local = _resolve_container_center_local(collector, params, container_label)
    if container_center_local is None:
        print(
            "[ERROR] Missing container center/direction for hang. "
            "Resolve container hang affordance first, then ensure the same directory contains "
            "labeled_result.pt or labeled_result.npz."
        )
        return False

    pre_hang_distance = float(params.get("pre_hang_distance", 0.10))
    target_mount_angle_deg = params.get("mount_direction_angle_deg")
    prev_success_mask = list(
        context.get(
            "env_alive_mask",
            context.get("post_pick_success_mask", [True] * collector.num_envs),
        )
    )
    target_available_mask = compute_target_availability_mask(collector, container_label)
    log_target_missing_envs("hang", container_label, target_available_mask, prev_success_mask)
    env_origins = collector._get_env_origins()
    current_ee_pos_env, current_ee_quat = collector.get_current_ee_pose()

    all_pre_positions = []
    all_final_positions = []
    all_orientations = []
    all_mount_directions = []
    all_meta = []

    lift_positions = np.asarray(context.get("lift_positions", context.get("all_pick_pos")), dtype=np.float64)
    fallback_quats = np.asarray(context.get("all_pick_quat"), dtype=np.float64)

    for env_idx in range(collector.num_envs):
        fallback_pos = np.array(lift_positions[env_idx], dtype=np.float64)
        fallback_quat = np.array(fallback_quats[env_idx], dtype=np.float64)

        if not prev_success_mask[env_idx]:
            all_pre_positions.append(fallback_pos)
            all_final_positions.append(fallback_pos)
            all_orientations.append(fallback_quat)
            all_mount_directions.append(np.array([1.0, 0.0, 0.0], dtype=np.float64))
            all_meta.append({"status": "skip_after_pick_failure"})
            continue
        if not target_available_mask[env_idx]:
            all_pre_positions.append(fallback_pos)
            all_final_positions.append(fallback_pos)
            all_orientations.append(fallback_quat)
            all_mount_directions.append(np.array([1.0, 0.0, 0.0], dtype=np.float64))
            all_meta.append({"status": "target_missing"})
            continue

        obj_pos_world, obj_quat_world = collector._get_object_pose_global(target_label, env_idx)
        cont_pos_world, cont_quat_world = collector._get_object_pose_global(container_label, env_idx)
        if obj_pos_world is None or cont_pos_world is None:
            print(f"  ⚠️ Env {env_idx}: hang pose unavailable, using lift fallback.")
            all_pre_positions.append(fallback_pos)
            all_final_positions.append(fallback_pos)
            all_orientations.append(fallback_quat)
            all_mount_directions.append(np.array([1.0, 0.0, 0.0], dtype=np.float64))
            all_meta.append({"status": "missing_pose"})
            continue

        obj_pos_world = np.asarray(obj_pos_world, dtype=np.float64)
        obj_quat_world = np.asarray(obj_quat_world, dtype=np.float64)
        cont_pos_world = np.asarray(cont_pos_world, dtype=np.float64)
        cont_quat_world = np.asarray(cont_quat_world, dtype=np.float64)
        env_origin = np.asarray(env_origins[env_idx], dtype=np.float64)
        ee_pos_world = np.asarray(current_ee_pos_env[env_idx], dtype=np.float64) + env_origin
        ee_quat_wxyz = np.asarray(current_ee_quat[env_idx], dtype=np.float64)

        rot_obj = _quat_wxyz_to_matrix(obj_quat_world)
        rot_cont = _quat_wxyz_to_matrix(cont_quat_world)
        rot_ee_current = _quat_wxyz_to_matrix(ee_quat_wxyz)

        target_mount_world = rot_obj @ target_mount_local + obj_pos_world
        container_mount_world = rot_cont @ container_mount_local + cont_pos_world
        container_center_world = rot_cont @ container_center_local + cont_pos_world
        inward_direction_world_raw = container_center_world - container_mount_world

        try:
            inward_direction_unit = _adjust_mount_direction_with_angle(
                original_direction_world=inward_direction_world_raw,
                target_angle_deg=target_mount_angle_deg,
            )
        except ValueError as exc:
            print(f"  ⚠️ Env {env_idx}: {exc}, using lift fallback.")
            all_pre_positions.append(fallback_pos)
            all_final_positions.append(fallback_pos)
            all_orientations.append(fallback_quat)
            all_mount_directions.append(np.array([1.0, 0.0, 0.0], dtype=np.float64))
            all_meta.append({"status": "degenerate_direction", "error": str(exc)})
            continue

        try:
            candidate_quats = _compute_hang_orientation_candidates(inward_direction_unit, ee_quat_wxyz)
        except ValueError as exc:
            print(f"  ⚠️ Env {env_idx}: {exc}, using lift fallback.")
            all_pre_positions.append(fallback_pos)
            all_final_positions.append(fallback_pos)
            all_orientations.append(fallback_quat)
            all_mount_directions.append(np.array([1.0, 0.0, 0.0], dtype=np.float64))
            all_meta.append({"status": "orientation_failed", "error": str(exc)})
            continue

        target_mount_offset_in_ee = rot_ee_current.T @ (target_mount_world - ee_pos_world)
        initial_pose_candidates = []
        for candidate_name, candidate_quat in candidate_quats:
            candidate_quat = np.asarray(candidate_quat, dtype=np.float64)
            z_down_score, z_axis_world = _compute_orientation_z_down_score(candidate_quat)
            rot_ee_candidate = _quat_wxyz_to_matrix(candidate_quat)
            candidate_final_ee_world = container_mount_world - rot_ee_candidate @ target_mount_offset_in_ee
            initial_pose_candidates.append(
                {
                    "name": candidate_name,
                    "quat": candidate_quat,
                    "ee_world_z": float(candidate_final_ee_world[2]),
                    "final_ee_world": candidate_final_ee_world.tolist(),
                    "z_down_score": z_down_score,
                    "z_axis_world": z_axis_world,
                }
            )

        selected_initial = min(initial_pose_candidates, key=lambda item: item["ee_world_z"])
        final_quat, roll_meta = _roll_hang_orientation_to_make_z_down(selected_initial["quat"])
        rot_ee_target = _quat_wxyz_to_matrix(final_quat)
        final_ee_world = container_mount_world - rot_ee_target @ target_mount_offset_in_ee
        pre_ee_world = final_ee_world - inward_direction_unit * pre_hang_distance

        all_pre_positions.append(pre_ee_world - env_origin)
        all_final_positions.append(final_ee_world - env_origin)
        all_orientations.append(final_quat)
        all_mount_directions.append(inward_direction_unit)
        all_meta.append(
            {
                "status": "ok",
                "target_mount_world": target_mount_world.tolist(),
                "container_mount_world": container_mount_world.tolist(),
                "container_center_world": container_center_world.tolist(),
                "container_mount_direction_world_raw": inward_direction_world_raw.tolist(),
                "mount_direction_angle_deg": target_mount_angle_deg,
                "target_mount_offset_in_ee": target_mount_offset_in_ee.tolist(),
                "selected_orientation_mode": selected_initial["name"],
                "initial_candidate_ee_world_z": {
                    item["name"]: item["ee_world_z"] for item in initial_pose_candidates
                },
                "initial_candidate_final_ee_world": {
                    item["name"]: item["final_ee_world"] for item in initial_pose_candidates
                },
                "initial_orientation_z_down_scores": {
                    item["name"]: item["z_down_score"] for item in initial_pose_candidates
                },
                "initial_orientation_z_axes_world": {
                    item["name"]: item["z_axis_world"] for item in initial_pose_candidates
                },
                "selected_roll_meta": roll_meta,
            }
        )

    context["hang_pose"] = {
        "mode": mode or "mount",
        "held_object_label": target_label,
        "target_label": container_label,
        "target_available_mask": list(target_available_mask),
        "pre_positions": np.asarray(all_pre_positions, dtype=np.float64),
        "positions": np.asarray(all_final_positions, dtype=np.float64),
        "orientations": np.asarray(all_orientations, dtype=np.float64),
        "mount_directions": np.asarray(all_mount_directions, dtype=np.float64),
        "pre_hang_distance": pre_hang_distance,
        "target_mount_local": np.asarray(target_mount_local, dtype=np.float64),
        "container_mount_local": np.asarray(container_mount_local, dtype=np.float64),
        "meta": all_meta,
    }
    print(f"  🪝 [Primitive] Computed hang poses for '{target_label}' -> '{container_label}'")
    return True


def primitive_execute_hang(collector, params: Dict[str, Any], context: Dict[str, Any]) -> bool:
    hang_pose = context.get("hang_pose")
    if hang_pose is None:
        print("[ERROR] Primitive 'execute_hang' requires hang poses first.")
        return False

    prev_success_mask = list(
        context.get(
            "env_alive_mask",
            context.get("post_pick_success_mask", [True] * collector.num_envs),
        )
    )
    target_available_mask = list(hang_pose.get("target_available_mask", [True] * collector.num_envs))
    pre_positions = np.asarray(hang_pose["pre_positions"], dtype=np.float64)
    final_positions = np.asarray(hang_pose["positions"], dtype=np.float64)
    orientations = np.asarray(hang_pose["orientations"], dtype=np.float64)
    mount_directions = np.asarray(hang_pose["mount_directions"], dtype=np.float64)
    target_mount_local = np.asarray(hang_pose["target_mount_local"], dtype=np.float64)
    container_mount_local = np.asarray(hang_pose["container_mount_local"], dtype=np.float64)
    held_object_label = hang_pose.get("held_object_label")
    container_label = hang_pose.get("target_label")
    extra_insert_distance = float(params.get("extra_insert_distance", 0.05))

    print(f"  🚀 [Primitive] Executing HANG to '{hang_pose.get('target_label')}'")

    pre_plans, pre_plan_successes = collector.motion_planner.plan_to_pose_batch(
        collector.robot.data.joint_pos[: collector.num_envs].cpu().numpy(),
        collector.robot.data.joint_vel[: collector.num_envs].cpu().numpy(),
        collector.robot.joint_names,
        pre_positions,
        orientations,
        target_prim_path=collector.target_prim_path,
    )
    pre_success_mask = [prev and available and cur for prev, available, cur in zip(prev_success_mask, target_available_mask, pre_plan_successes)]

    collector.execute_trajectory_batch(
        pre_plans,
        [-1.0 if success else 0.0 for success in pre_success_mask],
        desc="Primitive Pre-Hang",
        active_mask=pre_success_mask,
    )

    _, insert_successes = collector.execute_ik_motion(
        target_positions=final_positions,
        target_quaternions=orientations,
        gripper_states=[-1.0 if success else 0.0 for success in pre_success_mask],
        interpolation_steps=int(params.get("insert_interpolation_steps", 20)),
        desc="Primitive Hang Insert",
        active_mask=pre_success_mask,
    )
    final_success_mask = [a and b for a, b in zip(pre_success_mask, insert_successes)]

    release_positions = final_positions.copy()
    if extra_insert_distance > 0.0:
        extra_insert_positions = final_positions + mount_directions * extra_insert_distance
        _, extra_insert_successes = collector.execute_ik_motion(
            target_positions=extra_insert_positions,
            target_quaternions=orientations,
            gripper_states=[-1.0 if success else 0.0 for success in final_success_mask],
            interpolation_steps=int(params.get("extra_insert_interpolation_steps", 10)),
            desc="Primitive Hang Extra Insert",
            active_mask=final_success_mask,
        )
        final_success_mask = [a and b for a, b in zip(final_success_mask, extra_insert_successes)]
        release_positions = extra_insert_positions

    collector.control_gripper_batch(
        [1.0 if success else 0.0 for success in final_success_mask],
        steps=FIXED_GRIPPER_STEPS,
        active_mask=final_success_mask,
    )

    retreat_distance = float(params.get("post_release_retreat_distance", hang_pose.get("pre_hang_distance", 0.10)))
    if retreat_distance > 0.0:
        retreat_positions = release_positions.copy()
        for env_idx in range(collector.num_envs):
            gripper_z_axis = _quat_wxyz_to_matrix(orientations[env_idx])[:, 2]
            retreat_positions[env_idx] -= np.asarray(gripper_z_axis, dtype=np.float64) * retreat_distance
        _, retreat_successes = collector.execute_ik_motion(
            target_positions=retreat_positions,
            target_quaternions=orientations,
            gripper_states=[1.0 if success else 0.0 for success in final_success_mask],
            interpolation_steps=int(params.get("retreat_interpolation_steps", 10)),
            desc="Primitive Hang Retreat",
            active_mask=final_success_mask,
        )
        final_success_mask = [a and b for a, b in zip(final_success_mask, retreat_successes)]

    step_failure_reason_mask = ["none"] * collector.num_envs
    for env_idx in range(collector.num_envs):
        if not prev_success_mask[env_idx]:
            step_failure_reason_mask[env_idx] = "inactive"
        elif not target_available_mask[env_idx]:
            step_failure_reason_mask[env_idx] = "target_missing"
        elif not final_success_mask[env_idx]:
            step_failure_reason_mask[env_idx] = "planning_failed"

    success_eval_enabled = bool(params.get("success_eval", False))
    if success_eval_enabled:
        state_successes = []
        state_metrics = []
        for env_idx in range(collector.num_envs):
            env_success, env_metrics = _evaluate_hang_success_with_metrics(
                scene=collector.scene,
                held_object_label=held_object_label,
                container_label=container_label,
                container_mount_local=container_mount_local,
                env_idx=env_idx,
            )
            state_successes.append(env_success)
            state_metrics.append(env_metrics)

        final_success_mask = [motion_ok and state_ok for motion_ok, state_ok in zip(final_success_mask, state_successes)]
        for env_idx in range(collector.num_envs):
            if not prev_success_mask[env_idx]:
                continue
            if final_success_mask[env_idx]:
                step_failure_reason_mask[env_idx] = "none"
            elif step_failure_reason_mask[env_idx] in {"planning_failed", "target_missing"}:
                continue
            else:
                step_failure_reason_mask[env_idx] = "success_eval_failed"

        context["task_success_metrics"] = list(state_metrics)
        for env_idx, metrics in enumerate(state_metrics):
            obj_z = metrics["held_object_z_env"]
            z_ok = np.isfinite(obj_z) and (obj_z > HANG_SUCCESS_MIN_Z)
            mount_dist = metrics["mount_xy_dist"]
            dist_ok = np.isfinite(mount_dist) and (mount_dist <= HANG_SUCCESS_XY_THRESHOLD)
            range_ok = bool(metrics["container_in_range"] > 0.5)
            z_text = f"{obj_z:.4f}" if np.isfinite(obj_z) else "nan"
            dist_text = f"{mount_dist:.4f}" if np.isfinite(mount_dist) else "inf"
            print(
                f"    [TaskCheck][env {env_idx}] "
                f"held_z={z_text}m (>{HANG_SUCCESS_MIN_Z:.2f}: {'OK' if z_ok else 'FAIL'}), "
                f"mount_xy_dist={dist_text}m (<={HANG_SUCCESS_XY_THRESHOLD:.3f}: {'OK' if dist_ok else 'FAIL'}), "
                f"container_in_range={'OK' if range_ok else 'FAIL'}"
            )
        print(
            f"  ✅ [Primitive] Task success (hang): "
            f"{sum(final_success_mask)}/{collector.num_envs}"
        )

    context["post_hang_success_mask"] = list(final_success_mask)
    context["env_alive_mask"] = list(final_success_mask)
    context["task_success_mask"] = list(final_success_mask)
    context["current_gripper_state"] = "open"
    context["held_object"] = None
    context["primitive_executed_motion"] = True
    context["hang_completed"] = any(final_success_mask)
    context["step_failure_reason_mask"] = step_failure_reason_mask

    print(f"  ✅ [Primitive] Hang success: {sum(final_success_mask)}/{collector.num_envs}")
    return any(final_success_mask)
