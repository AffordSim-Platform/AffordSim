from typing import Any, Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..success.constants import PLACE_SUCCESS_GRIPPER_OPEN_THRESHOLD, PLACE_SUCCESS_XY_THRESHOLD
from ..success.evaluators import (
    _evaluate_place_pose_success_with_metrics,
    _evaluate_place_success_with_metrics,
    _merge_gripper_release_metrics,
)
from ..tools.affordance_resolver import resolve_affordance_path
from ..tools.pose_utils import (
    _quat_wxyz_to_matrix,
    compute_face_target_place_quat,
    normalize_ee_orientation_wxyz,
    parse_facing_axis_local,
)
from .utils import (
    FAILURE_REASON_TARGET_MISSING,
    commit_plan_failures,
    commit_task_failures,
    get_prev_alive_mask,
    init_local_active_mask,
    load_affordance_centroid,
    mark_local_inactive_with_reason,
    merge_segment_into_local_active,
    normalize_primitive_mode,
    parse_position_offset,
    resolve_scene_entry_height,
)

FIXED_GRIPPER_STEPS = 3


def _parse_bool_param(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", "n"}
    return bool(value)


def _format_gripper_release_metric(metrics: dict) -> str:
    if float(metrics.get("gripper_release_required", 0.0) or 0.0) <= 0.5:
        return ""
    released = bool(float(metrics.get("gripper_released", 0.0) or 0.0) > 0.5)
    finger_pos = float(metrics.get("gripper_finger_pos", float("nan")))
    direction = metrics.get("gripper_open_direction", "auto")
    threshold = float(metrics.get("gripper_open_threshold", PLACE_SUCCESS_GRIPPER_OPEN_THRESHOLD))
    return (
        f", gripper_released={'OK' if released else 'FAIL'} "
        f"(pos={finger_pos:.4f}, {direction}, threshold={threshold:.4f})"
    )


def _get_task_object_scale(collector, object_label: str) -> np.ndarray:
    """Return configured object scale, or ones when scale metadata is unavailable."""
    if not getattr(collector, "task_config", None):
        return np.ones(3, dtype=np.float64)

    object_cfg = collector.task_config.get_object_by_name(object_label)
    if object_cfg is None:
        object_cfg = collector.task_config.articulations.get(object_label)
    if object_cfg is None or getattr(object_cfg, "scale", None) is None:
        return np.ones(3, dtype=np.float64)
    return np.asarray(object_cfg.scale, dtype=np.float64)


def _load_scaled_place_affordance_centroid(
    collector,
    object_label: str,
    affordance_npz_path: str | None = None,
) -> tuple[np.ndarray | None, str | None]:
    """Load the first place-affordance centroid and apply task-level object scaling."""
    affordance_path = affordance_npz_path or resolve_affordance_path(object_label, "place")
    if not affordance_path:
        return None, None

    affordance_local = load_affordance_centroid(affordance_path)
    affordance_local = affordance_local * _get_task_object_scale(collector, object_label)
    return affordance_local, str(affordance_path)


def _evaluate_place_affordance_success_with_metrics(
    collector,
    pick_target: str,
    place_target: str,
    pick_affordance_local: np.ndarray,
    target_affordance_local: np.ndarray,
    env_idx: int = 0,
    distance_threshold: float = PLACE_SUCCESS_XY_THRESHOLD,
    require_gripper_release: bool = True,
    gripper_open_threshold: float = PLACE_SUCCESS_GRIPPER_OPEN_THRESHOLD,
    gripper_joint_name_substr: str = "finger,drive",
    gripper_open_direction: str = "auto",
) -> tuple[bool, dict[str, float]]:
    """Evaluate affordance-mode placement by affordance-centroid distance only."""
    pick_pos_world, pick_quat_world = collector._get_object_pose_global(pick_target, env_idx)
    place_pos_world, place_quat_world = collector._get_object_pose_global(place_target, env_idx)

    if pick_pos_world is None or pick_quat_world is None or place_pos_world is None or place_quat_world is None:
        _, metrics = _merge_gripper_release_metrics(
            {"pick_place_affordance_dist": float("inf")},
            collector.scene,
            env_idx,
            require_gripper_release,
            gripper_open_threshold,
            gripper_joint_name_substr,
            gripper_open_direction,
        )
        return False, metrics

    pick_affordance_world = (
        _quat_wxyz_to_matrix(np.asarray(pick_quat_world, dtype=np.float64))
        @ np.asarray(pick_affordance_local, dtype=np.float64)
        + np.asarray(pick_pos_world, dtype=np.float64)
    )
    target_affordance_world = (
        _quat_wxyz_to_matrix(np.asarray(place_quat_world, dtype=np.float64))
        @ np.asarray(target_affordance_local, dtype=np.float64)
        + np.asarray(place_pos_world, dtype=np.float64)
    )

    affordance_dist = float(np.linalg.norm(pick_affordance_world - target_affordance_world))
    gripper_ok, metrics = _merge_gripper_release_metrics(
        {"pick_place_affordance_dist": affordance_dist},
        collector.scene,
        env_idx,
        require_gripper_release,
        gripper_open_threshold,
        gripper_joint_name_substr,
        gripper_open_direction,
    )
    return affordance_dist <= float(distance_threshold) and gripper_ok, metrics


def _find_enclosing_semi_enclosed_container_xy(collector, env_idx: int, point_xy_env: np.ndarray):
    """Return the semi-enclosed container whose XY footprint contains the env-local query point."""
    if not hasattr(collector, "_containers_registry"):
        collector._containers_registry = collector._load_containers_registry()

    registry = getattr(collector, "_containers_registry", {}) or {}
    if not registry:
        return None, None, None

    point_xy_env = np.asarray(point_xy_env, dtype=np.float64)
    for container_key, cfg in registry.items():
        if "dimensions" not in cfg:
            # affordance-only placement target (only declares facing_axis_local), does not participate in semi-enclosed XY detection
            continue
        cont_pos_world, cont_quat = collector._get_object_pose_global(container_key, env_idx)
        if cont_pos_world is None or cont_quat is None:
            continue
        env_origin = collector._get_env_origins()[env_idx]
        cont_pos = np.asarray(cont_pos_world, dtype=np.float64) - np.asarray(env_origin, dtype=np.float64)

        cont_pos = np.asarray(cont_pos, dtype=np.float64)
        w, qx, qy, qz = [float(v) for v in cont_quat]
        rot = R.from_quat([qx, qy, qz, w])
        query_env = np.array([point_xy_env[0], point_xy_env[1], cont_pos[2]], dtype=np.float64)
        local_pos = rot.inv().apply(query_env - cont_pos)

        dims = np.asarray(cfg["dimensions"], dtype=np.float64)
        half_dims = dims / 2.0
        if abs(float(local_pos[0])) <= half_dims[0] and abs(float(local_pos[1])) <= half_dims[1]:
            return container_key, cont_pos, np.asarray(cont_quat, dtype=np.float64)

    return None, None, None


def _compute_place_pose_affordance(
    collector,
    params: Dict[str, Any],
    context: Dict[str, Any],
    container_label: str,
    target_available_mask,
    place_offset: np.ndarray,
    ee_quat_w_place: np.ndarray,
) -> bool:
    """Compute place poses by aligning the held object's place affordance centroid
    with the target's place affordance centroid.

    Two-step strategy: pre_place is offset 10cm horizontally toward the current EE
    (the robot side), and the final pose places the centroids exactly aligned while
    keeping the current EE orientation.
    """
    pick_target_label = context.get("pick_target_label") or context.get("held_object")
    if not pick_target_label:
        print(
            "[ERROR] Primitive 'compute_place_pose' affordance mode requires a held object "
            "(pick must be executed first)."
        )
        return False

    pick_aff_path = params.get("pick_affordance_npz_path") or resolve_affordance_path(pick_target_label, "place")
    target_aff_path = params.get("target_affordance_npz_path") or resolve_affordance_path(container_label, "place")
    if not pick_aff_path:
        print(
            f"[ERROR] Missing place affordance for picked object '{pick_target_label}'. "
            f"Expected pc_dataset_generate/pc/{pick_target_label}_place/affordance_clusters.npz "
            "or set params.pick_affordance_npz_path."
        )
        return False
    if not target_aff_path:
        print(
            f"[ERROR] Missing place affordance for target '{container_label}'. "
            f"Expected pc_dataset_generate/pc/{container_label}_place/affordance_clusters.npz "
            "or set params.target_affordance_npz_path."
        )
        return False

    print(
        f"  🎯 [Primitive] Loading place affordances: "
        f"pick='{pick_aff_path}', target='{target_aff_path}'"
    )
    pick_aff_local, _ = _load_scaled_place_affordance_centroid(
        collector,
        pick_target_label,
        affordance_npz_path=pick_aff_path,
    )
    target_aff_local, _ = _load_scaled_place_affordance_centroid(
        collector,
        container_label,
        affordance_npz_path=target_aff_path,
    )
    if pick_aff_local is None or target_aff_local is None:
        print("[ERROR] Failed to load scaled place affordance centroid for affordance-mode placement.")
        return False

    pre_place_distance = float(params.get("pre_place_distance", 0.15))
    affordance_lift_height = float(params.get("affordance_lift_height", 0.05))
    env_origins = collector._get_env_origins()
    current_ee_pos_env, current_ee_quat = collector.get_current_ee_pose()

    # Query target container's facing_axis_local (placement target registry, shared with semi-enclosed containers.yaml)
    if not hasattr(collector, "_containers_registry"):
        collector._containers_registry = collector._load_containers_registry()
    container_cfg = (collector._containers_registry or {}).get(container_label) or {}
    try:
        target_facing_axis_local = parse_facing_axis_local(container_cfg.get("facing_axis_local"))
    except ValueError as exc:
        print(f"[ERROR] Invalid facing_axis_local for affordance place target '{container_label}': {exc}")
        return False
    if target_facing_axis_local is None:
        print(
            f"[ERROR] Affordance place target '{container_label}' must define "
            "`facing_axis_local` in containers.yaml."
        )
        return False

    fallback_place_position = np.array([0.3, 0.0, 0.25], dtype=np.float64)
    fallback_place_quat = np.array([0.0, -1.0, 0.0, 0.0], dtype=np.float64)

    target_available_mask = list(target_available_mask)
    pre_place_positions = []
    pre_place_quaternions = []
    final_place_positions = []
    final_place_quaternions = []
    execution_modes = ["normal"] * collector.num_envs
    enclosing_container_labels = [None] * collector.num_envs

    for env_idx in range(collector.num_envs):
        env_ee_quat_ref = ee_quat_w_place[env_idx] if ee_quat_w_place.ndim > 1 else ee_quat_w_place

        if not target_available_mask[env_idx]:
            fallback_pos = fallback_place_position.copy()
            fallback_quat = normalize_ee_orientation_wxyz(
                fallback_place_quat.copy(),
                env_ee_quat_ref,
                mode="place",
            )
            pre_place_positions.append(fallback_pos.copy())
            pre_place_quaternions.append(fallback_quat.copy())
            final_place_positions.append(fallback_pos)
            final_place_quaternions.append(fallback_quat)
            continue

        obj_pos_w, obj_quat_w = collector._get_object_pose_global(pick_target_label, env_idx)
        tgt_pos_w, tgt_quat_w = collector._get_object_pose_global(container_label, env_idx)
        if obj_pos_w is None or tgt_pos_w is None:
            print(
                f"  ⚠️ Env {env_idx}: place affordance unavailable "
                f"(pick_pose={obj_pos_w is not None}, target_pose={tgt_pos_w is not None}); "
                "falling back."
            )
            target_available_mask[env_idx] = False
            fallback_pos = fallback_place_position.copy()
            fallback_quat = normalize_ee_orientation_wxyz(
                fallback_place_quat.copy(),
                env_ee_quat_ref,
                mode="place",
            )
            pre_place_positions.append(fallback_pos.copy())
            pre_place_quaternions.append(fallback_quat.copy())
            final_place_positions.append(fallback_pos)
            final_place_quaternions.append(fallback_quat)
            continue

        origin = np.asarray(env_origins[env_idx], dtype=np.float64)
        ee_pos_world = np.asarray(current_ee_pos_env[env_idx], dtype=np.float64) + origin
        ee_quat_wxyz = np.asarray(current_ee_quat[env_idx], dtype=np.float64)
        rot_ee = _quat_wxyz_to_matrix(ee_quat_wxyz)

        obj_pos_world = np.asarray(obj_pos_w, dtype=np.float64)
        obj_quat_world = np.asarray(obj_quat_w, dtype=np.float64)
        tgt_pos_world = np.asarray(tgt_pos_w, dtype=np.float64)
        tgt_quat_world = np.asarray(tgt_quat_w, dtype=np.float64)

        rot_obj = _quat_wxyz_to_matrix(obj_quat_world)
        rot_tgt = _quat_wxyz_to_matrix(tgt_quat_world)

        # Picked object's affordance centroid in the current EE frame.
        aff_obj_world_now = rot_obj @ pick_aff_local + obj_pos_world
        aff_obj_in_ee = rot_ee.T @ (aff_obj_world_now - ee_pos_world)

        # Target affordance centroid in world.
        aff_tgt_world = rot_tgt @ target_aff_local + tgt_pos_world

        # Compute placement pose:
        # Rotate the grasp around world z to face the container interior.
        try:
            place_quat_raw = compute_face_target_place_quat(
                target_quat_wxyz=tgt_quat_world,
                facing_axis_local=target_facing_axis_local,
                grasp_quat_wxyz=ee_quat_wxyz,
            )
        except ValueError as exc:
            print(f"[ERROR] Env {env_idx}: affordance place face-target orientation failed: {exc}")
            return False
        rot_ee_target = _quat_wxyz_to_matrix(place_quat_raw)
        # The held object's affordance centroid keeps the same local coordinates in the EE,
        # so transform aff_obj_in_ee using the new rot_ee_target to align in world frame.
        # Approach direction: +facing_axis_local rotated to world.
        # facing_axis_local points outward from the opening, pre-place retreats along this direction
        # from the final pose, so the EE approaches from outside the container inward.
        approach_dir = rot_tgt @ target_facing_axis_local

        final_ee_world = aff_tgt_world - rot_ee_target @ aff_obj_in_ee
        final_ee_world[2] += affordance_lift_height

        approach_norm = float(np.linalg.norm(approach_dir))
        if approach_norm < 1e-6:
            approach_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            approach_dir = approach_dir / approach_norm

        pre_ee_world = final_ee_world + approach_dir * pre_place_distance

        final_pos_env = final_ee_world - origin + place_offset
        pre_pos_env = pre_ee_world - origin + place_offset

        place_quat = normalize_ee_orientation_wxyz(
            place_quat_raw,
            env_ee_quat_ref,
            mode="place",
        )

        pre_place_positions.append(pre_pos_env)
        pre_place_quaternions.append(place_quat.copy())
        final_place_positions.append(final_pos_env)
        final_place_quaternions.append(place_quat)
        execution_modes[env_idx] = "affordance"

    context["place_pose"] = {
        "mode": "affordance",
        "target_label": container_label,
        "target_available_mask": list(target_available_mask),
        "execution_modes": list(execution_modes),
        "enclosing_container_labels": list(enclosing_container_labels),
        "pick_affordance_npz_path": str(pick_aff_path),
        "target_affordance_npz_path": str(target_aff_path),
        "pick_affordance_local": np.asarray(pick_aff_local, dtype=float).copy(),
        "target_affordance_local": np.asarray(target_aff_local, dtype=float).copy(),
        "pre_positions": np.asarray(pre_place_positions, dtype=float),
        "pre_orientations": np.asarray(pre_place_quaternions, dtype=float),
        "positions": np.asarray(final_place_positions, dtype=float),
        "orientations": np.asarray(final_place_quaternions, dtype=float),
        "pre_place_distance": pre_place_distance,
    }
    aff_count = sum(mode_name == "affordance" for mode_name in execution_modes)
    print(
        f"  🎯 [Primitive] Computed place pose for mode='affordance', "
        f"pick='{pick_target_label}', target='{container_label}', "
        f"affordance_envs={aff_count}/{collector.num_envs}, "
        f"pre_place_distance={pre_place_distance:.3f}"
    )
    return True


def _scatter_subset_plans(num_envs: int, active_indices, subset_plans, subset_successes):
    """Scatter subset planning outputs back to full env-sized lists."""
    full_plans = [None] * num_envs
    full_successes = [False] * num_envs
    for subset_idx, env_idx in enumerate(active_indices):
        full_plans[env_idx] = subset_plans[subset_idx]
        full_successes[env_idx] = subset_successes[subset_idx]
    return full_plans, full_successes


def primitive_place(collector, mode: Optional[str], params: Dict[str, Any], context: Dict[str, Any]) -> bool:
    """Composite place primitive: compute place pose -> execute place."""
    if not primitive_compute_place_pose(collector, mode, params, context):
        return False
    return primitive_execute_place(collector, params, context)


def primitive_compute_place_pose(collector, mode: Optional[str], params: Dict[str, Any], context: Dict[str, Any]) -> bool:
    if context.get("post_pick_success_mask") is None:
        print("[ERROR] Primitive 'compute_place_pose' requires pick execution first.")
        return False
    if "place_pos" in params:
        print("[ERROR] Parameter `place_pos` is no longer supported. Use `position`.")
        return False
    if "place_quat" in params:
        print("[ERROR] Parameter `place_quat` is no longer supported. Use `orientation`.")
        return False
    if "place_offset" in params:
        print("[ERROR] Parameter `place_offset` is no longer supported in place params. Use top-level `offsets.place`.")
        return False
    if "ignore_obstacles_mode" in params:
        print(
            "[ERROR] Parameter `ignore_obstacles_mode` is no longer supported. "
            "Semi-enclosed place now always uses ignore-container strategy."
        )
        return False
    if "use_container_height_for_in" in params:
        print(
            "[ERROR] Parameter `use_container_height_for_in` is no longer supported. "
            "For mode='in', container half-height is no longer added by default."
        )
        return False
    if "default_height_margin" in params:
        print(
            "[ERROR] Parameter `default_height_margin` is no longer supported. "
            "Fallback relation height margin is now fixed to 0.05m."
        )
        return False
    if "fallback_position" in params:
        print(
            "[ERROR] Parameter `fallback_position` is no longer supported. "
            "Missing-container fallback position is now fixed internally."
        )
        return False
    if "fallback_quat" in params:
        print(
            "[ERROR] Parameter `fallback_quat` is no longer supported. "
            "Missing-container fallback orientation is now fixed internally."
        )
        return False
    if "normalize_orientation" in params:
        print(
            "[ERROR] Parameter `normalize_orientation` is no longer supported. "
            "Place orientation normalization is now always enabled."
        )
        return False
    if "post_settle_steps" in params:
        print(
            "[ERROR] Parameter `post_settle_steps` is no longer supported in place params. "
            "Place no longer performs settle-step hold."
        )
        return False
    if "planner_position_offset" in params:
        print(
            "[ERROR] Parameter `planner_position_offset` is no longer supported. "
            "Place planning now directly targets final place pose."
        )
        return False
    if "ik_descend_offset" in params:
        print(
            "[ERROR] Parameter `ik_descend_offset` is no longer supported. "
            "Place no longer performs a second IK-descend stage."
        )
        return False
    if "require_post_release_lift_success" in params:
        print(
            "[ERROR] Parameter `require_post_release_lift_success` is no longer supported. "
            "When post-release lift is configured, its success is now always required."
        )
        return False

    requested_mode = normalize_primitive_mode(mode) or "on"
    container_label = params.get("target")
    if not container_label and requested_mode != "pose":
        print("[ERROR] Primitive 'compute_place_pose' requires a target container.")
        return False

    if requested_mode == "semi_enclosed":
        requested_mode = "in"

    place_offset = np.array(getattr(collector.task_config, "place_offset", [0.0, 0.0, 0.0]), dtype=float)

    _, ee_quat_w_place = collector.get_current_ee_pose()
    execution_modes = ["normal"] * collector.num_envs
    enclosing_container_labels = [None] * collector.num_envs
    target_available_mask = [True] * collector.num_envs
    pre_place_positions = []
    pre_place_quaternions = []

    if container_label:
        for env_idx in range(collector.num_envs):
            cont_pos, _ = collector._get_object_pose_global(container_label, env_idx)
            target_available_mask[env_idx] = cont_pos is not None
        missing_count = sum(not available for available in target_available_mask)
        if missing_count > 0:
            print(
                f"  ⚠️ [Primitive] Place target '{container_label}' missing in "
                f"{missing_count}/{collector.num_envs} envs; those envs will fail this place step"
            )

    if requested_mode == "affordance":
        return _compute_place_pose_affordance(
            collector,
            params,
            context,
            container_label=container_label,
            target_available_mask=target_available_mask,
            place_offset=place_offset,
            ee_quat_w_place=ee_quat_w_place,
        )

    if requested_mode == "pose":
        target_position = params.get("position")
        if target_position is None:
            print("[ERROR] Primitive 'compute_place_pose' with mode 'pose' requires `position`.")
            return False

        target_position = np.array(target_position, dtype=float)
        if target_position.ndim == 1:
            all_place_pos = np.tile(target_position, (collector.num_envs, 1))
        else:
            all_place_pos = target_position
        all_place_pos = np.array(all_place_pos, dtype=float) + place_offset

        target_orientation = params.get("orientation")
        if target_orientation is None:
            all_place_quat = np.array(context["all_pick_quat"], dtype=float)
        else:
            target_orientation = np.array(target_orientation, dtype=float)
            if target_orientation.ndim == 1:
                all_place_quat = np.tile(target_orientation, (collector.num_envs, 1))
            else:
                all_place_quat = target_orientation
        _, ee_quat_w_place = collector.get_current_ee_pose()
        for env_idx in range(collector.num_envs):
            env_ee_quat = ee_quat_w_place[env_idx] if ee_quat_w_place.ndim > 1 else ee_quat_w_place
            all_place_quat[env_idx] = normalize_ee_orientation_wxyz(
                np.array(all_place_quat[env_idx], dtype=float),
                env_ee_quat,
                mode="place",
            )
        all_place_pos = np.asarray(all_place_pos, dtype=float)
        all_place_quat = np.asarray(all_place_quat, dtype=float)
    else:
        relation = requested_mode.lower()
        if relation not in ["in", "inside", "on", "on_top"]:
            raise NotImplementedError(f"Primitive 'compute_place_pose' mode '{requested_mode}' is not implemented yet.")

        container_entry = collector._get_object_entry_from_scene_desc(container_label, context["scene_desc"])
        container_height = resolve_scene_entry_height(container_entry, 0.1)
        on_height_margin = float(params.get("on_height_margin", 0.01))
        fallback_relation_height_margin = 0.05
        fallback_place_position = np.array([0.3, 0.0, 0.25], dtype=float)
        fallback_place_quat = np.array([0.0, -1.0, 0.0, 0.0], dtype=float)
        orientation_override = params.get("orientation")
        orientation_override_np = None
        if orientation_override is not None:
            orientation_override_np = np.array(orientation_override, dtype=float)

        all_place_pos = []
        all_place_quat = []
        env_origins = collector._get_env_origins()

        for env_idx in range(collector.num_envs):
            cont_pos, _ = collector._get_object_pose_global(container_label, env_idx)
            hang_below = context["all_hang_below"][env_idx]

            if cont_pos is not None:
                place_pos = (cont_pos - env_origins[env_idx]).copy() + place_offset
                if relation in ["in", "inside"]:
                    place_pos[2] += hang_below
                elif relation in ["on", "on_top"]:
                    place_pos[2] += container_height / 2.0 + hang_below + on_height_margin
                if orientation_override_np is not None:
                    if orientation_override_np.ndim == 1:
                        place_quat = orientation_override_np.copy()
                    else:
                        place_quat = np.array(orientation_override_np[env_idx], dtype=float)
                else:
                    place_quat = np.array(context["all_pick_quat"][env_idx], dtype=float)
            else:
                place_pos = fallback_place_position.copy()
                place_pos[2] += hang_below + fallback_relation_height_margin
                if orientation_override_np is not None:
                    if orientation_override_np.ndim == 1:
                        place_quat = orientation_override_np.copy()
                    else:
                        place_quat = np.array(orientation_override_np[env_idx], dtype=float)
                else:
                    place_quat = fallback_place_quat.copy()

            env_ee_quat = ee_quat_w_place[env_idx] if ee_quat_w_place.ndim > 1 else ee_quat_w_place
            place_quat = normalize_ee_orientation_wxyz(place_quat, env_ee_quat, mode="place")

            all_place_pos.append(place_pos)
            all_place_quat.append(place_quat)

        all_place_pos = np.asarray(all_place_pos, dtype=float)
        all_place_quat = np.asarray(all_place_quat, dtype=float)

    for env_idx in range(collector.num_envs):
        env_ee_quat = ee_quat_w_place[env_idx] if ee_quat_w_place.ndim > 1 else ee_quat_w_place
        execution_mode = "normal"
        final_place_pos = np.array(all_place_pos[env_idx], dtype=float)
        final_place_quat = np.array(all_place_quat[env_idx], dtype=float)
        pre_place_pos = final_place_pos.copy()
        pre_place_quat = final_place_quat.copy()

        if target_available_mask[env_idx] and requested_mode in {"in", "inside", "pose"}:
            container_name, container_pos, container_quat = _find_enclosing_semi_enclosed_container_xy(
                collector,
                env_idx,
                final_place_pos[:2],
            )
            if container_name is not None:
                semi_pre_pose, semi_place_pose = collector.compute_semi_enclosed_placement_poses(
                    container_name=container_name,
                    container_position=np.array(container_pos),
                    container_quat_wxyz=np.array(container_quat),
                    reference_quat_wxyz=env_ee_quat,
                )
                pre_place_pos = np.array(semi_pre_pose["position"], dtype=float)
                pre_place_quat = normalize_ee_orientation_wxyz(
                    np.array(semi_pre_pose["orientation"], dtype=float),
                    env_ee_quat,
                    mode="place",
                )
                final_place_pos = np.array(semi_place_pose["position"], dtype=float) + place_offset
                final_place_quat = normalize_ee_orientation_wxyz(
                    np.array(semi_place_pose["orientation"], dtype=float),
                    env_ee_quat,
                    mode="place",
                )
                execution_mode = "semi_enclosed"
                enclosing_container_labels[env_idx] = container_name

        execution_modes[env_idx] = execution_mode
        pre_place_positions.append(pre_place_pos)
        pre_place_quaternions.append(pre_place_quat)
        all_place_pos[env_idx] = final_place_pos
        all_place_quat[env_idx] = final_place_quat

    context["place_pose"] = {
        "mode": requested_mode.lower(),
        "target_label": container_label,
        "target_available_mask": list(target_available_mask),
        "execution_modes": list(execution_modes),
        "enclosing_container_labels": list(enclosing_container_labels),
        "pre_positions": np.asarray(pre_place_positions, dtype=float),
        "pre_orientations": np.asarray(pre_place_quaternions, dtype=float),
        "positions": np.asarray(all_place_pos, dtype=float),
        "orientations": np.asarray(all_place_quat, dtype=float),
    }
    semi_count = sum(mode_name == "semi_enclosed" for mode_name in execution_modes)
    print(
        f"  📦 [Primitive] Computed place pose for mode='{requested_mode.lower()}', target='{container_label}', "
        f"semi_enclosed_envs={semi_count}/{collector.num_envs}"
    )
    return True


def primitive_execute_place(collector, params: Dict[str, Any], context: Dict[str, Any]) -> bool:
    place_pose = context.get("place_pose")
    if place_pose is None:
        print("[ERROR] Primitive 'execute_place' requires place poses first.")
        return False

    n = collector.num_envs
    prev_alive = get_prev_alive_mask(context, n)
    local_active = init_local_active_mask(context, n)
    ik_interpolation_steps = int(params.get("ik_interpolation_steps", 30))
    target_available_mask = list(place_pose.get("target_available_mask", [True] * n))
    for env_idx in range(n):
        if prev_alive[env_idx] and not target_available_mask[env_idx]:
            mark_local_inactive_with_reason(
                context, local_active, env_idx, FAILURE_REASON_TARGET_MISSING
            )
    execution_modes = list(place_pose.get("execution_modes", ["normal"] * n))
    pre_place_positions = np.array(place_pose.get("pre_positions", place_pose["positions"]), dtype=float)
    pre_place_quaternions = np.array(place_pose.get("pre_orientations", place_pose["orientations"]), dtype=float)
    place_positions = np.array(place_pose["positions"], dtype=float)
    place_quaternions = np.array(place_pose["orientations"])

    two_step_modes = {"semi_enclosed", "affordance"}
    two_step_active_mask = [
        local_active[env_idx] and execution_modes[env_idx] in two_step_modes
        for env_idx in range(n)
    ]
    normal_active_mask = [
        local_active[env_idx] and execution_modes[env_idx] not in two_step_modes
        for env_idx in range(n)
    ]

    if any(two_step_active_mask):
        two_step_mode_names = sorted({execution_modes[env_idx] for env_idx, active in enumerate(two_step_active_mask) if active})
        print(
            f"  🚀 [Primitive] Executing two-step PLACE ({','.join(two_step_mode_names)}) "
            f"for {sum(two_step_active_mask)}/{n} envs"
        )
        two_step_indices = [env_idx for env_idx, active in enumerate(two_step_active_mask) if active]
        subset_pre_place_plans, subset_pre_place_successes = collector.motion_planner.plan_to_pose_batch(
            collector.robot.data.joint_pos[two_step_indices].cpu().numpy(),
            collector.robot.data.joint_vel[two_step_indices].cpu().numpy(),
            collector.robot.joint_names,
            pre_place_positions[two_step_indices],
            pre_place_quaternions[two_step_indices],
            env_indices=two_step_indices,
            target_prim_path=collector.target_prim_path,
        )
        pre_place_plans, pre_place_successes = _scatter_subset_plans(
            n,
            two_step_indices,
            subset_pre_place_plans,
            subset_pre_place_successes,
        )
        # Only fold pre-place planning failures for envs that took the two-step branch.
        two_step_pre_segment = [
            (not two_step_active_mask[i]) or bool(pre_place_successes[i]) for i in range(n)
        ]
        merge_segment_into_local_active(local_active, two_step_pre_segment)
        active_for_pre = [two_step_active_mask[i] and local_active[i] for i in range(n)]
        collector.execute_trajectory_batch(
            pre_place_plans,
            [-1.0 if s else 0.0 for s in active_for_pre],
            desc="Primitive Pre-Place",
            active_mask=active_for_pre,
        )

        active_for_descend = [two_step_active_mask[i] and local_active[i] for i in range(n)]
        _, two_step_place_successes = collector.execute_ik_motion(
            target_positions=place_positions,
            target_quaternions=place_quaternions,
            gripper_states=[-1.0 if s else 0.0 for s in active_for_descend],
            interpolation_steps=ik_interpolation_steps,
            desc="Primitive Place IK Descend",
            active_mask=active_for_descend,
        )
        two_step_descend_segment = [
            (not two_step_active_mask[i]) or bool(two_step_place_successes[i]) for i in range(n)
        ]
        merge_segment_into_local_active(local_active, two_step_descend_segment)

    if any(normal_active_mask):
        print(f"  🚀 [Primitive] Executing normal PLACE for {sum(normal_active_mask)}/{n} envs")
        normal_indices = [env_idx for env_idx, active in enumerate(normal_active_mask) if active]
        subset_place_plans, subset_place_successes = collector.motion_planner.plan_to_pose_batch(
            collector.robot.data.joint_pos[normal_indices].cpu().numpy(),
            collector.robot.data.joint_vel[normal_indices].cpu().numpy(),
            collector.robot.joint_names,
            place_positions[normal_indices],
            place_quaternions[normal_indices],
            env_indices=normal_indices,
            target_prim_path=collector.target_prim_path,
        )
        place_plans, place_successes = _scatter_subset_plans(
            n,
            normal_indices,
            subset_place_plans,
            subset_place_successes,
        )
        normal_segment = [
            (not normal_active_mask[i]) or bool(place_successes[i]) for i in range(n)
        ]
        merge_segment_into_local_active(local_active, normal_segment)
        active_for_normal = [normal_active_mask[i] and local_active[i] for i in range(n)]
        collector.execute_trajectory_batch(
            place_plans,
            [-1.0 if s else 0.0 for s in active_for_normal],
            desc="Primitive Placing",
            active_mask=active_for_normal,
        )

    collector.control_gripper_batch(
        [1.0 if s else 0.0 for s in local_active],
        steps=FIXED_GRIPPER_STEPS,
        active_mask=list(local_active),
    )

    success_eval_enabled = bool(params.get("success_eval", False))
    post_release_lift_offset = parse_position_offset(params.get("post_release_lift_offset"), [0.0, 0.0, 0.08])
    post_release_retreat_positions = np.array(place_positions, dtype=float).copy()
    post_release_retreat_quaternions = np.array(place_quaternions, dtype=float).copy()
    retreat_to_pre_place_count = 0
    vertical_retreat_count = 0
    retreat_modes = {"semi_enclosed", "affordance"}
    for env_idx in range(n):
        if execution_modes[env_idx] in retreat_modes:
            post_release_retreat_positions[env_idx] = np.array(pre_place_positions[env_idx], dtype=float)
            post_release_retreat_quaternions[env_idx] = np.array(pre_place_quaternions[env_idx], dtype=float)
            retreat_to_pre_place_count += 1
        else:
            post_release_retreat_positions[env_idx] += post_release_lift_offset
            vertical_retreat_count += 1

    if any(local_active):
        print(
            "  🚀 [Primitive] Post-release retreat: "
            f"pre_place={retreat_to_pre_place_count}/{n}, "
            f"vertical={vertical_retreat_count}/{n}"
        )
        _, post_release_retreat_successes = collector.execute_ik_motion(
            target_positions=post_release_retreat_positions,
            target_quaternions=post_release_retreat_quaternions,
            gripper_states=[1.0 if s else 0.0 for s in local_active],
            interpolation_steps=int(params.get("post_release_lift_steps", 10)),
            desc="Primitive Post-Release Retreat",
            active_mask=list(local_active),
        )
        merge_segment_into_local_active(local_active, post_release_retreat_successes)

    # ``local_active`` now reflects the cumulative planning chain for this primitive.
    plan_active_after = list(local_active)
    final_success_mask = list(local_active)

    if success_eval_enabled:
        pick_target = context.get("pick_target_label")
        place_target = place_pose.get("target_label")
        place_mode = str(place_pose.get("mode", "")).lower()
        require_gripper_release = _parse_bool_param(params.get("require_gripper_release"), True)
        gripper_open_threshold = float(
            params.get("gripper_open_threshold", PLACE_SUCCESS_GRIPPER_OPEN_THRESHOLD)
        )
        gripper_joint_name_substr = str(params.get("gripper_joint_name_substr", "finger,drive"))
        gripper_open_direction = str(params.get("gripper_open_direction", "auto"))
        if not pick_target:
            print("[ERROR] Place success_eval requires pick target.")
            commit_plan_failures(context, prev_alive, local_active)
            commit_task_failures(context, prev_alive, plan_active_after, [False] * n)
            context["post_place_success_mask"] = [False] * n
            context["env_alive_mask"] = [False] * n
            return False

        state_successes = []
        state_metrics = []
        held_object_offset_in_ee = list(context.get("held_object_offset_in_ee", [None] * collector.num_envs))
        affordance_pick_local = None
        affordance_target_local = None
        if place_mode == "affordance":
            cached_pick_affordance = place_pose.get("pick_affordance_local")
            cached_target_affordance = place_pose.get("target_affordance_local")
            if cached_pick_affordance is None:
                affordance_pick_local, _ = _load_scaled_place_affordance_centroid(
                    collector,
                    pick_target,
                    affordance_npz_path=place_pose.get("pick_affordance_npz_path"),
                )
            else:
                affordance_pick_local = np.asarray(cached_pick_affordance, dtype=float)

            if cached_target_affordance is None:
                affordance_target_local, _ = _load_scaled_place_affordance_centroid(
                    collector,
                    place_target,
                    affordance_npz_path=place_pose.get("target_affordance_npz_path"),
                )
            else:
                affordance_target_local = np.asarray(cached_target_affordance, dtype=float)

            if affordance_pick_local is None or affordance_target_local is None:
                print("[ERROR] Place affordance success_eval requires valid affordance centroids.")
                commit_plan_failures(context, prev_alive, local_active)
                commit_task_failures(context, prev_alive, plan_active_after, [False] * n)
                context["post_place_success_mask"] = [False] * n
                context["env_alive_mask"] = [False] * n
                return False

        execution_modes_list = place_pose.get("execution_modes", ["normal"] * n)
        if place_mode not in ("pose", "affordance") and not place_target:
            print("[ERROR] Place success_eval requires place target when mode is not 'pose'.")
            commit_plan_failures(context, prev_alive, local_active)
            commit_task_failures(context, prev_alive, plan_active_after, [False] * n)
            context["post_place_success_mask"] = [False] * n
            context["env_alive_mask"] = [False] * n
            return False

        for env_idx in range(n):
            if place_mode == "pose":
                env_success, env_metrics = _evaluate_place_pose_success_with_metrics(
                    scene=collector.scene,
                    pick_target=pick_target,
                    target_position_env=np.array(place_pose["positions"][env_idx], dtype=float),
                    target_quat_wxyz=np.array(place_pose["orientations"][env_idx], dtype=float),
                    pick_target_offset_in_ee=held_object_offset_in_ee[env_idx],
                    env_idx=env_idx,
                    require_gripper_release=require_gripper_release,
                    gripper_open_threshold=gripper_open_threshold,
                    gripper_joint_name_substr=gripper_joint_name_substr,
                    gripper_open_direction=gripper_open_direction,
                )
            elif place_mode == "affordance":
                env_success, env_metrics = _evaluate_place_affordance_success_with_metrics(
                    collector=collector,
                    pick_target=pick_target,
                    place_target=place_target,
                    pick_affordance_local=affordance_pick_local,
                    target_affordance_local=affordance_target_local,
                    env_idx=env_idx,
                    require_gripper_release=require_gripper_release,
                    gripper_open_threshold=gripper_open_threshold,
                    gripper_joint_name_substr=gripper_joint_name_substr,
                    gripper_open_direction=gripper_open_direction,
                )
            else:
                env_execution_mode = execution_modes_list[env_idx]
                env_offset_in_ee = held_object_offset_in_ee[env_idx]
                env_target_quat = np.array(place_pose["orientations"][env_idx], dtype=float)

                if env_execution_mode == "semi_enclosed":
                    env_success, env_metrics = _evaluate_place_pose_success_with_metrics(
                        scene=collector.scene,
                        pick_target=pick_target,
                        target_position_env=np.array(place_pose["positions"][env_idx], dtype=float),
                        target_quat_wxyz=env_target_quat,
                        pick_target_offset_in_ee=env_offset_in_ee,
                        env_idx=env_idx,
                        require_gripper_release=require_gripper_release,
                        gripper_open_threshold=gripper_open_threshold,
                        gripper_joint_name_substr=gripper_joint_name_substr,
                        gripper_open_direction=gripper_open_direction,
                    )
                else:
                    env_success, env_metrics = _evaluate_place_success_with_metrics(
                        scene=collector.scene,
                        pick_target=pick_target,
                        place_target=place_target,
                        env_idx=env_idx,
                        ee_target_quat_wxyz=env_target_quat,
                        pick_target_offset_in_ee=env_offset_in_ee,
                        require_gripper_release=require_gripper_release,
                        gripper_open_threshold=gripper_open_threshold,
                        gripper_joint_name_substr=gripper_joint_name_substr,
                        gripper_open_direction=gripper_open_direction,
                    )
            state_successes.append(env_success)
            state_metrics.append(env_metrics)

        # Task line: only judges envs that survived the planning chain.
        commit_task_failures(context, prev_alive, plan_active_after, list(state_successes))
        final_success_mask = [
            bool(plan_active_after[i] and state_successes[i]) for i in range(n)
        ]
        context["task_success_metrics"] = list(state_metrics)
        for env_idx, metrics in enumerate(state_metrics):
            gripper_metric = _format_gripper_release_metric(metrics)
            if place_mode == "pose":
                dist_ok = metrics["pick_pose_xy_dist"] <= float(PLACE_SUCCESS_XY_THRESHOLD)
                compensated = bool(metrics.get("pose_offset_compensated", 0.0) > 0.5)
                print(
                    f"    [TaskCheck][env {env_idx}] "
                    f"pick_pose_xy_dist={metrics['pick_pose_xy_dist']:.4f}m ({'OK' if dist_ok else 'FAIL'}), "
                    f"offset_compensated={'OK' if compensated else 'NO'}"
                    f"{gripper_metric}"
                )
            elif place_mode == "affordance":
                dist_ok = metrics["pick_place_affordance_dist"] <= float(PLACE_SUCCESS_XY_THRESHOLD)
                print(
                    f"    [TaskCheck][env {env_idx}] "
                    f"pick_place_affordance_dist={metrics['pick_place_affordance_dist']:.4f}m "
                    f"({'OK' if dist_ok else 'FAIL'})"
                    f"{gripper_metric}"
                )
            else:
                env_execution_mode = execution_modes_list[env_idx]
                if env_execution_mode == "semi_enclosed":
                    dist_ok = metrics["pick_pose_xy_dist"] <= float(PLACE_SUCCESS_XY_THRESHOLD)
                    compensated = bool(metrics.get("pose_offset_compensated", 0.0) > 0.5)
                    comp_tag = "compensated" if compensated else "raw"
                    print(
                        f"    [TaskCheck][env {env_idx}] "
                        f"pick_place_xy_dist(semi_enclosed,{comp_tag})="
                        f"{metrics['pick_pose_xy_dist']:.4f}m ({'OK' if dist_ok else 'FAIL'})"
                        f"{gripper_metric}"
                    )
                else:
                    target_pos, _ = collector._get_object_pose_global(pick_target, env_idx)
                    container_pos, _ = collector._get_object_pose_global(place_target, env_idx)
                    target_z = float(target_pos[2]) if target_pos is not None else float("nan")
                    container_z = float(container_pos[2]) if container_pos is not None else float("nan")
                    place_ok = bool(metrics["place_target_in_range"] > 0.5)
                    dist_ok = metrics["pick_place_xy_dist"] <= float(PLACE_SUCCESS_XY_THRESHOLD)
                    z_ok = bool(metrics["pick_higher_than_place"] > 0.5)
                    compensated = bool(metrics.get("place_target_offset_compensated", 0.0) > 0.5)
                    comp_tag = "compensated" if compensated else "raw"
                    print(
                        f"    [TaskCheck][env {env_idx}] "
                        f"place_target_in_range={'OK' if place_ok else 'FAIL'}, "
                        f"pick_place_xy_dist({comp_tag})={metrics['pick_place_xy_dist']:.4f}m "
                        f"({'OK' if dist_ok else 'FAIL'}), "
                        f"pick_z={target_z:.4f}, place_z={container_z:.4f} ({'OK' if z_ok else 'FAIL'})"
                        f"{gripper_metric}"
                    )
        gripper_criterion = ", gripper_released" if require_gripper_release else ""
        if place_mode == "pose":
            print(
                "  ✅ [Primitive] Task success (place/pose): "
                f"{sum(final_success_mask)}/{n} "
                f"(pick_pose_xy_dist<={PLACE_SUCCESS_XY_THRESHOLD:.3f}{gripper_criterion})"
            )
        elif place_mode == "affordance":
            print(
                "  ✅ [Primitive] Task success (place/affordance): "
                f"{sum(final_success_mask)}/{n} "
                f"(pick_place_affordance_dist<={PLACE_SUCCESS_XY_THRESHOLD:.3f}{gripper_criterion})"
            )
        else:
            print(
                "  ✅ [Primitive] Task success (place): "
                f"{sum(final_success_mask)}/{n} "
                f"(place_target_in_range, pick_place_xy_dist<={PLACE_SUCCESS_XY_THRESHOLD:.3f}, "
                f"pick_z>=place_z{gripper_criterion})"
            )

    commit_plan_failures(context, prev_alive, local_active)

    plan_mask_final = list(context.get("plan_success_mask", [True] * n))
    task_mask_final = list(context.get("task_success_mask", [True] * n))
    final_alive = [
        bool(prev_alive[i] and plan_mask_final[i] and task_mask_final[i])
        for i in range(n)
    ]
    context["post_place_success_mask"] = list(final_alive)
    context["env_alive_mask"] = list(final_alive)
    context["current_gripper_state"] = "open"
    context["place_completed"] = any(final_alive)
    context["primitive_executed_motion"] = True

    print(f"  ✅ [Primitive] Place success (alive): {sum(final_alive)}/{n}")
    return any(final_alive)
