import numpy as np

from ..tools.pose_utils import _quat_wxyz_to_matrix
from .constants import (
    CLOSE_SUCCESS_MAX_ANGLE_RAD,
    HANG_SUCCESS_MIN_Z,
    HANG_SUCCESS_XY_THRESHOLD,
    HOLD_SUCCESS_MIN_Z,
    OPEN_SUCCESS_MIN_ANGLE_RAD,
    PICK_SUCCESS_LIFT_THRESHOLD,
    PLACE_SUCCESS_GRIPPER_OPEN_THRESHOLD,
    PLACE_SUCCESS_XY_THRESHOLD,
    PLACE_SUCCESS_Z_MARGIN,
    POSITION_RANGE_TOLERANCE,
    POUR_PROCESS_TILT_MONOTONIC_SLACK,
    POUR_PROCESS_XY_TOLERANCE,
    POUR_SUCCESS_TILT_THRESHOLD,
    PULL_SUCCESS_MIN_DISPLACEMENT,
    PUSH_SUCCESS_TILT_THRESHOLD,
    PUSH_SUCCESS_XY_THRESHOLD,
)
from .scene_helpers import (
    _get_articulation_all_joint_pos,
    _get_articulation_joint_pos,
    _get_gripper_max_finger_pos,
    _get_object_pose_env_from_scene,
    _get_object_pose_world_from_scene,
    _get_position_range,
    _is_position_in_range,
)


def _check_gripper_released(
    scene,
    env_idx: int = 0,
    threshold: float = PLACE_SUCCESS_GRIPPER_OPEN_THRESHOLD,
    joint_name_substr: str = "finger,drive",
    open_direction: str = "auto",
    robot_label: str = "robot",
) -> tuple[bool, float, str]:
    """Return whether the robot gripper is released, plus measured joint position."""
    pos, matched_names = _get_gripper_max_finger_pos(
        scene,
        env_idx=env_idx,
        robot_label=robot_label,
        joint_name_substr=joint_name_substr,
    )
    if pos is None:
        return False, float("nan"), "missing"

    direction = str(open_direction or "auto").strip().lower()
    if direction == "auto":
        names_text = " ".join(str(name).lower() for name in matched_names)
        direction = "ge" if ("fr3" in names_text or "panda" in names_text) else "le"

    if direction in {"ge", ">=", "greater", "greater_equal"}:
        released = pos >= float(threshold)
        direction_tag = "ge"
    elif direction in {"le", "<=", "less", "less_equal"}:
        released = pos <= float(threshold)
        direction_tag = "le"
    else:
        released = pos >= float(threshold)
        direction_tag = "ge"
    return bool(released), float(pos), direction_tag


def _merge_gripper_release_metrics(
    metrics: dict[str, float],
    scene,
    env_idx: int,
    require_gripper_release: bool,
    gripper_open_threshold: float,
    gripper_joint_name_substr: str,
    gripper_open_direction: str,
) -> tuple[bool, dict[str, float]]:
    if not require_gripper_release:
        metrics.update(
            {
                "gripper_release_required": 0.0,
                "gripper_released": 1.0,
                "gripper_finger_pos": float("nan"),
                "gripper_open_threshold": float(gripper_open_threshold),
                "gripper_open_direction": "disabled",
            }
        )
        return True, metrics

    released, finger_pos, direction = _check_gripper_released(
        scene,
        env_idx=env_idx,
        threshold=gripper_open_threshold,
        joint_name_substr=gripper_joint_name_substr,
        open_direction=gripper_open_direction,
    )
    metrics.update(
        {
            "gripper_release_required": 1.0,
            "gripper_released": 1.0 if released else 0.0,
            "gripper_finger_pos": finger_pos,
            "gripper_open_threshold": float(gripper_open_threshold),
            "gripper_open_direction": direction,
        }
    )
    return released, metrics


def _evaluate_place_success_with_metrics(
    scene,
    pick_target: str,
    place_target: str,
    env_idx: int = 0,
    xy_threshold: float = PLACE_SUCCESS_XY_THRESHOLD,
    z_margin: float = PLACE_SUCCESS_Z_MARGIN,
    range_tolerance: float = POSITION_RANGE_TOLERANCE,
    ee_target_quat_wxyz: np.ndarray | None = None,
    pick_target_offset_in_ee: np.ndarray | None = None,
    require_gripper_release: bool = True,
    gripper_open_threshold: float = PLACE_SUCCESS_GRIPPER_OPEN_THRESHOLD,
    gripper_joint_name_substr: str = "finger,drive",
    gripper_open_direction: str = "auto",
) -> tuple[bool, dict[str, float]]:
    pick_pos_world, _ = _get_object_pose_world_from_scene(scene, pick_target, env_idx)
    place_pos_world, _ = _get_object_pose_world_from_scene(scene, place_target, env_idx)
    place_pos_env, _ = _get_object_pose_env_from_scene(scene, place_target, env_idx)

    if pick_pos_world is None or place_pos_world is None or place_pos_env is None:
        _, metrics = _merge_gripper_release_metrics(
            {
                "pick_place_xy_dist": float("inf"),
                "pick_higher_than_place": 0.0,
                "place_target_in_range": 0.0,
                "place_target_offset_compensated": 0.0,
            },
            scene,
            env_idx,
            require_gripper_release,
            gripper_open_threshold,
            gripper_joint_name_substr,
            gripper_open_direction,
        )
        return False, metrics

    offset_compensated = False
    if ee_target_quat_wxyz is not None and pick_target_offset_in_ee is not None:
        rot_ee = _quat_wxyz_to_matrix(np.array(ee_target_quat_wxyz, dtype=float))
        offset_world = rot_ee @ np.array(pick_target_offset_in_ee, dtype=float)
        compensated_target_xy = place_pos_world[:2] + offset_world[:2]
        offset_compensated = True
    else:
        compensated_target_xy = place_pos_world[:2]

    place_in_range = _is_position_in_range(
        place_pos_env,
        _get_position_range(scene, place_target),
        tolerance=range_tolerance,
        axes=("x", "y"),
    )
    xy_dist = float(np.linalg.norm(pick_pos_world[:2] - compensated_target_xy))
    z_ok = bool(float(pick_pos_world[2]) >= float(place_pos_world[2]) + float(z_margin))
    gripper_ok, metrics = _merge_gripper_release_metrics(
        {
            "pick_place_xy_dist": xy_dist,
            "pick_higher_than_place": 1.0 if z_ok else 0.0,
            "place_target_in_range": 1.0 if place_in_range else 0.0,
            "place_target_offset_compensated": 1.0 if offset_compensated else 0.0,
        },
        scene,
        env_idx,
        require_gripper_release,
        gripper_open_threshold,
        gripper_joint_name_substr,
        gripper_open_direction,
    )
    success = place_in_range and xy_dist <= float(xy_threshold) and z_ok and gripper_ok
    return success, metrics


def _evaluate_place_on_success_with_metrics(
    scene,
    pick_target: str,
    place_target: str,
    place_target_height: float = 0.1,
    on_height_margin: float = 0.01,
    env_idx: int = 0,
    dist_threshold: float = 0.08,
    require_gripper_release: bool = True,
    gripper_open_threshold: float = PLACE_SUCCESS_GRIPPER_OPEN_THRESHOLD,
    gripper_joint_name_substr: str = "finger,drive",
    gripper_open_direction: str = "auto",
) -> tuple[bool, dict[str, float]]:
    """Evaluate 'on' mode placement using 3D distance to expected position.

    Expected position = place_target origin + [0, 0, height/2 + on_height_margin].
    """
    pick_pos_w, _ = _get_object_pose_world_from_scene(scene, pick_target, env_idx)
    place_pos_w, _ = _get_object_pose_world_from_scene(scene, place_target, env_idx)
    if pick_pos_w is None or place_pos_w is None:
        _, metrics = _merge_gripper_release_metrics(
            {
                "pick_place_on_dist_3d": float("inf"),
                "pick_pos_world": None,
                "expected_place_pos": None,
            },
            scene,
            env_idx,
            require_gripper_release,
            gripper_open_threshold,
            gripper_joint_name_substr,
            gripper_open_direction,
        )
        return False, metrics
    expected_z = float(place_pos_w[2]) + place_target_height / 2.0 + on_height_margin
    expected_pos = np.array([float(place_pos_w[0]), float(place_pos_w[1]), expected_z], dtype=np.float64)
    pick_pos = np.asarray(pick_pos_w, dtype=np.float64)
    dist_3d = float(np.linalg.norm(pick_pos - expected_pos))
    gripper_ok, metrics = _merge_gripper_release_metrics(
        {
            "pick_place_on_dist_3d": dist_3d,
            "pick_pos_world": pick_pos.tolist(),
            "expected_place_pos": expected_pos.tolist(),
        },
        scene,
        env_idx,
        require_gripper_release,
        gripper_open_threshold,
        gripper_joint_name_substr,
        gripper_open_direction,
    )
    return dist_3d <= dist_threshold and gripper_ok, metrics


def _evaluate_place_affordance_xy_success_with_metrics(
    scene,
    pick_target: str,
    place_target: str,
    pick_aff_local: np.ndarray,
    target_aff_local: np.ndarray,
    env_idx: int = 0,
    xy_threshold: float = 0.10,
    require_gripper_release: bool = True,
    gripper_open_threshold: float = PLACE_SUCCESS_GRIPPER_OPEN_THRESHOLD,
    gripper_joint_name_substr: str = "finger,drive",
    gripper_open_direction: str = "auto",
) -> tuple[bool, dict[str, float]]:
    """Evaluate affordance-mode placement by XY distance between affordance centroids."""
    pick_pos_w, pick_quat_w = _get_object_pose_world_from_scene(scene, pick_target, env_idx)
    place_pos_w, place_quat_w = _get_object_pose_world_from_scene(scene, place_target, env_idx)
    if pick_pos_w is None or pick_quat_w is None or place_pos_w is None or place_quat_w is None:
        _, metrics = _merge_gripper_release_metrics(
            {
                "pick_place_affordance_xy_dist": float("inf"),
                "pick_aff_world": None,
                "target_aff_world": None,
            },
            scene,
            env_idx,
            require_gripper_release,
            gripper_open_threshold,
            gripper_joint_name_substr,
            gripper_open_direction,
        )
        return False, metrics
    rot_pick = _quat_wxyz_to_matrix(np.asarray(pick_quat_w, dtype=np.float64))
    rot_place = _quat_wxyz_to_matrix(np.asarray(place_quat_w, dtype=np.float64))
    pick_aff_world = rot_pick @ np.asarray(pick_aff_local, dtype=np.float64) + np.asarray(pick_pos_w, dtype=np.float64)
    target_aff_world = rot_place @ np.asarray(target_aff_local, dtype=np.float64) + np.asarray(place_pos_w, dtype=np.float64)
    xy_dist = float(np.linalg.norm(pick_aff_world[:2] - target_aff_world[:2]))
    gripper_ok, metrics = _merge_gripper_release_metrics(
        {
            "pick_place_affordance_xy_dist": xy_dist,
            "pick_aff_world": pick_aff_world.tolist(),
            "target_aff_world": target_aff_world.tolist(),
        },
        scene,
        env_idx,
        require_gripper_release,
        gripper_open_threshold,
        gripper_joint_name_substr,
        gripper_open_direction,
    )
    return xy_dist <= float(xy_threshold) and gripper_ok, metrics


def _evaluate_place_pose_success_with_metrics(
    scene,
    pick_target: str,
    target_position_env: np.ndarray,
    target_quat_wxyz: np.ndarray | None = None,
    pick_target_offset_in_ee: np.ndarray | None = None,
    env_idx: int = 0,
    xy_threshold: float = PLACE_SUCCESS_XY_THRESHOLD,
    require_gripper_release: bool = True,
    gripper_open_threshold: float = PLACE_SUCCESS_GRIPPER_OPEN_THRESHOLD,
    gripper_joint_name_substr: str = "finger,drive",
    gripper_open_direction: str = "auto",
) -> tuple[bool, dict[str, float]]:
    pick_pos_env, _ = _get_object_pose_env_from_scene(scene, pick_target, env_idx)
    if pick_pos_env is None:
        _, metrics = _merge_gripper_release_metrics(
            {
                "pick_pose_xy_dist": float("inf"),
                "pose_offset_compensated": 0.0,
            },
            scene,
            env_idx,
            require_gripper_release,
            gripper_open_threshold,
            gripper_joint_name_substr,
            gripper_open_direction,
        )
        return False, metrics

    target_position_env = np.array(target_position_env, dtype=float)
    offset_compensated = False
    if pick_target_offset_in_ee is not None and target_quat_wxyz is not None:
        target_rot = _quat_wxyz_to_matrix(np.array(target_quat_wxyz, dtype=float))
        target_position_env = target_position_env + target_rot @ np.array(pick_target_offset_in_ee, dtype=float)
        offset_compensated = True

    xy_dist = float(np.linalg.norm(pick_pos_env[:2] - target_position_env[:2]))
    gripper_ok, metrics = _merge_gripper_release_metrics(
        {
            "pick_pose_xy_dist": xy_dist,
            "pose_offset_compensated": 1.0 if offset_compensated else 0.0,
        },
        scene,
        env_idx,
        require_gripper_release,
        gripper_open_threshold,
        gripper_joint_name_substr,
        gripper_open_direction,
    )
    success = xy_dist <= float(xy_threshold) and gripper_ok
    return success, metrics


def _evaluate_pour_success_with_metrics(
    scene,
    pick_target: str,
    pour_target: str,
    env_idx: int = 0,
    tilt_threshold: float = POUR_SUCCESS_TILT_THRESHOLD,
    range_tolerance: float = POSITION_RANGE_TOLERANCE,
) -> tuple[bool, dict[str, float]]:
    _, pick_quat = _get_object_pose_world_from_scene(scene, pick_target, env_idx)
    pour_pos_env, _ = _get_object_pose_env_from_scene(scene, pour_target, env_idx)

    if pick_quat is None or pour_pos_env is None:
        return False, {
            "object_local_z_world_z": float("nan"),
            "pour_target_in_range": 0.0,
        }

    rot = _quat_wxyz_to_matrix(np.array(pick_quat, dtype=float))
    world_z_component = float(rot[2, 2])
    pour_in_range = _is_position_in_range(
        pour_pos_env,
        _get_position_range(scene, pour_target),
        tolerance=range_tolerance,
        axes=("x", "y"),
    )
    success = pour_in_range and (world_z_component <= float(tilt_threshold))
    return success, {
        "object_local_z_world_z": world_z_component,
        "pour_target_in_range": 1.0 if pour_in_range else 0.0,
    }


def _evaluate_pour_process_with_metrics(
    per_step_trajectory: list[dict],
    pour_center_world_xy: np.ndarray,
    tilt_horizontal_offset: float,
    xy_tolerance: float = POUR_PROCESS_XY_TOLERANCE,
    tilt_monotonic_slack: float = POUR_PROCESS_TILT_MONOTONIC_SLACK,
) -> tuple[bool, dict[str, float]]:
    """Evaluate pour *process* quality by checking per-step tilt trajectory.

    Two conditions:
      1) pour tip world XY stays within ``tilt_horizontal_offset + xy_tolerance``
         of the container pour center.
      2) object local +z in world z (``z_axis_world_z``) is (near-)monotonically
         decreasing (slack allowed), so the pour is a continuous tilt rather
         than a discontinuous flick. Total tilt magnitude is NOT checked here -
         the terminal check ``POUR_SUCCESS_TILT_THRESHOLD`` already enforces
         that the final tilt angle is at least 90 deg.
    """
    nan = float("nan")
    metrics = {
        "max_xy_excess": nan,
        "max_tilt_reverse": nan,
        "total_tilt_decrease": nan,
        "pour_tip_in_range": 0.0,
        "pour_tilt_monotonic": 0.0,
    }
    if not per_step_trajectory:
        return False, metrics

    pour_center_world_xy = np.asarray(pour_center_world_xy, dtype=float)
    allowed_radius = float(tilt_horizontal_offset) + float(xy_tolerance)

    max_xy_excess = -float("inf")
    for sample in per_step_trajectory:
        tip_xy = np.asarray(sample.get("tip_xy"), dtype=float)
        if tip_xy.shape != (2,) or not np.all(np.isfinite(tip_xy)):
            metrics["max_xy_excess"] = float("inf")
            return False, metrics
        dist = float(np.linalg.norm(tip_xy - pour_center_world_xy))
        excess = dist - allowed_radius
        if excess > max_xy_excess:
            max_xy_excess = excess
    metrics["max_xy_excess"] = max_xy_excess
    range_ok = max_xy_excess <= 0.0
    metrics["pour_tip_in_range"] = 1.0 if range_ok else 0.0

    z_values = [float(sample.get("z_axis_world_z", float("nan"))) for sample in per_step_trajectory]
    if any(not np.isfinite(z) for z in z_values) or len(z_values) < 2:
        metrics["max_tilt_reverse"] = float("inf")
        metrics["total_tilt_decrease"] = nan
        return False, metrics

    max_tilt_reverse = 0.0
    for prev_z, cur_z in zip(z_values[:-1], z_values[1:]):
        delta = cur_z - prev_z  # >0 means tilt reversed (z went up again)
        if delta > max_tilt_reverse:
            max_tilt_reverse = delta
    total_tilt_decrease = z_values[0] - z_values[-1]
    metrics["max_tilt_reverse"] = max_tilt_reverse
    metrics["total_tilt_decrease"] = total_tilt_decrease  # informational only

    monotonic_ok = max_tilt_reverse <= float(tilt_monotonic_slack)
    metrics["pour_tilt_monotonic"] = 1.0 if monotonic_ok else 0.0

    return bool(range_ok and monotonic_ok), metrics


def _evaluate_push_success_with_metrics(
    scene,
    push_target_label: str,
    push_target_position_env: np.ndarray,
    env_idx: int = 0,
    xy_threshold: float = PUSH_SUCCESS_XY_THRESHOLD,
    tilt_threshold: float = PUSH_SUCCESS_TILT_THRESHOLD,
) -> tuple[bool, dict[str, float]]:
    obj_pos_env, _ = _get_object_pose_env_from_scene(scene, push_target_label, env_idx)
    _, obj_quat_world = _get_object_pose_world_from_scene(scene, push_target_label, env_idx)
    if obj_pos_env is None or obj_quat_world is None:
        return False, {
            "push_xy_dist": float("inf"),
            "object_local_z_world_z": float("nan"),
        }

    push_target_position_env = np.array(push_target_position_env, dtype=float)
    xy_dist = float(np.linalg.norm(obj_pos_env[:2] - push_target_position_env[:2]))

    rot = _quat_wxyz_to_matrix(np.array(obj_quat_world, dtype=float))
    world_z_component = float(rot[2, 2])

    dist_ok = xy_dist <= float(xy_threshold)
    upright_ok = world_z_component >= float(tilt_threshold)
    success = dist_ok and upright_ok
    return success, {
        "push_xy_dist": xy_dist,
        "object_local_z_world_z": world_z_component,
    }


def _evaluate_open_success_with_metrics(
    scene,
    target_label: str,
    joint_name: str | None = None,
    min_angle_rad: float = OPEN_SUCCESS_MIN_ANGLE_RAD,
    env_idx: int = 0,
    range_tolerance: float = POSITION_RANGE_TOLERANCE,
) -> tuple[bool, dict[str, float]]:
    joint_pos = _get_articulation_joint_pos(scene, target_label, joint_name, env_idx)
    art_pos_env, _ = _get_object_pose_env_from_scene(scene, target_label, env_idx)

    if joint_pos is None:
        return False, {
            "joint_pos_rad": float("nan"),
            "articulation_in_range": 0.0,
        }

    in_range = _is_position_in_range(
        art_pos_env,
        _get_position_range(scene, target_label),
        tolerance=range_tolerance,
        axes=("x", "y"),
    )
    angle_ok = abs(joint_pos) >= float(min_angle_rad)
    success = angle_ok and in_range
    return success, {
        "joint_pos_rad": joint_pos,
        "articulation_in_range": 1.0 if in_range else 0.0,
    }


def _evaluate_pull_success_with_metrics(
    scene,
    target_label: str,
    joint_name: str,
    min_displacement: float = PULL_SUCCESS_MIN_DISPLACEMENT,
    env_idx: int = 0,
    range_tolerance: float = POSITION_RANGE_TOLERANCE,
    peak_displacement: float | None = None,
) -> tuple[bool, dict[str, float]]:
    """pull :
    -  `target_label` articulation  YAML  `joint_name`  (prismatic, )
    -  ``peak_displacement`` (rollout  |joint_pos| ), ;
       pull .  scene  joint_pos.
    -  >= `min_displacement` 
    - articulation  position_range  ( open/close )
    """
    if not joint_name:
        raise ValueError("pull success evaluation requires an explicit joint_name")
    joint_pos = _get_articulation_joint_pos(scene, target_label, joint_name, env_idx)
    all_joints = _get_articulation_all_joint_pos(scene, target_label, env_idx) or {}
    art_pos_env, _ = _get_object_pose_env_from_scene(scene, target_label, env_idx)

    if joint_pos is None and peak_displacement is None:
        return False, {
            "joint_displacement_m": float("nan"),
            "joint_peak_displacement_m": float("nan"),
            "articulation_in_range": 0.0,
            "all_joints": all_joints,
        }

    current_disp = abs(float(joint_pos)) if joint_pos is not None else float("nan")
    if peak_displacement is not None:
        eval_disp = float(peak_displacement)
    else:
        eval_disp = current_disp

    in_range = _is_position_in_range(
        art_pos_env,
        _get_position_range(scene, target_label),
        tolerance=range_tolerance,
        axes=("x", "y"),
    )
    displacement_ok = eval_disp >= float(min_displacement)
    success = displacement_ok and in_range
    return success, {
        "joint_displacement_m": float(joint_pos) if joint_pos is not None else float("nan"),
        "joint_peak_displacement_m": eval_disp,
        "articulation_in_range": 1.0 if in_range else 0.0,
        "all_joints": all_joints,
    }


def _evaluate_close_success_with_metrics(
    scene,
    target_label: str,
    joint_name: str | None = None,
    max_angle_rad: float = CLOSE_SUCCESS_MAX_ANGLE_RAD,
    env_idx: int = 0,
    range_tolerance: float = POSITION_RANGE_TOLERANCE,
) -> tuple[bool, dict[str, float]]:
    joint_pos = _get_articulation_joint_pos(scene, target_label, joint_name, env_idx)
    art_pos_env, _ = _get_object_pose_env_from_scene(scene, target_label, env_idx)

    if joint_pos is None:
        return False, {
            "joint_pos_rad": float("nan"),
            "articulation_in_range": 0.0,
        }

    in_range = _is_position_in_range(
        art_pos_env,
        _get_position_range(scene, target_label),
        tolerance=range_tolerance,
        axes=("x", "y"),
    )
    angle_ok = abs(joint_pos) <= float(max_angle_rad)
    success = angle_ok and in_range
    return success, {
        "joint_pos_rad": joint_pos,
        "articulation_in_range": 1.0 if in_range else 0.0,
    }


def _evaluate_hang_success_with_metrics(
    scene,
    held_object_label: str,
    container_label: str,
    container_mount_local: np.ndarray,
    env_idx: int = 0,
    xy_threshold: float = HANG_SUCCESS_XY_THRESHOLD,
    min_z: float = HANG_SUCCESS_MIN_Z,
    range_tolerance: float = POSITION_RANGE_TOLERANCE,
) -> tuple[bool, dict[str, float]]:
    obj_pos_world, _ = _get_object_pose_world_from_scene(scene, held_object_label, env_idx)
    obj_pos_env, _ = _get_object_pose_env_from_scene(scene, held_object_label, env_idx)
    cont_pos_world, cont_quat = _get_object_pose_world_from_scene(scene, container_label, env_idx)
    cont_pos_env, _ = _get_object_pose_env_from_scene(scene, container_label, env_idx)

    if obj_pos_world is None or cont_pos_world is None or obj_pos_env is None or cont_pos_env is None:
        return False, {
            "held_object_z_env": float("nan"),
            "mount_xy_dist": float("inf"),
            "container_in_range": 0.0,
        }

    cont_pos_world = np.array(cont_pos_world, dtype=float)
    mount_local = np.array(container_mount_local, dtype=float)
    rot_cont = _quat_wxyz_to_matrix(np.array(cont_quat, dtype=float))
    mount_world = rot_cont @ mount_local + cont_pos_world
    midpoint_xy = (cont_pos_world[:2] + mount_world[:2]) / 2.0

    z_ok = float(obj_pos_env[2]) > float(min_z)
    mount_xy_dist = float(np.linalg.norm(obj_pos_world[:2] - midpoint_xy))
    cont_in_range = _is_position_in_range(
        cont_pos_env,
        _get_position_range(scene, container_label),
        tolerance=range_tolerance,
        axes=("x", "y"),
    )
    success = z_ok and mount_xy_dist <= float(xy_threshold) and cont_in_range
    return success, {
        "held_object_z_env": float(obj_pos_env[2]),
        "mount_xy_dist": mount_xy_dist,
        "container_in_range": 1.0 if cont_in_range else 0.0,
    }


def _evaluate_pick_success_with_metrics(
    scene,
    pick_target: str,
    initial_z_world: float,
    env_idx: int = 0,
    lift_threshold: float = PICK_SUCCESS_LIFT_THRESHOLD,
    peak_z_world: float | None = None,
) -> tuple[bool, dict[str, float]]:
    """Pick success.

     ``peak_z_world`` (rollout  z ), 
    :  (place )  pick
    .  scene  z ().
    """
    if peak_z_world is None:
        pick_pos_world, _ = _get_object_pose_world_from_scene(scene, pick_target, env_idx)
        if pick_pos_world is None:
            return False, {
                "pick_lift_delta_z": float("nan"),
                "pick_initial_z": float(initial_z_world),
                "pick_current_z": float("nan"),
            }
        eval_z = float(pick_pos_world[2])
    else:
        eval_z = float(peak_z_world)

    delta_z = eval_z - float(initial_z_world)
    success = delta_z >= float(lift_threshold)
    return success, {
        "pick_lift_delta_z": delta_z,
        "pick_initial_z": float(initial_z_world),
        "pick_current_z": eval_z,
    }


def _evaluate_hold_success_with_metrics(
    scene,
    held_object_label: str,
    env_idx: int = 0,
    min_z: float = HOLD_SUCCESS_MIN_Z,
) -> tuple[bool, dict[str, float]]:
    obj_pos_env, _ = _get_object_pose_env_from_scene(scene, held_object_label, env_idx)
    if obj_pos_env is None:
        return False, {"held_object_z_env": float("nan")}

    z_ok = float(obj_pos_env[2]) > float(min_z)
    return z_ok, {"held_object_z_env": float(obj_pos_env[2])}
