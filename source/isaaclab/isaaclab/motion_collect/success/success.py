import numpy as np

from .constants import (
    CLOSE_SUCCESS_MAX_ANGLE_RAD,
    HANG_SUCCESS_MIN_Z,
    HANG_SUCCESS_XY_THRESHOLD,
    HOLD_SUCCESS_MIN_Z,
    OPEN_SUCCESS_MIN_ANGLE_RAD,
    PICK_SUCCESS_LIFT_THRESHOLD,
    PLACE_SUCCESS_XY_THRESHOLD,
    PLACE_SUCCESS_Z_MARGIN,
    POSITION_RANGE_TOLERANCE,
    POUR_SUCCESS_TILT_THRESHOLD,
    PUSH_SUCCESS_TILT_THRESHOLD,
    PUSH_SUCCESS_XY_THRESHOLD,
)
from .evaluators import (
    _evaluate_close_success_with_metrics,
    _evaluate_hang_success_with_metrics,
    _evaluate_hold_success_with_metrics,
    _evaluate_open_success_with_metrics,
    _evaluate_pick_success_with_metrics,
    _evaluate_place_success_with_metrics,
    _evaluate_pour_success_with_metrics,
    _evaluate_push_success_with_metrics,
)
from .scene_helpers import (
    _get_object_pose_world_from_scene,
    _get_scene_num_envs,
    attach_scene_success_meta,
)

# Re-export for backwards compatibility
__all__ = ["attach_scene_success_meta"]


def evaluate_place_success(
    scene,
    pick_target: str,
    place_target: str,
    env_idx: int = 0,
    xy_threshold: float = PLACE_SUCCESS_XY_THRESHOLD,
    z_margin: float = PLACE_SUCCESS_Z_MARGIN,
    range_tolerance: float = POSITION_RANGE_TOLERANCE,
) -> bool:
    success, _ = _evaluate_place_success_with_metrics(
        scene=scene,
        pick_target=pick_target,
        place_target=place_target,
        env_idx=env_idx,
        xy_threshold=xy_threshold,
        z_margin=z_margin,
        range_tolerance=range_tolerance,
    )
    return success


def evaluate_place_success_batch(
    scene,
    pick_target: str,
    place_target: str,
    xy_threshold: float = PLACE_SUCCESS_XY_THRESHOLD,
    z_margin: float = PLACE_SUCCESS_Z_MARGIN,
    range_tolerance: float = POSITION_RANGE_TOLERANCE,
) -> list[bool]:
    return [
        evaluate_place_success(
            scene=scene,
            pick_target=pick_target,
            place_target=place_target,
            env_idx=env_idx,
            xy_threshold=xy_threshold,
            z_margin=z_margin,
            range_tolerance=range_tolerance,
        )
        for env_idx in range(_get_scene_num_envs(scene))
    ]


def evaluate_pour_success(
    scene,
    pick_target: str,
    pour_target: str,
    env_idx: int = 0,
    tilt_threshold: float = POUR_SUCCESS_TILT_THRESHOLD,
    range_tolerance: float = POSITION_RANGE_TOLERANCE,
) -> bool:
    success, _ = _evaluate_pour_success_with_metrics(
        scene=scene,
        pick_target=pick_target,
        pour_target=pour_target,
        env_idx=env_idx,
        tilt_threshold=tilt_threshold,
        range_tolerance=range_tolerance,
    )
    return success


def evaluate_pour_success_batch(
    scene,
    pick_target: str,
    pour_target: str,
    tilt_threshold: float = POUR_SUCCESS_TILT_THRESHOLD,
    range_tolerance: float = POSITION_RANGE_TOLERANCE,
) -> list[bool]:
    return [
        evaluate_pour_success(
            scene=scene,
            pick_target=pick_target,
            pour_target=pour_target,
            env_idx=env_idx,
            tilt_threshold=tilt_threshold,
            range_tolerance=range_tolerance,
        )
        for env_idx in range(_get_scene_num_envs(scene))
    ]


def evaluate_push_success(
    scene,
    push_target_label: str,
    push_target_position_env: np.ndarray,
    env_idx: int = 0,
    xy_threshold: float = PUSH_SUCCESS_XY_THRESHOLD,
    tilt_threshold: float = PUSH_SUCCESS_TILT_THRESHOLD,
) -> bool:
    success, _ = _evaluate_push_success_with_metrics(
        scene=scene,
        push_target_label=push_target_label,
        push_target_position_env=push_target_position_env,
        env_idx=env_idx,
        xy_threshold=xy_threshold,
        tilt_threshold=tilt_threshold,
    )
    return success


def evaluate_push_success_batch(
    scene,
    push_target_label: str,
    push_target_position_env: np.ndarray,
    xy_threshold: float = PUSH_SUCCESS_XY_THRESHOLD,
    tilt_threshold: float = PUSH_SUCCESS_TILT_THRESHOLD,
) -> list[bool]:
    return [
        evaluate_push_success(
            scene=scene,
            push_target_label=push_target_label,
            push_target_position_env=push_target_position_env,
            env_idx=env_idx,
            xy_threshold=xy_threshold,
            tilt_threshold=tilt_threshold,
        )
        for env_idx in range(_get_scene_num_envs(scene))
    ]


def evaluate_open_success(
    scene,
    target_label: str,
    joint_name: str | None = None,
    min_angle_rad: float = OPEN_SUCCESS_MIN_ANGLE_RAD,
    env_idx: int = 0,
    range_tolerance: float = POSITION_RANGE_TOLERANCE,
) -> bool:
    success, _ = _evaluate_open_success_with_metrics(
        scene=scene,
        target_label=target_label,
        joint_name=joint_name,
        min_angle_rad=min_angle_rad,
        env_idx=env_idx,
        range_tolerance=range_tolerance,
    )
    return success


def evaluate_open_success_batch(
    scene,
    target_label: str,
    joint_name: str | None = None,
    min_angle_rad: float = OPEN_SUCCESS_MIN_ANGLE_RAD,
    range_tolerance: float = POSITION_RANGE_TOLERANCE,
) -> list[bool]:
    return [
        evaluate_open_success(
            scene=scene,
            target_label=target_label,
            joint_name=joint_name,
            min_angle_rad=min_angle_rad,
            env_idx=env_idx,
            range_tolerance=range_tolerance,
        )
        for env_idx in range(_get_scene_num_envs(scene))
    ]


def evaluate_close_success(
    scene,
    target_label: str,
    joint_name: str | None = None,
    max_angle_rad: float = CLOSE_SUCCESS_MAX_ANGLE_RAD,
    env_idx: int = 0,
    range_tolerance: float = POSITION_RANGE_TOLERANCE,
) -> bool:
    success, _ = _evaluate_close_success_with_metrics(
        scene=scene,
        target_label=target_label,
        joint_name=joint_name,
        max_angle_rad=max_angle_rad,
        env_idx=env_idx,
        range_tolerance=range_tolerance,
    )
    return success


def evaluate_close_success_batch(
    scene,
    target_label: str,
    joint_name: str | None = None,
    max_angle_rad: float = CLOSE_SUCCESS_MAX_ANGLE_RAD,
    range_tolerance: float = POSITION_RANGE_TOLERANCE,
) -> list[bool]:
    return [
        evaluate_close_success(
            scene=scene,
            target_label=target_label,
            joint_name=joint_name,
            max_angle_rad=max_angle_rad,
            env_idx=env_idx,
            range_tolerance=range_tolerance,
        )
        for env_idx in range(_get_scene_num_envs(scene))
    ]


def evaluate_hang_success(
    scene,
    held_object_label: str,
    container_label: str,
    container_mount_local: np.ndarray,
    env_idx: int = 0,
    xy_threshold: float = HANG_SUCCESS_XY_THRESHOLD,
    min_z: float = HANG_SUCCESS_MIN_Z,
    range_tolerance: float = POSITION_RANGE_TOLERANCE,
) -> bool:
    success, _ = _evaluate_hang_success_with_metrics(
        scene=scene,
        held_object_label=held_object_label,
        container_label=container_label,
        container_mount_local=container_mount_local,
        env_idx=env_idx,
        xy_threshold=xy_threshold,
        min_z=min_z,
        range_tolerance=range_tolerance,
    )
    return success


def evaluate_hang_success_batch(
    scene,
    held_object_label: str,
    container_label: str,
    container_mount_local: np.ndarray,
    xy_threshold: float = HANG_SUCCESS_XY_THRESHOLD,
    min_z: float = HANG_SUCCESS_MIN_Z,
    range_tolerance: float = POSITION_RANGE_TOLERANCE,
) -> list[bool]:
    return [
        evaluate_hang_success(
            scene=scene,
            held_object_label=held_object_label,
            container_label=container_label,
            container_mount_local=container_mount_local,
            env_idx=env_idx,
            xy_threshold=xy_threshold,
            min_z=min_z,
            range_tolerance=range_tolerance,
        )
        for env_idx in range(_get_scene_num_envs(scene))
    ]


def evaluate_hold_success(
    scene,
    held_object_label: str,
    env_idx: int = 0,
    min_z: float = HOLD_SUCCESS_MIN_Z,
) -> bool:
    success, _ = _evaluate_hold_success_with_metrics(
        scene=scene,
        held_object_label=held_object_label,
        env_idx=env_idx,
        min_z=min_z,
    )
    return success


def evaluate_hold_success_batch(
    scene,
    held_object_label: str,
    min_z: float = HOLD_SUCCESS_MIN_Z,
) -> list[bool]:
    return [
        evaluate_hold_success(
            scene=scene,
            held_object_label=held_object_label,
            env_idx=env_idx,
            min_z=min_z,
        )
        for env_idx in range(_get_scene_num_envs(scene))
    ]


def capture_object_initial_z(scene, label: str, env_idx: int = 0) -> float | None:
    """Helper to record an object's world-frame z right after env.reset()."""
    pos_world, _ = _get_object_pose_world_from_scene(scene, label, env_idx)
    if pos_world is None:
        return None
    return float(pos_world[2])


def evaluate_pick_success(
    scene,
    pick_target: str,
    initial_z_world: float,
    env_idx: int = 0,
    lift_threshold: float = PICK_SUCCESS_LIFT_THRESHOLD,
) -> bool:
    """Pick is considered successful when the target object's world-frame z has
    risen by at least ``lift_threshold`` relative to its episode-start position.

    The caller is responsible for capturing ``initial_z_world`` right after the
    environment reset (before any policy action is applied).
    """
    success, _ = _evaluate_pick_success_with_metrics(
        scene=scene,
        pick_target=pick_target,
        initial_z_world=initial_z_world,
        env_idx=env_idx,
        lift_threshold=lift_threshold,
    )
    return success


def evaluate_pick_success_batch(
    scene,
    pick_target: str,
    initial_z_world_per_env: list[float],
    lift_threshold: float = PICK_SUCCESS_LIFT_THRESHOLD,
) -> list[bool]:
    return [
        evaluate_pick_success(
            scene=scene,
            pick_target=pick_target,
            initial_z_world=initial_z_world_per_env[env_idx],
            env_idx=env_idx,
            lift_threshold=lift_threshold,
        )
        for env_idx in range(_get_scene_num_envs(scene))
    ]
