from typing import Any, Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..success.constants import CLOSE_SUCCESS_JOINT_NAME, CLOSE_SUCCESS_MAX_ANGLE_RAD
from ..success.evaluators import _evaluate_close_success_with_metrics
from ..tools.pose_utils import parse_facing_axis_local
from .utils import compute_target_availability_mask, log_target_missing_envs

FIXED_GRIPPER_STEPS = 10


def primitive_close(collector, mode: Optional[str], params: Dict[str, Any], context: Dict[str, Any]) -> bool:
    """Microwave close primitive driven by a pre-close CuRobo waypoint and a single IK push."""
    del mode
    return primitive_execute_close(collector, params, context)


def primitive_execute_close(collector, params: Dict[str, Any], context: Dict[str, Any]) -> bool:
    target_label = params.get("target")
    if not target_label:
        print("[ERROR] Primitive 'close' requires a target articulation.")
        return False

    prev_success_mask = list(context.get("env_alive_mask", [True] * collector.num_envs))
    target_available_mask = compute_target_availability_mask(collector, target_label)
    log_target_missing_envs("close", target_label, target_available_mask, prev_success_mask)

    if not hasattr(collector, "_containers_registry"):
        collector._containers_registry = collector._load_containers_registry()
    container_cfg = (collector._containers_registry or {}).get(target_label)
    if container_cfg is None:
        print(
            f"[ERROR] Primitive 'close' requires target '{target_label}' to be registered in containers.yaml "
            "with `facing_axis_local`."
        )
        return False

    try:
        facing_axis_local = parse_facing_axis_local(container_cfg.get("facing_axis_local"))
    except ValueError as exc:
        print(f"[ERROR] Invalid facing_axis_local for close target '{target_label}': {exc}")
        return False
    if facing_axis_local is None:
        print(
            f"[ERROR] Primitive 'close' requires '{target_label}' to define `facing_axis_local` in containers.yaml."
        )
        return False
    facing_axis_local = np.asarray(facing_axis_local, dtype=np.float64)

    pre_close_facing_offset = float(params.get("pre_close_facing_offset", 0.25))
    pre_close_open_normal_offset = float(params.get("pre_close_open_normal_offset", 0.40))
    close_facing_offset = float(params.get("close_facing_offset", 0.15))
    close_open_normal_backoff = float(params.get("close_open_normal_backoff", 0.15))
    close_height_offset = float(params.get("close_height_offset", 0.0))
    close_interp_steps = int(params.get("close_interpolation_steps", 30))

    object_key = collector._find_object_in_scene(target_label)
    if object_key is None or object_key not in collector.scene.articulations:
        print(f"[ERROR] Articulation '{target_label}' not found in scene.")
        _set_failure_context(context, collector.num_envs, prev_success_mask, target_available_mask)
        return False

    microwave = collector.scene.articulations[object_key]
    slide_rot_world = R.from_rotvec(-0.5 * np.pi * np.array([0.0, 0.0, 1.0], dtype=np.float64))
    env_origins = collector._get_env_origins()

    pre_close_positions = []
    close_positions = []
    close_quaternions = []
    active_targets = []

    for env_idx in range(collector.num_envs):
        if not (prev_success_mask[env_idx] and target_available_mask[env_idx]):
            pre_close_positions.append(np.array([0.5, 0.0, 0.5], dtype=np.float64))
            close_positions.append(np.array([0.5, 0.0, 0.5], dtype=np.float64))
            close_quaternions.append(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64))
            active_targets.append(False)
            continue

        root_pos_w = microwave.data.root_pos_w[env_idx].cpu().numpy()
        root_quat_wxyz = microwave.data.root_quat_w[env_idx].cpu().numpy()
        root_rot = R.from_quat([root_quat_wxyz[1], root_quat_wxyz[2], root_quat_wxyz[3], root_quat_wxyz[0]])

        facing_world = root_rot.apply(facing_axis_local)
        facing_norm = np.linalg.norm(facing_world)
        if facing_norm > 1e-6:
            facing_world /= facing_norm

        open_normal_world = slide_rot_world.apply(facing_world)
        open_normal_norm = np.linalg.norm(open_normal_world)
        if open_normal_norm > 1e-6:
            open_normal_world /= open_normal_norm

        ee_z = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        ee_y = -open_normal_world.copy()
        ee_y[2] = 0.0
        ee_y_norm = np.linalg.norm(ee_y)
        if ee_y_norm > 1e-6:
            ee_y /= ee_y_norm
        else:
            ee_y = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        ee_x = np.cross(ee_y, ee_z)
        ee_x_norm = np.linalg.norm(ee_x)
        if ee_x_norm > 1e-6:
            ee_x /= ee_x_norm
        else:
            ee_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        close_rot_mat = np.column_stack([ee_x, ee_y, ee_z])
        close_quat_xyzw = R.from_matrix(close_rot_mat).as_quat()
        close_quat_wxyz = np.array(
            [close_quat_xyzw[3], close_quat_xyzw[0], close_quat_xyzw[1], close_quat_xyzw[2]],
            dtype=np.float64,
        )

        base_pos = np.array(root_pos_w - env_origins[env_idx], dtype=np.float64)
        base_pos[2] += close_height_offset

        pre_close_pos = (
            base_pos
            + pre_close_facing_offset * facing_world
            + pre_close_open_normal_offset * open_normal_world
        )
        close_pos = (
            base_pos
            + close_facing_offset * facing_world
            - close_open_normal_backoff * open_normal_world
        )

        pre_close_positions.append(pre_close_pos)
        close_positions.append(close_pos)
        close_quaternions.append(close_quat_wxyz)
        active_targets.append(True)

    collector.control_gripper_batch(
        [1.0 if active else 0.0 for active in active_targets],
        steps=FIXED_GRIPPER_STEPS,
        active_mask=active_targets,
    )

    print(f"  🚪 [Primitive] Planning microwave pre-close pose for '{target_label}'")
    pre_close_plans, pre_close_plan_successes = collector.motion_planner.plan_to_pose_batch(
        collector.robot.data.joint_pos[: collector.num_envs].cpu().numpy(),
        collector.robot.data.joint_vel[: collector.num_envs].cpu().numpy(),
        collector.robot.joint_names,
        np.array(pre_close_positions, dtype=np.float64),
        np.array(close_quaternions, dtype=np.float64),
        target_prim_path=collector.target_prim_path,
    )
    pre_close_successes = [
        prev and available and plan_ok
        for prev, available, plan_ok in zip(prev_success_mask, target_available_mask, pre_close_plan_successes)
    ]
    collector.execute_trajectory_batch(
        pre_close_plans,
        [1.0 if success else 0.0 for success in pre_close_successes],
        desc="Primitive Microwave Pre-Close",
        active_mask=pre_close_successes,
    )

    if not any(pre_close_successes):
        _set_failure_context(context, collector.num_envs, prev_success_mask, target_available_mask)
        return False

    _, close_motion_successes = collector.execute_ik_motion(
        target_positions=np.array(close_positions, dtype=np.float64),
        target_quaternions=np.array(close_quaternions, dtype=np.float64),
        gripper_states=[1.0 if success else 0.0 for success in pre_close_successes],
        interpolation_steps=close_interp_steps,
        desc="Primitive Close Microwave Door",
        active_mask=pre_close_successes,
    )
    final_successes = [plan_ok and motion_ok for plan_ok, motion_ok in zip(pre_close_successes, close_motion_successes)]

    step_failure_reason_mask = [
        "inactive"
        if not prev_success_mask[env_idx]
        else "target_missing"
        if not target_available_mask[env_idx]
        else "planning_failed"
        if not final_successes[env_idx]
        else "none"
        for env_idx in range(collector.num_envs)
    ]

    success_eval_enabled = bool(params.get("success_eval", False))
    if success_eval_enabled:
        max_angle = float(params.get("close_success_max_angle_rad", CLOSE_SUCCESS_MAX_ANGLE_RAD))
        state_successes = []
        state_metrics = []
        for env_idx in range(collector.num_envs):
            env_success, env_metrics = _evaluate_close_success_with_metrics(
                scene=collector.scene,
                target_label=target_label,
                joint_name=CLOSE_SUCCESS_JOINT_NAME,
                max_angle_rad=max_angle,
                env_idx=env_idx,
            )
            state_successes.append(env_success)
            state_metrics.append(env_metrics)

        final_successes = [motion_ok and state_ok for motion_ok, state_ok in zip(final_successes, state_successes)]
        for env_idx in range(collector.num_envs):
            if not prev_success_mask[env_idx]:
                continue
            if final_successes[env_idx]:
                step_failure_reason_mask[env_idx] = "none"
            elif step_failure_reason_mask[env_idx] in {"planning_failed", "target_missing"}:
                continue
            else:
                step_failure_reason_mask[env_idx] = "success_eval_failed"

        context["task_success_mask"] = list(final_successes)
        context["task_success_metrics"] = list(state_metrics)
        for env_idx, metrics in enumerate(state_metrics):
            joint_val = metrics["joint_pos_rad"]
            angle_ok = np.isfinite(joint_val) and (abs(joint_val) <= max_angle)
            range_ok = bool(metrics["articulation_in_range"] > 0.5)
            joint_text = f"{joint_val:.4f}" if np.isfinite(joint_val) else "nan"
            print(
                f"    [TaskCheck][env {env_idx}] "
                f"joint_pos={joint_text} rad (|val|<={max_angle:.4f}: {'OK' if angle_ok else 'FAIL'}), "
                f"articulation_in_range={'OK' if range_ok else 'FAIL'}"
            )
        print(
            f"  ✅ [Primitive] Task success (close): "
            f"{sum(final_successes)}/{collector.num_envs} "
            f"(joint_pos<={max_angle:.3f} rad)"
        )

    context["post_close_success_mask"] = list(final_successes)
    context["step_failure_reason_mask"] = list(step_failure_reason_mask)
    context["env_alive_mask"] = list(final_successes)
    context["current_gripper_state"] = "open"
    context["close_completed"] = any(final_successes)
    context["primitive_executed_motion"] = True

    print(f"  ✅ [Primitive] Close success: {sum(final_successes)}/{collector.num_envs}")
    return any(final_successes)


def _set_failure_context(context: Dict[str, Any], num_envs: int, prev_mask: list, target_mask: list):
    context["post_close_success_mask"] = [False] * num_envs
    context["step_failure_reason_mask"] = [
        "inactive"
        if not prev_mask[env_idx]
        else "target_missing"
        if not target_mask[env_idx]
        else "planning_failed"
        for env_idx in range(num_envs)
    ]
    context["env_alive_mask"] = [False] * num_envs
