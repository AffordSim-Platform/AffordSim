from typing import Any, Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..success.constants import CLOSE_SUCCESS_MAX_ANGLE_RAD
from ..success.evaluators import _evaluate_close_success_with_metrics
from ..tools.pose_utils import parse_facing_axis_local
from .utils import (
    FAILURE_REASON_TARGET_MISSING,
    commit_plan_failures,
    commit_task_failures,
    compute_target_availability_mask,
    get_prev_alive_mask,
    init_local_active_mask,
    log_target_missing_envs,
    mark_local_inactive_with_reason,
    merge_segment_into_local_active,
)

FIXED_GRIPPER_STEPS = 3


def primitive_close(collector, mode: Optional[str], params: Dict[str, Any], context: Dict[str, Any]) -> bool:
    """Microwave close primitive driven by a pre-close CuRobo waypoint and a single IK push."""
    del mode
    return primitive_execute_close(collector, params, context)


def primitive_execute_close(collector, params: Dict[str, Any], context: Dict[str, Any]) -> bool:
    target_label = params.get("target")
    if not target_label:
        print("[ERROR] Primitive 'close' requires a target articulation.")
        return False

    n = collector.num_envs
    prev_alive = get_prev_alive_mask(context, n)
    local_active = init_local_active_mask(context, n)
    target_available_mask = compute_target_availability_mask(collector, target_label)
    log_target_missing_envs("close", target_label, target_available_mask, prev_alive)
    for env_idx in range(n):
        if prev_alive[env_idx] and not target_available_mask[env_idx]:
            mark_local_inactive_with_reason(
                context, local_active, env_idx, FAILURE_REASON_TARGET_MISSING
            )
    success_eval_enabled = bool(params.get("success_eval", False))
    success_joint_name = params.get("success_joint_name")
    if success_eval_enabled and not success_joint_name:
        print("[ERROR] Primitive 'close' with success_eval=true requires `params.success_joint_name` in YAML.")
        for i in range(n):
            local_active[i] = False
        commit_plan_failures(context, prev_alive, local_active)
        commit_task_failures(context, prev_alive, list(local_active), [False] * n)
        context["post_close_success_mask"] = [False] * n
        context["env_alive_mask"] = [False] * n
        return False

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
        for i in range(n):
            local_active[i] = False
        commit_plan_failures(context, prev_alive, local_active)
        context["post_close_success_mask"] = [False] * n
        context["env_alive_mask"] = [False] * n
        return False

    microwave = collector.scene.articulations[object_key]
    slide_rot_world = R.from_rotvec(-0.5 * np.pi * np.array([0.0, 0.0, 1.0], dtype=np.float64))
    env_origins = collector._get_env_origins()

    pre_close_positions = []
    close_positions = []
    close_quaternions = []
    active_targets = []

    for env_idx in range(n):
        if not local_active[env_idx]:
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
        [1.0 if s else 0.0 for s in local_active],
        steps=FIXED_GRIPPER_STEPS,
        active_mask=list(local_active),
    )

    print(f"  🚪 [Primitive] Planning microwave pre-close pose for '{target_label}'")
    pre_close_plans, pre_close_plan_successes = collector.motion_planner.plan_to_pose_batch(
        collector.robot.data.joint_pos[:n].cpu().numpy(),
        collector.robot.data.joint_vel[:n].cpu().numpy(),
        collector.robot.joint_names,
        np.array(pre_close_positions, dtype=np.float64),
        np.array(close_quaternions, dtype=np.float64),
        target_prim_path=collector.target_prim_path,
    )
    pre_segment = [local_active[i] and bool(pre_close_plan_successes[i]) for i in range(n)]
    merge_segment_into_local_active(local_active, pre_segment)
    collector.execute_trajectory_batch(
        pre_close_plans,
        [1.0 if s else 0.0 for s in local_active],
        desc="Primitive Microwave Pre-Close",
        active_mask=list(local_active),
    )

    _, close_motion_successes = collector.execute_ik_motion(
        target_positions=np.array(close_positions, dtype=np.float64),
        target_quaternions=np.array(close_quaternions, dtype=np.float64),
        gripper_states=[1.0 if s else 0.0 for s in local_active],
        interpolation_steps=close_interp_steps,
        desc="Primitive Close Microwave Door",
        active_mask=list(local_active),
    )
    merge_segment_into_local_active(local_active, close_motion_successes)
    plan_active_after = list(local_active)

    if success_eval_enabled:
        max_angle = float(params.get("close_success_max_angle_rad", CLOSE_SUCCESS_MAX_ANGLE_RAD))
        state_successes = []
        state_metrics = []
        for env_idx in range(n):
            env_success, env_metrics = _evaluate_close_success_with_metrics(
                scene=collector.scene,
                target_label=target_label,
                joint_name=success_joint_name,
                max_angle_rad=max_angle,
                env_idx=env_idx,
            )
            state_successes.append(env_success)
            state_metrics.append(env_metrics)

        commit_task_failures(context, prev_alive, plan_active_after, list(state_successes))
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
            f"{sum(1 for i in range(n) if plan_active_after[i] and state_successes[i])}/{n} "
            f"(joint_pos<={max_angle:.3f} rad)"
        )

    commit_plan_failures(context, prev_alive, local_active)

    plan_mask_final = list(context.get("plan_success_mask", [True] * n))
    task_mask_final = list(context.get("task_success_mask", [True] * n))
    final_alive = [
        bool(prev_alive[i] and plan_mask_final[i] and task_mask_final[i])
        for i in range(n)
    ]
    context["post_close_success_mask"] = list(final_alive)
    context["env_alive_mask"] = list(final_alive)
    context["current_gripper_state"] = "open"
    context["close_completed"] = any(final_alive)
    context["primitive_executed_motion"] = True

    print(f"  ✅ [Primitive] Close success (alive): {sum(final_alive)}/{n}")
    return any(final_alive)
