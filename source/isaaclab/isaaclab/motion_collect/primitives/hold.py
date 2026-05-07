from typing import Any, Dict, Optional

import numpy as np

from ..success.constants import HOLD_SUCCESS_MIN_Z
from ..success.evaluators import _evaluate_hold_success_with_metrics
from .utils import (
    commit_plan_failures,
    commit_task_failures,
    get_prev_alive_mask,
    init_local_active_mask,
    merge_segment_into_local_active,
)


def _expand_pose_array(value: Any, num_envs: int, expected_dim: int, name: str) -> np.ndarray:
    """Expand pose-like config to [num_envs, expected_dim]."""
    arr = np.array(value, dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] != expected_dim:
            raise ValueError(f"Primitive 'hold' param '{name}' expects {expected_dim} values, got {arr.shape[0]}.")
        return np.tile(arr, (num_envs, 1))
    if arr.ndim == 2 and arr.shape == (num_envs, expected_dim):
        return arr
    raise ValueError(
        f"Primitive 'hold' param '{name}' expects shape [{expected_dim}] or [{num_envs}, {expected_dim}], got {arr.shape}."
    )


def primitive_hold(collector, mode: Optional[str], params: Dict[str, Any], context: Dict[str, Any]) -> bool:
    """Move to hold pose and keep object held for several frames."""
    del mode

    n = collector.num_envs
    prev_alive = get_prev_alive_mask(context, n)
    local_active = init_local_active_mask(context, n)

    if not any(prev_alive):
        print("  [ABORT] Primitive 'hold': no active envs.")
        context["post_hold_success_mask"] = list(prev_alive)
        context["env_alive_mask"] = list(prev_alive)
        return False

    hold_steps = int(params.get("hold_steps", 10))
    hold_steps = max(0, hold_steps)

    current_ee_pos, current_ee_quat = collector.get_current_ee_pose()
    current_ee_pos = np.atleast_2d(np.array(current_ee_pos, dtype=float))
    current_ee_quat = np.atleast_2d(np.array(current_ee_quat, dtype=float))

    # Target position:
    # 1) explicit `position`, or
    # 2) use `lift_positions`/current_ee and add `lift_height` (default 0.10m).
    if "position" in params:
        target_positions = _expand_pose_array(params["position"], n, 3, "position")
    else:
        lift_height = float(params.get("lift_height", 0.10))
        if context.get("lift_positions") is not None:
            base_positions = np.array(context["lift_positions"], dtype=float)
        else:
            base_positions = current_ee_pos.copy()
        target_positions = base_positions.copy()
        target_positions[:, 2] += lift_height

    # Target orientation:
    # 1) explicit `orientation`, or
    # 2) keep pick orientation if available, else current EE orientation.
    if "orientation" in params:
        target_quaternions = _expand_pose_array(params["orientation"], n, 4, "orientation")
    elif context.get("all_pick_quat") is not None:
        target_quaternions = np.array(context["all_pick_quat"], dtype=float)
    else:
        target_quaternions = current_ee_quat.copy()

    print(
        f"   [Primitive] HOLD: moving to target pose, then holding for {hold_steps} frames "
        f"(active envs: {sum(prev_alive)}/{n})"
    )

    if collector.motion_planner is None:
        print("  [ERROR] Primitive 'hold' requires motion planner to be enabled.")
        # Treat missing planner as a planning-side failure for all alive envs.
        for i in range(n):
            local_active[i] = False
        commit_plan_failures(context, prev_alive, local_active)
        context["post_hold_success_mask"] = [False] * n
        context["env_alive_mask"] = [False] * n
        return False

    move_plans, move_plan_successes = collector.motion_planner.plan_to_pose_batch(
        collector.robot.data.joint_pos[:n].cpu().numpy(),
        collector.robot.data.joint_vel[:n].cpu().numpy(),
        collector.robot.joint_names,
        target_positions,
        target_quaternions,
        target_prim_path=collector.target_prim_path,
    )
    move_segment = [local_active[i] and bool(move_plan_successes[i]) for i in range(n)]
    merge_segment_into_local_active(local_active, move_segment)

    collector.execute_trajectory_batch(
        move_plans,
        [-1.0 if s else 0.0 for s in local_active],
        desc="Primitive Hold Move",
        active_mask=list(local_active),
    )

    # Keep holding (and record) by reusing gripper control loop.
    if hold_steps > 0 and any(local_active):
        collector.control_gripper_batch(
            [-1.0 if s else 0.0 for s in local_active],
            steps=hold_steps,
            active_mask=list(local_active),
        )

    plan_active_after = list(local_active)

    success_eval_enabled = bool(params.get("success_eval", False))
    if success_eval_enabled:
        held_object_label = context.get("held_object") or context.get("pick_target_label")
        if not held_object_label:
            print("[ERROR] Hold success_eval requires held object label.")
        else:
            state_successes = []
            state_metrics = []
            for env_idx in range(n):
                env_success, env_metrics = _evaluate_hold_success_with_metrics(
                    scene=collector.scene,
                    held_object_label=held_object_label,
                    env_idx=env_idx,
                )
                state_successes.append(env_success)
                state_metrics.append(env_metrics)

            commit_task_failures(context, prev_alive, plan_active_after, list(state_successes))
            context["task_success_metrics"] = list(state_metrics)
            for env_idx, metrics in enumerate(state_metrics):
                obj_z = metrics["held_object_z_env"]
                z_ok = np.isfinite(obj_z) and (obj_z > HOLD_SUCCESS_MIN_Z)
                z_text = f"{obj_z:.4f}" if np.isfinite(obj_z) else "nan"
                # print(
                #     f"    [TaskCheck][env {env_idx}] "
                #     f"held_z={z_text}m (>{HOLD_SUCCESS_MIN_Z:.2f}: {'OK' if z_ok else 'FAIL'})"
                # )
            print(
                f"   [Primitive] Task success (hold): "
                f"{sum(1 for i in range(n) if plan_active_after[i] and state_successes[i])}/{n}"
            )

    commit_plan_failures(context, prev_alive, local_active)

    plan_mask_final = list(context.get("plan_success_mask", [True] * n))
    task_mask_final = list(context.get("task_success_mask", [True] * n))
    final_alive = [
        bool(prev_alive[i] and plan_mask_final[i] and task_mask_final[i])
        for i in range(n)
    ]
    context["hold_pose"] = {
        "positions": np.array(target_positions, dtype=float),
        "orientations": np.array(target_quaternions, dtype=float),
        "hold_steps": hold_steps,
    }
    context["post_hold_success_mask"] = list(final_alive)
    context["env_alive_mask"] = list(final_alive)
    context["current_gripper_state"] = "closed"
    context["primitive_executed_motion"] = True

    print(f"   [Primitive] Hold success (alive): {sum(final_alive)}/{n}")
    return any(final_alive)
