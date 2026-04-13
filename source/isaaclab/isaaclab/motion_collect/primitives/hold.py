from typing import Any, Dict, Optional

import numpy as np

from ..success.constants import HOLD_SUCCESS_MIN_Z
from ..success.evaluators import _evaluate_hold_success_with_metrics


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

    prev_success_mask = list(
        context.get(
            "env_alive_mask",
            context.get("post_pick_success_mask", [True] * collector.num_envs),
        )
    )

    if not any(prev_success_mask):
        print("  [ABORT] Primitive 'hold': no active envs.")
        context["post_hold_success_mask"] = list(prev_success_mask)
        context["step_failure_reason_mask"] = ["inactive"] * collector.num_envs
        context["env_alive_mask"] = list(prev_success_mask)
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
        target_positions = _expand_pose_array(params["position"], collector.num_envs, 3, "position")
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
        target_quaternions = _expand_pose_array(params["orientation"], collector.num_envs, 4, "orientation")
    elif context.get("all_pick_quat") is not None:
        target_quaternions = np.array(context["all_pick_quat"], dtype=float)
    else:
        target_quaternions = current_ee_quat.copy()

    print(
        f"  🤲 [Primitive] HOLD: moving to target pose, then holding for {hold_steps} frames "
        f"(active envs: {sum(prev_success_mask)}/{collector.num_envs})"
    )

    if collector.motion_planner is None:
        print("  [ERROR] Primitive 'hold' requires motion planner to be enabled.")
        context["post_hold_success_mask"] = [False] * collector.num_envs
        context["step_failure_reason_mask"] = [
            "inactive" if not prev_success_mask[i] else "planning_failed" for i in range(collector.num_envs)
        ]
        context["env_alive_mask"] = [False] * collector.num_envs
        return False

    collector._update_all_env_obstacles()
    move_plans, move_plan_successes = collector.motion_planner.plan_to_pose_batch(
        collector.robot.data.joint_pos[: collector.num_envs].cpu().numpy(),
        collector.robot.data.joint_vel[: collector.num_envs].cpu().numpy(),
        collector.robot.joint_names,
        target_positions,
        target_quaternions,
    )
    move_successes = [prev and cur for prev, cur in zip(prev_success_mask, move_plan_successes)]

    collector.execute_trajectory_batch(
        move_plans,
        [-1.0 if success else 0.0 for success in move_successes],
        desc="Primitive Hold Move",
        active_mask=move_successes,
    )

    # Keep holding (and record) by reusing gripper control loop.
    if hold_steps > 0 and any(move_successes):
        collector.control_gripper_batch(
            [-1.0 if success else 0.0 for success in move_successes],
            steps=hold_steps,
            active_mask=move_successes,
        )

    step_failure_reason_mask = []
    for i, success in enumerate(move_successes):
        if not prev_success_mask[i]:
            step_failure_reason_mask.append("inactive")
        elif success:
            step_failure_reason_mask.append("none")
        else:
            step_failure_reason_mask.append("planning_failed")

    success_eval_enabled = bool(params.get("success_eval", False))
    if success_eval_enabled:
        held_object_label = context.get("held_object") or context.get("pick_target_label")
        if not held_object_label:
            print("[ERROR] Hold success_eval requires held object label.")
        else:
            state_successes = []
            state_metrics = []
            for env_idx in range(collector.num_envs):
                env_success, env_metrics = _evaluate_hold_success_with_metrics(
                    scene=collector.scene,
                    held_object_label=held_object_label,
                    env_idx=env_idx,
                )
                state_successes.append(env_success)
                state_metrics.append(env_metrics)

            move_successes = [motion_ok and state_ok for motion_ok, state_ok in zip(move_successes, state_successes)]
            for i, success in enumerate(move_successes):
                if not prev_success_mask[i]:
                    continue
                if success:
                    step_failure_reason_mask[i] = "none"
                elif step_failure_reason_mask[i] == "planning_failed":
                    continue
                else:
                    step_failure_reason_mask[i] = "success_eval_failed"

            context["task_success_mask"] = list(move_successes)
            context["task_success_metrics"] = list(state_metrics)
            for env_idx, metrics in enumerate(state_metrics):
                obj_z = metrics["held_object_z_env"]
                z_ok = np.isfinite(obj_z) and (obj_z > HOLD_SUCCESS_MIN_Z)
                z_text = f"{obj_z:.4f}" if np.isfinite(obj_z) else "nan"
                print(
                    f"    [TaskCheck][env {env_idx}] "
                    f"held_z={z_text}m (>{HOLD_SUCCESS_MIN_Z:.2f}: {'OK' if z_ok else 'FAIL'})"
                )
            print(
                f"  ✅ [Primitive] Task success (hold): "
                f"{sum(move_successes)}/{collector.num_envs}"
            )

    context["hold_pose"] = {
        "positions": np.array(target_positions, dtype=float),
        "orientations": np.array(target_quaternions, dtype=float),
        "hold_steps": hold_steps,
    }
    context["post_hold_success_mask"] = list(move_successes)
    context["step_failure_reason_mask"] = step_failure_reason_mask
    context["env_alive_mask"] = list(move_successes)
    context["current_gripper_state"] = "closed"
    context["primitive_executed_motion"] = True

    print(f"  ✅ [Primitive] Hold success: {sum(move_successes)}/{collector.num_envs}")
    return any(move_successes)
