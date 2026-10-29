from typing import Any, Dict, Optional

import numpy as np
from ..hold_action import make_joint_hold_action
from ..success.evaluators import _evaluate_open_success_with_metrics, _evaluate_pull_success_with_metrics

from .utils import (
    commit_plan_failures,
    commit_task_failures,
    get_prev_alive_mask,
    init_local_active_mask,
    merge_segment_into_local_active,
)


def primitive_return_home(collector, mode: Optional[str], params: Dict[str, Any], context: Dict[str, Any]) -> bool:
    del mode
    initial_ee_pos = context.get("initial_ee_pos")
    initial_ee_quat = context.get("initial_ee_quat")
    if initial_ee_pos is None or initial_ee_quat is None:
        print("[ERROR] Primitive 'return_home' requires initial EE pose in context.")
        return False

    n = collector.num_envs
    prev_alive = get_prev_alive_mask(context, n)
    local_active = init_local_active_mask(context, n)
    current_gripper_state = str(context.get("current_gripper_state", "open")).lower()
    gripper_cmd = -1.0 if current_gripper_state == "closed" else 1.0

    home_plans, home_segment_successes = collector.motion_planner.plan_to_pose_batch(
        collector.robot.data.joint_pos[:n].cpu().numpy(),
        collector.robot.data.joint_vel[:n].cpu().numpy(),
        collector.robot.joint_names,
        np.array(initial_ee_pos, dtype=float),
        np.array(initial_ee_quat, dtype=float),
        target_prim_path=collector.target_prim_path,
    )
    merge_segment_into_local_active(local_active, home_segment_successes)
    collector.execute_trajectory_batch(
        home_plans,
        [gripper_cmd if s else 0.0 for s in local_active],
        desc="Primitive Return-Home",
        active_mask=list(local_active),
    )

    post_settle_steps = int(params.get("post_settle_steps", 20))
    if post_settle_steps > 0:
        for _ in range(post_settle_steps):
            action = make_joint_hold_action(collector.env, collector.robot, n, device=collector.device)
            result = collector.env.step(action)
            recorder = getattr(collector, "recorder", None)
            if recorder is not None and getattr(recorder, "preserve_idle_steps", False):
                obs, _, term, trunc, _ = result
                recorder.add_step(obs, action, term, trunc)

    commit_plan_failures(context, prev_alive, local_active)

    if params.get("monitor_robot_cabinet_contact", False):
        monitor = getattr(collector, "cabinet_retreat_monitor", None)
        if monitor is None:
            contact_results = [{"status": "unknown", "contact_detected": None,
                                "reason": "robot-cabinet contact monitor unavailable"} for _ in range(n)]
        else:
            contact_results = monitor.stop()
        if not isinstance(contact_results, list) or len(contact_results) != n:
            raise RuntimeError("Robot-cabinet contact monitor returned an invalid environment mapping")
        clear = [isinstance(result, dict) and result.get("status") == "complete"
                 and result.get("contact_detected") is False for result in contact_results]
        commit_task_failures(context, prev_alive, local_active, clear)
        context["cabinet_retreat_contact"] = contact_results
        print(f"   [Primitive] Collision-free retreat and return: {sum(clear)}/{n}")

    check = context.pop("pull_after_return_check", None)
    if check:
        outcomes, final_metrics = [], []
        for i in range(n):
            if check["joint_type"] == "prismatic":
                ok, metrics = _evaluate_pull_success_with_metrics(
                    collector.scene, check["target"], check["joint_name"],
                    min_displacement=check["threshold"], env_idx=i,
                )
            else:
                ok, metrics = _evaluate_open_success_with_metrics(
                    collector.scene, check["target"], check["joint_name"],
                    min_angle_rad=check["threshold"], env_idx=i,
                )
            outcomes.append(ok)
            final_metrics.append(metrics)
        commit_task_failures(context, prev_alive, local_active, outcomes)
        previous_metrics = context.get("task_success_metrics", [{} for _ in range(n)])
        context["task_success_metrics"] = [
            {**(previous_metrics[i] if i < len(previous_metrics) and isinstance(previous_metrics[i], dict) else {}),
             "post_return_home": final_metrics[i]}
            for i in range(n)
        ]
        print(f"   [Primitive] Drawer remains open after return: {sum(outcomes)}/{n}")

    plan_mask_final = list(context.get("plan_success_mask", [True] * n))
    task_mask_final = list(context.get("task_success_mask", [True] * n))
    final_alive = [
        bool(prev_alive[i] and plan_mask_final[i] and task_mask_final[i])
        for i in range(n)
    ]
    context["current_gripper_state"] = current_gripper_state
    context["post_return_home_success_mask"] = list(final_alive)
    context["env_alive_mask"] = list(final_alive)
    context["primitive_executed_motion"] = True
    print(f"   [Primitive] Return-home success (alive): {sum(final_alive)}/{n}")

    return any(final_alive)
