from typing import Any, Dict, Optional

import numpy as np
import torch

from .utils import (
    commit_plan_failures,
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
        action_dim = collector.env.env.action_manager.total_action_dim
        for _ in range(post_settle_steps):
            action = torch.zeros(n, action_dim, device=collector.device)
            action[:, :] = collector.robot.data.joint_pos[:n, :action_dim]
            collector.env.step(action)

    commit_plan_failures(context, prev_alive, local_active)

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
