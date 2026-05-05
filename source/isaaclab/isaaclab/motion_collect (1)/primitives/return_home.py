from typing import Any, Dict, Optional

import numpy as np
import torch

def primitive_return_home(collector, mode: Optional[str], params: Dict[str, Any], context: Dict[str, Any]) -> bool:
    del mode
    initial_ee_pos = context.get("initial_ee_pos")
    initial_ee_quat = context.get("initial_ee_quat")
    if initial_ee_pos is None or initial_ee_quat is None:
        print("[ERROR] Primitive 'return_home' requires initial EE pose in context.")
        return False

    prev_success_mask = list(context.get("env_alive_mask", [True] * collector.num_envs))
    current_gripper_state = str(context.get("current_gripper_state", "open")).lower()
    gripper_cmd = -1.0 if current_gripper_state == "closed" else 1.0

    home_plans, home_successes = collector.motion_planner.plan_to_pose_batch(
        collector.robot.data.joint_pos[: collector.num_envs].cpu().numpy(),
        collector.robot.data.joint_vel[: collector.num_envs].cpu().numpy(),
        collector.robot.joint_names,
        np.array(initial_ee_pos, dtype=float),
        np.array(initial_ee_quat, dtype=float),
        target_prim_path=collector.target_prim_path,
    )
    home_successes = [prev and cur for prev, cur in zip(prev_success_mask, home_successes)]
    collector.execute_trajectory_batch(
        home_plans,
        [gripper_cmd if success else 0.0 for success in home_successes],
        desc="Primitive Return-Home",
        active_mask=home_successes,
    )

    post_settle_steps = int(params.get("post_settle_steps", 20))
    if post_settle_steps > 0:
        action_dim = collector.env.env.action_manager.total_action_dim
        for _ in range(post_settle_steps):
            action = torch.zeros(collector.num_envs, action_dim, device=collector.device)
            action[:, :] = collector.robot.data.joint_pos[: collector.num_envs, :action_dim]
            collector.env.step(action)

    context["current_gripper_state"] = current_gripper_state
    context["post_return_home_success_mask"] = list(home_successes)
    context["env_alive_mask"] = list(home_successes)
    context["primitive_executed_motion"] = True
    context["step_failure_reason_mask"] = [
        "none" if s else ("inactive" if not prev_success_mask[i] else "planning_failed")
        for i, s in enumerate(home_successes)
    ]
    print(f"  ✅ [Primitive] Return-home success: {sum(home_successes)}/{collector.num_envs}")

    return any(home_successes)
