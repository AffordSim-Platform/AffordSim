from typing import Any, Dict, Optional

import numpy as np

from ..tools.pose_utils import _quat_wxyz_to_matrix
from .utils import parse_position_offset, primitive_compute_grasp_pose

FIXED_GRIPPER_STEPS = 10


def primitive_pick(collector, mode: Optional[str], params: Dict[str, Any], context: Dict[str, Any]) -> bool:
    """Composite pick primitive: grasp pose -> container check -> execute pick."""
    if not primitive_compute_grasp_pose(collector, mode, params, context):
        return False
    if not primitive_detect_pick_container_context(collector, None, params, context):
        return False
    return primitive_execute_pick(collector, params, context)


def primitive_detect_pick_container_context(
    collector, mode: Optional[str], params: Dict[str, Any], context: Dict[str, Any]
) -> bool:
    if context.get("all_pick_pos") is None or context.get("pick_target_label") is None:
        print("[ERROR] Primitive 'detect_pick_container_context' requires grasp poses first.")
        return False

    target_label = params.get("target") or context["pick_target_label"]
    pre_pick_offset = parse_position_offset(
        params.get("pre_pick_offset"),
        context.get("pick_pre_pick_offset", 0.1),
    )

    all_pre_pick_pos = []
    for pos in context["all_pick_pos"]:
        pre_pos = np.array(pos, dtype=float).copy()
        pre_pos += pre_pick_offset
        all_pre_pick_pos.append(pre_pos)

    container_picks = collector._detect_and_adjust_for_container_pick(target_label, all_pre_pick_pos)
    container_modes = ["semi_enclosed" if container_pick is not None else "normal" for container_pick in container_picks]

    context["pre_pick_positions"] = np.array(all_pre_pick_pos)
    context["pick_container_context"] = {
        "mode": mode or "semi_enclosed_check",
        "target_label": target_label,
        "container_picks": container_picks,
        "per_env_modes": container_modes,
    }

    print(f"  [Primitive] Pick container modes: {container_modes}")
    return True


def primitive_execute_pick(collector, params: Dict[str, Any], context: Dict[str, Any]) -> bool:
    if context.get("all_pick_pos") is None or context.get("all_pick_quat") is None:
        print("[ERROR] Primitive 'execute_pick' requires grasp poses first.")
        return False

    execution_mode = "auto"
    robot_params = dict(context.get("pick_robot_params", collector._get_pick_robot_params()))
    pick_interpolation_steps = int(params.get("pick_interpolation_steps", context.get("pick_interpolation_steps", 5)))
    lift_interpolation_steps = int(params.get("lift_interpolation_steps", context.get("lift_interpolation_steps", 5)))
    lift_offset = float(params.get("lift_offset", robot_params.get("lift_offset", 0.1)))
    prev_success_mask = list(context.get("env_alive_mask", [True] * collector.num_envs))
    target_available_mask = list(context.get("grasp_pose", {}).get("target_available_mask", [True] * collector.num_envs))

    pre_pick_positions = context.get("pre_pick_positions")
    if pre_pick_positions is None:
        pre_pick_positions = np.array(context["all_pick_pos"], dtype=float).copy()
        pre_pick_positions += parse_position_offset(None, context.get("pick_pre_pick_offset", 0.1))

    container_context = context.get("pick_container_context", {})
    container_picks = list(container_context.get("container_picks", [None] * collector.num_envs))
    if execution_mode == "normal":
        container_picks = [None] * collector.num_envs

    print(f"  🚀 [Primitive] Planning Pre-PICK ({execution_mode})...")

    pre_pick_plans, pre_pick_plan_successes = collector.motion_planner.plan_to_pose_batch(
        collector.robot.data.joint_pos[: collector.num_envs].cpu().numpy(),
        collector.robot.data.joint_vel[: collector.num_envs].cpu().numpy(),
        collector.robot.joint_names,
        np.array(pre_pick_positions),
        np.array(context["all_pick_quat"]),
        target_prim_path=collector.target_prim_path,
    )
    pre_pick_successes = [prev and available and cur for prev, available, cur in zip(prev_success_mask, target_available_mask, pre_pick_plan_successes)]

    collector.execute_trajectory_batch(
        pre_pick_plans,
        [1.0 if success else 0.0 for success in pre_pick_successes],
        desc="Primitive Pre-Picking",
        active_mask=pre_pick_successes,
    )

    print("  🚀 [Primitive] Descending to PICK (IK)...")
    pick_gripper_open = [1.0 if success else 0.0 for success in pre_pick_successes]
    pick_target_positions = np.array(context["all_pick_pos"], dtype=float).copy()
    _, pick_env_successes = collector.execute_ik_motion(
        pick_target_positions,
        np.array(context["all_pick_quat"]),
        gripper_states=pick_gripper_open,
        interpolation_steps=pick_interpolation_steps,
        desc="Primitive Pick Descent",
        active_mask=pre_pick_successes,
    )
    pick_successes = [pre_success and ik_success for pre_success, ik_success in zip(pre_pick_successes, pick_env_successes)]

    collector.control_gripper_batch(
        [-1.0 if success else 0.0 for success in pick_successes],
        steps=FIXED_GRIPPER_STEPS,
        active_mask=pick_successes,
    )

    print("  🚀 [Primitive] Retreat/Lift after PICK (IK)...")
    lift_positions = []
    for pos in context["all_pick_pos"]:
        lift_pos = np.array(pos, dtype=float).copy()
        lift_pos[2] += lift_offset
        lift_positions.append(lift_pos)

    # For picks from inside a container, retreat to the reached pre-pick pose
    # instead of lifting upward.
    if execution_mode != "normal":
        retreat_count = 0
        for env_idx in range(collector.num_envs):
            if container_picks[env_idx] is not None:
                lift_positions[env_idx] = np.array(pre_pick_positions[env_idx], dtype=float)
                retreat_count += 1
        if retreat_count > 0:
            print(f"  [Primitive] Container pick retreat enabled: {retreat_count}/{collector.num_envs} envs -> pre-pick pose")

    _, lift_env_successes = collector.execute_ik_motion(
        np.array(lift_positions),
        np.array(context["all_pick_quat"]),
        gripper_states=[-1.0 if success else 0.0 for success in pick_successes],
        interpolation_steps=lift_interpolation_steps,
        desc="Primitive Lift",
        active_mask=pick_successes,
    )
    lift_successes = [pick_success and lift_success for pick_success, lift_success in zip(pick_successes, lift_env_successes)]

    held_object_offset_in_ee = [None] * collector.num_envs
    current_ee_pos, current_ee_quat = collector.get_current_ee_pose()
    target_label = context.get("pick_target_label")
    for env_idx in range(collector.num_envs):
        if not lift_successes[env_idx] or not target_label:
            continue
        object_pos_env, _ = collector.get_object_pose(target_label, env_idx=env_idx)
        if object_pos_env is None:
            continue
        ee_pos_env = np.array(current_ee_pos[env_idx], dtype=float)
        ee_quat_wxyz = np.array(current_ee_quat[env_idx], dtype=float)
        rot_ee = _quat_wxyz_to_matrix(ee_quat_wxyz)
        held_object_offset_in_ee[env_idx] = rot_ee.T @ (np.array(object_pos_env, dtype=float) - ee_pos_env)

    context["lift_positions"] = np.array(lift_positions)
    context["held_object_offset_in_ee"] = list(held_object_offset_in_ee)
    context["pick_successes"] = list(pick_successes)
    context["post_pick_success_mask"] = list(lift_successes)
    context["step_failure_reason_mask"] = [
        "inactive"
        if not prev_success_mask[env_idx]
        else "target_missing"
        if not target_available_mask[env_idx]
        else "planning_failed"
        if not lift_successes[env_idx]
        else "none"
        for env_idx in range(collector.num_envs)
    ]
    context["env_alive_mask"] = list(lift_successes)
    context["current_gripper_state"] = "closed"
    context["pick_completed"] = any(lift_successes)
    context["primitive_executed_motion"] = True

    print(f"  ✅ [Primitive] Pick success: {sum(lift_successes)}/{collector.num_envs}")
    return any(lift_successes)
