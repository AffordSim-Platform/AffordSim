from typing import Any, Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..success.constants import PULL_SUCCESS_JOINT_NAME, PULL_SUCCESS_MIN_DISPLACEMENT
from ..success.evaluators import _evaluate_open_success_with_metrics
from ..tools.pose_utils import normalize_ee_orientation_wxyz, parse_facing_axis_local
from .utils import compute_target_availability_mask, log_target_missing_envs, primitive_compute_grasp_pose

FIXED_GRIPPER_STEPS = 10


def primitive_pull(collector, mode: Optional[str], params: Dict[str, Any], context: Dict[str, Any]) -> bool:
    """Pull primitive for opening articulated drawers/cabinets.

    Motion sequence:
      1. Compute grasp pose on the handle via affordance
      2. CuRobo plan to pre-grasp (offset along pull direction)
      3. IK descend to grasp position
      4. Close gripper
      5. IK pull linearly along the pull direction
      6. Open gripper (optional)
    """
    del mode
    if not primitive_compute_grasp_pose(collector, "affordance_pull", params, context):
        return False
    return primitive_execute_pull(collector, params, context)


def primitive_execute_pull(collector, params: Dict[str, Any], context: Dict[str, Any]) -> bool:
    if context.get("grasp_pose") is None:
        print("[ERROR] Primitive 'execute_pull' requires grasp poses first.")
        return False

    target_label = params.get("target") or context.get("pick_target_label")
    if not target_label:
        print("[ERROR] Primitive 'execute_pull' requires a target articulation.")
        return False

    # --- Parameters ---
    if "pull_direction_local" in params:
        print(
            "[ERROR] Parameter `pull_direction_local` is no longer supported. "
            "Configure the target's `facing_axis_local` in containers.yaml instead."
        )
        return False

    pre_grasp_offset = float(params.get("pre_grasp_offset", 0.08))
    pick_interp_steps = int(params.get("pick_interpolation_steps", 10))
    pull_distance = float(params.get("pull_distance", 0.20))
    pull_interp_steps = int(params.get("pull_interpolation_steps", 30))
    release_after_pull = bool(params.get("release_after_pull", True))

    prev_success_mask = list(context.get("env_alive_mask", [True] * collector.num_envs))
    target_available_mask = list(
        context.get("grasp_pose", {}).get(
            "target_available_mask",
            compute_target_availability_mask(collector, target_label),
        )
    )
    log_target_missing_envs("pull", target_label, target_available_mask, prev_success_mask)

    all_pick_pos = np.array(context["all_pick_pos"], dtype=np.float64)
    all_pick_quat = np.array(context["all_pick_quat"], dtype=np.float64)

    if not hasattr(collector, "_containers_registry"):
        collector._containers_registry = collector._load_containers_registry()
    container_cfg = (collector._containers_registry or {}).get(target_label)
    if container_cfg is None:
        print(
            f"[ERROR] Primitive 'pull' requires target '{target_label}' to be registered in containers.yaml "
            "with `facing_axis_local`."
        )
        return False
    try:
        pull_direction_local = parse_facing_axis_local(container_cfg.get("facing_axis_local"))
    except ValueError as exc:
        print(f"[ERROR] Invalid facing_axis_local for pull target '{target_label}': {exc}")
        return False
    if pull_direction_local is None:
        print(
            f"[ERROR] Primitive 'pull' requires '{target_label}' to define `facing_axis_local` in containers.yaml."
        )
        return False
    pull_direction_local = np.asarray(pull_direction_local, dtype=np.float64)

    # Find the articulation to read its root pose for transforming pull direction
    object_key = collector._find_object_in_scene(target_label)
    if object_key is None or object_key not in collector.scene.articulations:
        print(f"[ERROR] Articulation '{target_label}' not found in scene.")
        context["post_pull_success_mask"] = [False] * collector.num_envs
        context["env_alive_mask"] = [False] * collector.num_envs
        return False

    articulation = collector.scene.articulations[object_key]

    # Compute world-frame pull direction per env
    all_pull_dirs_world = []
    for env_idx in range(collector.num_envs):
        root_quat_wxyz = articulation.data.root_quat_w[env_idx].cpu().numpy()
        root_rot = R.from_quat([root_quat_wxyz[1], root_quat_wxyz[2], root_quat_wxyz[3], root_quat_wxyz[0]])
        pull_dir_world = root_rot.apply(pull_direction_local)
        norm = np.linalg.norm(pull_dir_world)
        if norm > 1e-6:
            pull_dir_world /= norm
        all_pull_dirs_world.append(pull_dir_world)

    # Compute pre-grasp positions (offset along pull direction from handle)
    all_pre_grasp_pos = []
    for env_idx in range(collector.num_envs):
        pre_pos = all_pick_pos[env_idx] + pre_grasp_offset * all_pull_dirs_world[env_idx]
        all_pre_grasp_pos.append(pre_pos)

    # --- Phase 1: CuRobo plan to pre-grasp ---
    print(f"  🪝 [Primitive] Planning pull pre-grasp for '{target_label}'")
    collector._update_all_env_obstacles()
    pre_pick_plans, pre_pick_plan_successes = collector.motion_planner.plan_to_pose_batch(
        collector.robot.data.joint_pos[: collector.num_envs].cpu().numpy(),
        collector.robot.data.joint_vel[: collector.num_envs].cpu().numpy(),
        collector.robot.joint_names,
        np.array(all_pre_grasp_pos, dtype=np.float64),
        all_pick_quat,
    )
    pre_pick_successes = [
        prev and available and cur
        for prev, available, cur in zip(prev_success_mask, target_available_mask, pre_pick_plan_successes)
    ]
    collector.execute_trajectory_batch(
        pre_pick_plans,
        [1.0 if s else 0.0 for s in pre_pick_successes],
        desc="Primitive Pull Pre-Grasp",
        active_mask=pre_pick_successes,
    )

    if not any(pre_pick_successes):
        _set_failure_context(context, collector.num_envs, prev_success_mask, target_available_mask)
        return False

    # --- Phase 2: IK descend to grasp position ---
    print("  🪝 [Primitive] Descending to grasp handle (IK)...")
    _, pick_env_successes = collector.execute_ik_motion(
        target_positions=all_pick_pos,
        target_quaternions=all_pick_quat,
        gripper_states=[1.0 if s else 0.0 for s in pre_pick_successes],
        interpolation_steps=pick_interp_steps,
        desc="Primitive Pull Grasp Descent",
        active_mask=pre_pick_successes,
    )
    pick_successes = [a and b for a, b in zip(pre_pick_successes, pick_env_successes)]

    # --- Phase 3: Close gripper ---
    collector.control_gripper_batch(
        [-1.0 if s else 0.0 for s in pick_successes],
        steps=FIXED_GRIPPER_STEPS,
        active_mask=pick_successes,
    )

    if not any(pick_successes):
        print("  [ABORT] All envs failed at handle grasp.")
        _set_failure_context(context, collector.num_envs, prev_success_mask, target_available_mask)
        return False

    # --- Phase 4: IK pull along the pull direction ---
    print(f"  🪝 [Primitive] Pulling drawer '{target_label}' (distance={pull_distance:.3f}m)...")

    # Read current EE pose as pull start point
    ee_pos_cur, ee_quat_cur = collector.get_current_ee_pose()
    ee_pos_cur = np.atleast_2d(ee_pos_cur)
    ee_quat_cur = np.atleast_2d(ee_quat_cur)

    pull_targets = []
    pull_quats = []
    for env_idx in range(collector.num_envs):
        if not pick_successes[env_idx]:
            pull_targets.append(ee_pos_cur[env_idx])
            pull_quats.append(ee_quat_cur[env_idx])
            continue
        # Pull from current EE position along the pull direction
        pull_target = ee_pos_cur[env_idx] + pull_distance * all_pull_dirs_world[env_idx]
        pull_targets.append(pull_target)
        pull_quats.append(normalize_ee_orientation_wxyz(ee_quat_cur[env_idx], ee_quat_cur[env_idx]))

    _, pull_env_successes = collector.execute_ik_motion(
        target_positions=np.array(pull_targets, dtype=np.float64),
        target_quaternions=np.array(pull_quats, dtype=np.float64),
        gripper_states=[-1.0 if s else 0.0 for s in pick_successes],
        interpolation_steps=pull_interp_steps,
        desc="Primitive Pull Drawer",
        active_mask=pick_successes,
    )
    final_successes = [a and b for a, b in zip(pick_successes, pull_env_successes)]

    # --- Phase 5: Release gripper (optional) ---
    if release_after_pull:
        collector.control_gripper_batch(
            [1.0 if s else 0.0 for s in final_successes],
            steps=FIXED_GRIPPER_STEPS,
            active_mask=final_successes,
        )

    # --- Success evaluation ---
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
        min_displacement = float(PULL_SUCCESS_MIN_DISPLACEMENT)
        state_successes = []
        state_metrics = []
        for env_idx in range(collector.num_envs):
            env_success, env_metrics = _evaluate_open_success_with_metrics(
                scene=collector.scene,
                target_label=target_label,
                joint_name=PULL_SUCCESS_JOINT_NAME,
                min_angle_rad=min_displacement,
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
            disp_ok = np.isfinite(joint_val) and (abs(joint_val) >= min_displacement)
            range_ok = bool(metrics["articulation_in_range"] > 0.5)
            joint_text = f"{joint_val:.4f}" if np.isfinite(joint_val) else "nan"
            print(
                f"    [TaskCheck][env {env_idx}] "
                f"joint_pos={joint_text} ({'OK' if disp_ok else 'FAIL'}), "
                f"articulation_in_range={'OK' if range_ok else 'FAIL'}"
            )
        print(
            f"  [Primitive] Task success (pull): "
            f"{sum(final_successes)}/{collector.num_envs} "
            f"(joint_pos>={min_displacement:.3f})"
        )

    # --- Store context ---
    context["post_pull_success_mask"] = list(final_successes)
    context["step_failure_reason_mask"] = list(step_failure_reason_mask)
    context["env_alive_mask"] = list(final_successes)
    context["current_gripper_state"] = "open" if release_after_pull else "closed"
    context["pull_completed"] = any(final_successes)
    context["primitive_executed_motion"] = True

    print(f"  [Primitive] Pull success: {sum(final_successes)}/{collector.num_envs}")
    return any(final_successes)


def _set_failure_context(context: Dict[str, Any], num_envs: int, prev_mask: list, target_mask: list):
    """Helper to set context on total failure."""
    context["post_pull_success_mask"] = [False] * num_envs
    context["step_failure_reason_mask"] = [
        "inactive"
        if not prev_mask[env_idx]
        else "target_missing"
        if not target_mask[env_idx]
        else "planning_failed"
        for env_idx in range(num_envs)
    ]
    context["env_alive_mask"] = [False] * num_envs
