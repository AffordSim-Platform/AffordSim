from typing import Any, Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..success.constants import OPEN_SUCCESS_JOINT_NAME, OPEN_SUCCESS_MIN_ANGLE_RAD
from ..success.evaluators import _evaluate_open_success_with_metrics
from ..tools.pose_utils import normalize_ee_orientation_wxyz, parse_facing_axis_local
from .utils import compute_target_availability_mask, log_target_missing_envs, primitive_compute_grasp_pose

FIXED_GRIPPER_STEPS = 10
FINAL_RETRACT_DISTANCE = 0.15
FINAL_NORMAL_OFFSET = 0.10
FINAL_RETRACT_INTERP_STEPS = 5


def primitive_open(collector, mode: Optional[str], params: Dict[str, Any], context: Dict[str, Any]) -> bool:
    """Composite open primitive: compute handle grasp -> execute articulation motion."""
    del mode
    if not primitive_compute_grasp_pose(collector, "affordance_open", params, context):
        return False
    return primitive_execute_open(collector, params, context)


def primitive_execute_open(collector, params: Dict[str, Any], context: Dict[str, Any]) -> bool:
    if context.get("grasp_pose") is None:
        print("[ERROR] Primitive 'execute_open' requires grasp poses first.")
        return False

    target_label = params.get("target") or context.get("pick_target_label")
    if not target_label:
        print("[ERROR] Primitive 'execute_open' requires a target articulation.")
        return False

    if "door_normal_local" in params:
        print(
            "[ERROR] Parameter `door_normal_local` is no longer supported. "
            "Configure the target's `facing_axis_local` in containers.yaml instead."
        )
        return False

    pre_pick_offset = float(params.get("pre_pick_offset", 0.10))
    pick_interp_steps = int(params.get("pick_interpolation_steps", 10))
    fast_pull_distance = float(params.get("fast_pull_distance", 0.30))
    fast_pull_interp_steps = int(params.get("fast_pull_interpolation_steps", 5))
    lift_before_pre_insert = float(params.get("lift_before_pre_insert", 0.30))
    lift_interp_steps = int(params.get("lift_interpolation_steps", 10))
    pre_insert_outward_offset = float(params.get("pre_insert_outward_offset", 0.03))
    align_above_interp_steps = int(params.get("align_above_interpolation_steps", 10))
    door_tangent_move_distance = float(params.get("door_tangent_move_distance", 0.20))
    tangent_move_interp_steps = int(params.get("tangent_move_interpolation_steps", 10))
    prev_success_mask = list(context.get("env_alive_mask", [True] * collector.num_envs))
    target_available_mask = list(context.get("grasp_pose", {}).get("target_available_mask", compute_target_availability_mask(collector, target_label)))
    log_target_missing_envs("open", target_label, target_available_mask, prev_success_mask)

    all_pick_pos = np.array(context["all_pick_pos"], dtype=np.float64)
    all_pick_quat = np.array(context["all_pick_quat"], dtype=np.float64)

    if not hasattr(collector, "_containers_registry"):
        collector._containers_registry = collector._load_containers_registry()
    container_cfg = (collector._containers_registry or {}).get(target_label)
    if container_cfg is None:
        print(
            f"[ERROR] Primitive 'open' requires target '{target_label}' to be registered in containers.yaml "
            "with `facing_axis_local`."
        )
        return False
    try:
        door_normal_local = parse_facing_axis_local(container_cfg.get("facing_axis_local"))
    except ValueError as exc:
        print(f"[ERROR] Invalid facing_axis_local for open target '{target_label}': {exc}")
        return False
    if door_normal_local is None:
        print(
            f"[ERROR] Primitive 'open' requires '{target_label}' to define `facing_axis_local` in containers.yaml."
        )
        return False
    door_normal_local = np.asarray(door_normal_local, dtype=np.float64)

    all_door_normals = []
    for env_idx in range(collector.num_envs):
        obj_pos, obj_quat = collector._get_object_pose_global(target_label, env_idx)
        if obj_pos is None:
            all_door_normals.append(np.array([0.0, -1.0, 0.0], dtype=np.float64))
            continue
        obj_rot = R.from_quat([obj_quat[1], obj_quat[2], obj_quat[3], obj_quat[0]])
        door_normal_world = obj_rot.apply(door_normal_local)
        norm = np.linalg.norm(door_normal_world)
        if norm > 1e-6:
            door_normal_world /= norm
        all_door_normals.append(door_normal_world)

    all_pre_pick_pos = []
    for env_idx in range(collector.num_envs):
        all_pre_pick_pos.append(all_pick_pos[env_idx] + pre_pick_offset * all_door_normals[env_idx])

    print(f"  🚪 [Primitive] Planning open pre-grasp for '{target_label}'")
    collector._update_all_env_obstacles()
    pre_pick_plans, pre_pick_plan_successes = collector.motion_planner.plan_to_pose_batch(
        collector.robot.data.joint_pos[: collector.num_envs].cpu().numpy(),
        collector.robot.data.joint_vel[: collector.num_envs].cpu().numpy(),
        collector.robot.joint_names,
        np.array(all_pre_pick_pos, dtype=np.float64),
        all_pick_quat,
    )
    pre_pick_successes = [prev and available and cur for prev, available, cur in zip(prev_success_mask, target_available_mask, pre_pick_plan_successes)]
    collector.execute_trajectory_batch(
        pre_pick_plans,
        [1.0 if s else 0.0 for s in pre_pick_successes],
        desc="Primitive Open-Microwave Pre-Grasp",
        active_mask=pre_pick_successes,
    )

    if not any(pre_pick_successes):
        step_failure_reason_mask = [
            "inactive"
            if not prev_success_mask[env_idx]
            else "target_missing"
            if not target_available_mask[env_idx]
            else "planning_failed"
            for env_idx in range(collector.num_envs)
        ]
        context["post_open_success_mask"] = [False] * collector.num_envs
        context["step_failure_reason_mask"] = step_failure_reason_mask
        context["env_alive_mask"] = [False] * collector.num_envs
        return False

    _, pick_env_successes = collector.execute_ik_motion(
        target_positions=all_pick_pos,
        target_quaternions=all_pick_quat,
        gripper_states=[1.0 if s else 0.0 for s in pre_pick_successes],
        interpolation_steps=pick_interp_steps,
        desc="Primitive Grasp Microwave Handle",
        active_mask=pre_pick_successes,
    )
    pick_successes = [a and b for a, b in zip(pre_pick_successes, pick_env_successes)]
    collector.control_gripper_batch(
        [-1.0 if s else 0.0 for s in pick_successes],
        steps=FIXED_GRIPPER_STEPS,
        active_mask=pick_successes,
    )
    if not any(pick_successes):
        print("  [ABORT] All envs failed at handle grasp.")
        context["post_open_success_mask"] = [False] * collector.num_envs
        context["step_failure_reason_mask"] = [
            "inactive"
            if not prev_success_mask[env_idx]
            else "target_missing"
            if not target_available_mask[env_idx]
            else "planning_failed"
            for env_idx in range(collector.num_envs)
        ]
        context["env_alive_mask"] = [False] * collector.num_envs
        return False
    active_successes = list(pick_successes)

    object_key = collector._find_object_in_scene(target_label)
    if object_key is None or object_key not in collector.scene.articulations:
        step_failure_reason_mask = [
            "inactive"
            if not prev_success_mask[env_idx]
            else "target_missing"
            if not target_available_mask[env_idx]
            else "planning_failed"
            for env_idx in range(collector.num_envs)
        ]
        context["post_open_success_mask"] = [False] * collector.num_envs
        context["step_failure_reason_mask"] = step_failure_reason_mask
        context["env_alive_mask"] = [False] * collector.num_envs
        return False

    microwave = collector.scene.articulations[object_key]
    slide_rot_world = R.from_rotvec(-0.5 * np.pi * np.array([0.0, 0.0, 1.0]))
    ee_pos_cur, ee_quat_cur = collector.get_current_ee_pose()
    ee_pos_cur = np.atleast_2d(ee_pos_cur)
    ee_quat_cur = np.atleast_2d(ee_quat_cur)

    door_normals_world = []
    grasp_quat_targets = []
    for env_idx in range(collector.num_envs):
        root_quat_wxyz = microwave.data.root_quat_w[env_idx].cpu().numpy()
        root_rot = R.from_quat([root_quat_wxyz[1], root_quat_wxyz[2], root_quat_wxyz[3], root_quat_wxyz[0]]).as_matrix()

        door_normal_world = root_rot @ door_normal_local
        normal_norm = np.linalg.norm(door_normal_world)
        if normal_norm > 1e-6:
            door_normal_world /= normal_norm

        door_normals_world.append(door_normal_world)
        grasp_quat_targets.append(ee_quat_cur[env_idx])

    stage2_targets = []
    stage2_quats = []
    for env_idx in range(collector.num_envs):
        if not pick_successes[env_idx]:
            stage2_targets.append(all_pick_pos[env_idx])
            stage2_quats.append(all_pick_quat[env_idx])
            continue

        pull_dir = np.array(door_normals_world[env_idx], dtype=np.float64)
        pull_norm = np.linalg.norm(pull_dir)
        if pull_norm > 1e-6:
            pull_dir /= pull_norm
        stage2_targets.append(ee_pos_cur[env_idx] + fast_pull_distance * pull_dir)
        stage2_quats.append(normalize_ee_orientation_wxyz(grasp_quat_targets[env_idx], ee_quat_cur[env_idx]))

    _, stage2_successes = collector.execute_ik_motion(
        target_positions=stage2_targets,
        target_quaternions=stage2_quats,
        gripper_states=[-1.0 if s else 0.0 for s in active_successes],
        interpolation_steps=fast_pull_interp_steps,
        desc="Primitive Fast Pull Microwave",
        active_mask=active_successes,
    )
    active_successes = [ok and cur for ok, cur in zip(active_successes, stage2_successes)]

    print("  🚪 [Primitive] Re-grasping inside articulation gap...")
    collector.control_gripper_batch(
        [1.0 if s else 0.0 for s in active_successes],
        steps=FIXED_GRIPPER_STEPS,
        active_mask=active_successes,
    )

    ee_pos_cur, ee_quat_cur = collector.get_current_ee_pose()
    ee_pos_cur = np.atleast_2d(ee_pos_cur)
    ee_quat_cur = np.atleast_2d(ee_quat_cur)
    lift_targets = []
    lift_quats = []
    for env_idx in range(collector.num_envs):
        if not active_successes[env_idx]:
            lift_targets.append(ee_pos_cur[env_idx])
            lift_quats.append(ee_quat_cur[env_idx])
            continue
        lift_target = ee_pos_cur[env_idx].copy()
        lift_target[2] += lift_before_pre_insert
        lift_targets.append(lift_target)
        lift_quats.append(ee_quat_cur[env_idx])

    _, lift_successes = collector.execute_ik_motion(
        target_positions=lift_targets,
        target_quaternions=lift_quats,
        gripper_states=[1.0 if s else 0.0 for s in active_successes],
        interpolation_steps=lift_interp_steps,
        desc="Primitive Lift Before Gap Insert",
        active_mask=active_successes,
    )
    active_successes = [ok and cur for ok, cur in zip(active_successes, lift_successes)]
    ee_pos_after_lift, ee_quat_after_lift = collector.get_current_ee_pose()
    ee_pos_after_lift = np.atleast_2d(ee_pos_after_lift)
    ee_quat_after_lift = np.atleast_2d(ee_quat_after_lift)

    align_targets = []
    align_above_targets = []
    tangent_targets = []
    align_quats = []
    for env_idx in range(collector.num_envs):
        if not active_successes[env_idx]:
            align_targets.append(ee_pos_after_lift[env_idx])
            align_above_targets.append(ee_pos_after_lift[env_idx])
            tangent_targets.append(ee_pos_after_lift[env_idx])
            align_quats.append(ee_quat_after_lift[env_idx])
            continue

        # Transform local directions to world frame using microwave root rotation
        root_quat_wxyz = microwave.data.root_quat_w[env_idx].cpu().numpy()
        root_rot = R.from_quat([root_quat_wxyz[1], root_quat_wxyz[2], root_quat_wxyz[3], root_quat_wxyz[0]])
        outward_world = root_rot.apply(door_normal_local)
        slide_world = slide_rot_world.apply(outward_world)

        # Compute align EE orientation: Z down, Y anti-parallel to slide (grip axis along slide)
        ee_z = np.array([0.0, 0.0, -1.0])
        ee_y = -slide_world.copy()
        ee_y[2] = 0.0
        ee_y_norm = np.linalg.norm(ee_y)
        if ee_y_norm > 1e-6:
            ee_y /= ee_y_norm
        else:
            ee_y = np.array([0.0, -1.0, 0.0])
        ee_x = np.cross(ee_y, ee_z)
        ee_x /= np.linalg.norm(ee_x)
        align_rot_mat = np.column_stack([ee_x, ee_y, ee_z])
        align_xyzw = R.from_matrix(align_rot_mat).as_quat()
        insert_quat_wxyz = np.array([align_xyzw[3], align_xyzw[0], align_xyzw[1], align_xyzw[2]])

        align_pos = np.array(all_pick_pos[env_idx], dtype=np.float64) + pre_insert_outward_offset * outward_world
        align_above_pos = align_pos.copy()
        align_above_pos[2] += lift_before_pre_insert
        tangent_pos = align_pos + door_tangent_move_distance * slide_world

        align_targets.append(align_pos)
        align_above_targets.append(align_above_pos)
        tangent_targets.append(tangent_pos)
        align_quats.append(insert_quat_wxyz)

    _, align_above_successes = collector.execute_ik_motion(
        target_positions=np.array(align_above_targets),
        target_quaternions=np.array(align_quats),
        gripper_states=[1.0 if s else 0.0 for s in active_successes],
        interpolation_steps=align_above_interp_steps,
        desc="Primitive IK To Align-Above",
        active_mask=active_successes,
    )
    active_successes = [ok and cur for ok, cur in zip(active_successes, align_above_successes)]

    _, align_successes = collector.execute_ik_motion(
        target_positions=np.array(align_targets),
        target_quaternions=np.array(align_quats),
        gripper_states=[1.0 if s else 0.0 for s in active_successes],
        interpolation_steps=lift_interp_steps,
        desc="Primitive IK To Align",
        active_mask=active_successes,
    )
    active_successes = [ok and cur for ok, cur in zip(active_successes, align_successes)]

    _, tangent_move_successes = collector.execute_ik_motion(
        target_positions=np.array(tangent_targets),
        target_quaternions=np.array(align_quats),
        gripper_states=[1.0 if s else 0.0 for s in active_successes],
        interpolation_steps=tangent_move_interp_steps,
        desc="Primitive Slide Along Door Tangent",
        active_mask=active_successes,
    )
    active_successes = [ok and cur for ok, cur in zip(active_successes, tangent_move_successes)]

    retract_targets = []
    for env_idx in range(collector.num_envs):
        if not active_successes[env_idx]:
            retract_targets.append(tangent_targets[env_idx])
            continue

        root_quat_wxyz = microwave.data.root_quat_w[env_idx].cpu().numpy()
        root_rot = R.from_quat([root_quat_wxyz[1], root_quat_wxyz[2], root_quat_wxyz[3], root_quat_wxyz[0]])
        outward_world = root_rot.apply(door_normal_local)
        retract_dir = -slide_rot_world.apply(outward_world)
        retract_norm = np.linalg.norm(retract_dir)
        if retract_norm > 1e-6:
            retract_dir /= retract_norm
        outward_norm = np.linalg.norm(outward_world)
        if outward_norm > 1e-6:
            outward_world /= outward_norm
        retract_targets.append(
            np.array(tangent_targets[env_idx], dtype=np.float64)
            + FINAL_RETRACT_DISTANCE * retract_dir
            + FINAL_NORMAL_OFFSET * outward_world
        )

    _, retract_successes = collector.execute_ik_motion(
        target_positions=np.array(retract_targets, dtype=np.float64),
        target_quaternions=np.array(align_quats),
        gripper_states=[1.0 if s else 0.0 for s in active_successes],
        interpolation_steps=FINAL_RETRACT_INTERP_STEPS,
        desc="Primitive Retract Opposite Door Opening",
        active_mask=active_successes,
    )
    final_successes = [ok and cur for ok, cur in zip(active_successes, retract_successes)]

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
        min_angle = float(OPEN_SUCCESS_MIN_ANGLE_RAD)
        state_successes = []
        state_metrics = []
        for env_idx in range(collector.num_envs):
            env_success, env_metrics = _evaluate_open_success_with_metrics(
                scene=collector.scene,
                target_label=target_label,
                joint_name=OPEN_SUCCESS_JOINT_NAME,
                min_angle_rad=min_angle,
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
            angle_ok = np.isfinite(joint_val) and (abs(joint_val) >= min_angle)
            range_ok = bool(metrics["articulation_in_range"] > 0.5)
            joint_text = f"{joint_val:.4f}" if np.isfinite(joint_val) else "nan"
            print(
                f"    [TaskCheck][env {env_idx}] "
                f"joint_pos={joint_text} rad (|val|>={min_angle:.4f}: {'OK' if angle_ok else 'FAIL'}), "
                f"articulation_in_range={'OK' if range_ok else 'FAIL'}"
            )
        print(
            f"  ✅ [Primitive] Task success (open): "
            f"{sum(final_successes)}/{collector.num_envs} "
            f"(joint_pos>={min_angle:.3f} rad)"
        )

    context["post_open_success_mask"] = list(final_successes)
    context["step_failure_reason_mask"] = list(step_failure_reason_mask)
    context["env_alive_mask"] = list(final_successes)
    context["current_gripper_state"] = "open"
    context["open_completed"] = any(final_successes)
    context["primitive_executed_motion"] = True

    print(f"  ✅ [Primitive] Open success: {sum(final_successes)}/{collector.num_envs}")
    return any(final_successes)
