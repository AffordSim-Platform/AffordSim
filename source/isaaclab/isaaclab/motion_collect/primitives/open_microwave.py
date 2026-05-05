from typing import Any, Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..success.constants import OPEN_SUCCESS_MIN_ANGLE_RAD
from ..success.evaluators import _evaluate_open_success_with_metrics
from ..tools.pose_utils import normalize_ee_orientation_wxyz, parse_facing_axis_local
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
    primitive_compute_grasp_pose,
)

FIXED_GRIPPER_STEPS = 3
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
    n = collector.num_envs
    prev_alive = get_prev_alive_mask(context, n)
    local_active = init_local_active_mask(context, n)
    target_available_mask = list(context.get("grasp_pose", {}).get("target_available_mask", compute_target_availability_mask(collector, target_label)))
    log_target_missing_envs("open", target_label, target_available_mask, prev_alive)
    for env_idx in range(n):
        if prev_alive[env_idx] and not target_available_mask[env_idx]:
            mark_local_inactive_with_reason(
                context, local_active, env_idx, FAILURE_REASON_TARGET_MISSING
            )
    success_eval_enabled = bool(params.get("success_eval", False))
    success_joint_name = params.get("success_joint_name")
    if success_eval_enabled and not success_joint_name:
        print("[ERROR] Primitive 'open' with success_eval=true requires `params.success_joint_name` in YAML.")
        for i in range(n):
            local_active[i] = False
        commit_plan_failures(context, prev_alive, local_active)
        commit_task_failures(context, prev_alive, list(local_active), [False] * n)
        context["post_open_success_mask"] = [False] * n
        context["env_alive_mask"] = [False] * n
        return False

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
    for env_idx in range(n):
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
    for env_idx in range(n):
        all_pre_pick_pos.append(all_pick_pos[env_idx] + pre_pick_offset * all_door_normals[env_idx])

    print(f"  🚪 [Primitive] Planning open pre-grasp for '{target_label}'")
    pre_pick_plans, pre_pick_plan_successes = collector.motion_planner.plan_to_pose_batch(
        collector.robot.data.joint_pos[:n].cpu().numpy(),
        collector.robot.data.joint_vel[:n].cpu().numpy(),
        collector.robot.joint_names,
        np.array(all_pre_pick_pos, dtype=np.float64),
        all_pick_quat,
        target_prim_path=collector.target_prim_path,
    )
    pre_segment = [local_active[i] and bool(pre_pick_plan_successes[i]) for i in range(n)]
    merge_segment_into_local_active(local_active, pre_segment)
    collector.execute_trajectory_batch(
        pre_pick_plans,
        [1.0 if s else 0.0 for s in local_active],
        desc="Primitive Open-Microwave Pre-Grasp",
        active_mask=list(local_active),
    )

    _, pick_env_successes = collector.execute_ik_motion(
        target_positions=all_pick_pos,
        target_quaternions=all_pick_quat,
        gripper_states=[1.0 if s else 0.0 for s in local_active],
        interpolation_steps=pick_interp_steps,
        desc="Primitive Grasp Microwave Handle",
        active_mask=list(local_active),
    )
    merge_segment_into_local_active(local_active, pick_env_successes)
    collector.control_gripper_batch(
        [-1.0 if s else 0.0 for s in local_active],
        steps=FIXED_GRIPPER_STEPS,
        active_mask=list(local_active),
    )

    object_key = collector._find_object_in_scene(target_label)
    if object_key is None or object_key not in collector.scene.articulations:
        print(f"[ERROR] Articulation '{target_label}' not found in scene.")
        for i in range(n):
            local_active[i] = False
        commit_plan_failures(context, prev_alive, local_active)
        context["post_open_success_mask"] = [False] * n
        context["env_alive_mask"] = [False] * n
        return False

    microwave = collector.scene.articulations[object_key]
    slide_rot_world = R.from_rotvec(-0.5 * np.pi * np.array([0.0, 0.0, 1.0]))
    ee_pos_cur, ee_quat_cur = collector.get_current_ee_pose()
    ee_pos_cur = np.atleast_2d(ee_pos_cur)
    ee_quat_cur = np.atleast_2d(ee_quat_cur)

    door_normals_world = []
    grasp_quat_targets = []
    for env_idx in range(n):
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
    for env_idx in range(n):
        if not local_active[env_idx]:
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
        gripper_states=[-1.0 if s else 0.0 for s in local_active],
        interpolation_steps=fast_pull_interp_steps,
        desc="Primitive Fast Pull Microwave",
        active_mask=list(local_active),
    )
    merge_segment_into_local_active(local_active, stage2_successes)

    print("  🚪 [Primitive] Re-grasping inside articulation gap...")
    collector.control_gripper_batch(
        [1.0 if s else 0.0 for s in local_active],
        steps=FIXED_GRIPPER_STEPS,
        active_mask=list(local_active),
    )

    ee_pos_cur, ee_quat_cur = collector.get_current_ee_pose()
    ee_pos_cur = np.atleast_2d(ee_pos_cur)
    ee_quat_cur = np.atleast_2d(ee_quat_cur)
    lift_targets = []
    lift_quats = []
    for env_idx in range(n):
        if not local_active[env_idx]:
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
        gripper_states=[1.0 if s else 0.0 for s in local_active],
        interpolation_steps=lift_interp_steps,
        desc="Primitive Lift Before Gap Insert",
        active_mask=list(local_active),
    )
    merge_segment_into_local_active(local_active, lift_successes)
    ee_pos_after_lift, ee_quat_after_lift = collector.get_current_ee_pose()
    ee_pos_after_lift = np.atleast_2d(ee_pos_after_lift)
    ee_quat_after_lift = np.atleast_2d(ee_quat_after_lift)

    align_targets = []
    align_above_targets = []
    tangent_targets = []
    align_quats = []
    for env_idx in range(n):
        if not local_active[env_idx]:
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
        gripper_states=[1.0 if s else 0.0 for s in local_active],
        interpolation_steps=align_above_interp_steps,
        desc="Primitive IK To Align-Above",
        active_mask=list(local_active),
    )
    merge_segment_into_local_active(local_active, align_above_successes)

    _, align_successes = collector.execute_ik_motion(
        target_positions=np.array(align_targets),
        target_quaternions=np.array(align_quats),
        gripper_states=[1.0 if s else 0.0 for s in local_active],
        interpolation_steps=lift_interp_steps,
        desc="Primitive IK To Align",
        active_mask=list(local_active),
    )
    merge_segment_into_local_active(local_active, align_successes)

    _, tangent_move_successes = collector.execute_ik_motion(
        target_positions=np.array(tangent_targets),
        target_quaternions=np.array(align_quats),
        gripper_states=[1.0 if s else 0.0 for s in local_active],
        interpolation_steps=tangent_move_interp_steps,
        desc="Primitive Slide Along Door Tangent",
        active_mask=list(local_active),
    )
    merge_segment_into_local_active(local_active, tangent_move_successes)

    retract_targets = []
    for env_idx in range(n):
        if not local_active[env_idx]:
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
        gripper_states=[1.0 if s else 0.0 for s in local_active],
        interpolation_steps=FINAL_RETRACT_INTERP_STEPS,
        desc="Primitive Retract Opposite Door Opening",
        active_mask=list(local_active),
    )
    merge_segment_into_local_active(local_active, retract_successes)
    plan_active_after = list(local_active)

    if success_eval_enabled:
        min_angle = float(OPEN_SUCCESS_MIN_ANGLE_RAD)
        state_successes = []
        state_metrics = []
        for env_idx in range(n):
            env_success, env_metrics = _evaluate_open_success_with_metrics(
                scene=collector.scene,
                target_label=target_label,
                joint_name=success_joint_name,
                min_angle_rad=min_angle,
                env_idx=env_idx,
            )
            state_successes.append(env_success)
            state_metrics.append(env_metrics)

        commit_task_failures(context, prev_alive, plan_active_after, list(state_successes))
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
            f"{sum(1 for i in range(n) if plan_active_after[i] and state_successes[i])}/{n} "
            f"(joint_pos>={min_angle:.3f} rad)"
        )

    commit_plan_failures(context, prev_alive, local_active)

    plan_mask_final = list(context.get("plan_success_mask", [True] * n))
    task_mask_final = list(context.get("task_success_mask", [True] * n))
    final_alive = [
        bool(prev_alive[i] and plan_mask_final[i] and task_mask_final[i])
        for i in range(n)
    ]
    context["post_open_success_mask"] = list(final_alive)
    context["env_alive_mask"] = list(final_alive)
    context["current_gripper_state"] = "open"
    context["open_completed"] = any(final_alive)
    context["primitive_executed_motion"] = True
    print(f"  ✅ [Primitive] Open success (alive): {sum(final_alive)}/{n}")
    return any(final_alive)
