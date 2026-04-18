from typing import Any, Dict, Optional

import numpy as np
import torch

from ..success.evaluators import (
    _evaluate_pour_process_with_metrics,
    _evaluate_pour_success_with_metrics,
)
from ..success.scene_helpers import _get_object_pose_world_from_scene
from ..tools.affordance_resolver import resolve_affordance_path
from ..tools.pose_utils import _quat_wxyz_to_matrix
from .utils import (
    _normalize_xy,
    compute_target_availability_mask,
    compute_tilt_quaternion_from_pour_direction,
    load_affordance_centroid,
    load_affordance_clusters,
    prepare_pour_geometry,
    log_target_missing_envs,
    search_best_pour_rotation,
    select_effective_pour_point_local,
    select_pour_cluster,
)


def primitive_pour(collector, mode: Optional[str], params: Dict[str, Any], context: Dict[str, Any]) -> bool:
    """Composite pour primitive: compute pour pose -> execute pour."""
    if "source" in params:
        print("[ERROR] 'pour.params.source' is deprecated and no longer supported. Source object is inferred from pick context.")
        return False
    if "container" in params:
        print("[ERROR] 'pour.params.container' is deprecated and no longer supported. Use 'pour.params.target' instead.")
        return False
    if "directional_pour_threshold" in params:
        print(
            "[ERROR] 'pour.params.directional_pour_threshold' is deprecated and no longer supported. "
            "A fixed internal threshold is used."
        )
        return False
    if not primitive_compute_pour_pose(collector, mode, params, context):
        return False
    return primitive_execute_pour(collector, params, context)


def primitive_compute_pour_pose(collector, mode: Optional[str], params: Dict[str, Any], context: Dict[str, Any]) -> bool:
    if context.get("post_pick_success_mask") is None or context.get("lift_positions") is None:
        print("[ERROR] Primitive 'compute_pour_pose' requires pick execution first.")
        return False

    target_label = context.get("held_object") or context.get("pick_target_label")
    container_label = params.get("target")
    if not target_label or not container_label:
        print("[ERROR] Primitive 'compute_pour_pose' requires target object and container.")
        return False

    tilt_angle = float(params.get("tilt_angle_deg", 90.0))
    move_hover_height = float(params.get("move_hover_height", 0.15))
    tilt_lift_height = float(params.get("tilt_lift_height", 0.08))
    tilt_horizontal_offset = float(params.get("tilt_horizontal_offset", 0.05))
    n_rotation_candidates = 36

    env_origins = collector._get_env_origins()
    prev_success_mask = list(
        context.get(
            "env_alive_mask",
            context.get("post_pick_success_mask", [True] * collector.num_envs),
        )
    )
    target_available_mask = compute_target_availability_mask(collector, container_label)
    log_target_missing_envs("pour", container_label, target_available_mask, prev_success_mask)
    lift_positions = np.array(context["lift_positions"], dtype=np.float64)
    ee_pos_rel_pour, _ = collector.get_current_ee_pose()

    held_obj = context.get("pick_target_label") or target_label
    object_pour_path = params.get("pour_npz_path") or resolve_affordance_path(held_obj, "pourout")
    container_pour_path = params.get("container_pour_npz") or resolve_affordance_path(container_label, "pourin")
    grasp_centers_local = list(context.get("selected_grasp_centers_local", []))
    has_any_grasp_center = any(center is not None for center in grasp_centers_local)
    if not has_any_grasp_center or not object_pour_path or not container_pour_path:
        print("[ERROR] Primitive 'compute_pour_pose' requires prior pick grasp-center context and pour affordance paths.")
        return False
    print(f"  🫗 [Primitive] Loading object pour affordance: {object_pour_path}")
    pour_geometry = prepare_pour_geometry(
        object_pour_path,
        directional_threshold=0.35,
    )
    pour_centroids_local = load_affordance_clusters(object_pour_path)
    print(f"  🫗 [Primitive] Loading container pour affordance: {container_pour_path}")
    container_center_local = load_affordance_centroid(container_pour_path)

    object_scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    container_scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    if collector.task_config:
        obj_cfg = collector.task_config.get_object_by_name(target_label)
        if obj_cfg is None:
            obj_cfg = getattr(collector.task_config, "articulations", {}).get(target_label)
        if obj_cfg is not None and getattr(obj_cfg, "scale", None) is not None:
            object_scale = np.array(obj_cfg.scale, dtype=np.float64)
        cont_cfg = collector.task_config.get_object_by_name(container_label)
        if cont_cfg is None:
            cont_cfg = getattr(collector.task_config, "articulations", {}).get(container_label)
        if cont_cfg is not None and getattr(cont_cfg, "scale", None) is not None:
            container_scale = np.array(cont_cfg.scale, dtype=np.float64)

    all_pour_pos = []
    all_pour_quat = []
    all_pour_dir = []
    all_pour_meta = []

    for env_idx in range(collector.num_envs):
        if not prev_success_mask[env_idx]:
            all_pour_pos.append(np.array(lift_positions[env_idx], dtype=np.float64))
            all_pour_quat.append(np.array(context["all_pick_quat"][env_idx], dtype=np.float64))
            all_pour_dir.append(np.array([1.0, 0.0, 0.0], dtype=np.float64))
            all_pour_meta.append({"status": "skip_after_pick_failure"})
            continue
        if not target_available_mask[env_idx]:
            all_pour_pos.append(np.array(lift_positions[env_idx], dtype=np.float64))
            all_pour_quat.append(np.array(context["all_pick_quat"][env_idx], dtype=np.float64))
            all_pour_dir.append(np.array([1.0, 0.0, 0.0], dtype=np.float64))
            all_pour_meta.append({"status": "target_missing"})
            continue

        grasp_center_local = grasp_centers_local[env_idx] if env_idx < len(grasp_centers_local) else None
        if grasp_center_local is None:
            print(f"  ⚠️ Env {env_idx}: missing grasp center in pick context, using lift fallback.")
            all_pour_pos.append(np.array(lift_positions[env_idx], dtype=np.float64))
            all_pour_quat.append(np.array(context["all_pick_quat"][env_idx], dtype=np.float64))
            all_pour_dir.append(np.array([1.0, 0.0, 0.0], dtype=np.float64))
            all_pour_meta.append({"status": "missing_grasp_center"})
            continue

        aff_grasp_center_local = np.asarray(grasp_center_local, dtype=np.float64)
        obj_pos_w, obj_quat_w = collector._get_object_pose_global(target_label, env_idx)
        cont_pos_w, cont_quat_w = collector._get_object_pose_global(container_label, env_idx)
        if obj_pos_w is None or cont_pos_w is None:
            print(f"  ⚠️ Env {env_idx}: pour pose unavailable, using lift fallback.")
            all_pour_pos.append(np.array(lift_positions[env_idx], dtype=np.float64))
            all_pour_quat.append(np.array(context["all_pick_quat"][env_idx], dtype=np.float64))
            all_pour_dir.append(np.array([1.0, 0.0, 0.0], dtype=np.float64))
            all_pour_meta.append({"status": "missing_pose"})
            continue

        if len(pour_centroids_local) == 1:
            selected_pour_local = np.asarray(pour_centroids_local[0], dtype=np.float64)
            selected_idx = 0
        else:
            selected_pour_local, selected_idx = select_pour_cluster(
                cluster_centroids_local=pour_centroids_local,
                basket_pos_world=obj_pos_w,
                basket_quat_world=obj_quat_w,
                container_pos_world=cont_pos_w,
            )

        rot_cup = _quat_wxyz_to_matrix(obj_quat_w)
        pick_quat_wxyz = np.array(context["all_pick_quat"][env_idx], dtype=np.float64)
        rot_pick = _quat_wxyz_to_matrix(pick_quat_wxyz)
        ee_z_local = rot_cup.T @ rot_pick[:, 2]

        effective_pour_local, pour_meta = select_effective_pour_point_local(
            pour_geometry,
            selected_pour_local,
            aff_grasp_center_local,
            ee_z_axis_local=ee_z_local,
        )
        origin = env_origins[env_idx]
        current_ee_world = np.asarray(ee_pos_rel_pour[env_idx], dtype=np.float64) + origin
        rot_container = _quat_wxyz_to_matrix(cont_quat_w)
        effective_pour_world = rot_cup @ (effective_pour_local * object_scale) + obj_pos_w
        object_center_world = rot_cup @ (pour_geometry["object_center_local"] * object_scale) + obj_pos_w
        container_center_world = rot_container @ (container_center_local * container_scale) + cont_pos_w

        (
            pour_target_world,
            ee_target_robot,
            ee_target_quat,
            pour_dir_world,
            candidate_score,
            score_breakdown,
        ) = search_best_pour_rotation(
            container_center_world=container_center_world,
            effective_pour_world=effective_pour_world,
            current_ee_world=current_ee_world,
            object_center_world=object_center_world,
            env_origin=origin,
            lift_pos_robot=np.asarray(lift_positions[env_idx], dtype=np.float64),
            current_grasp_quat_wxyz=np.array(context["all_pick_quat"][env_idx], dtype=np.float64),
            n_candidates=n_rotation_candidates,
            tilt_angle_deg=tilt_angle,
            tilt_lift_height=tilt_lift_height,
            tilt_horizontal_offset=tilt_horizontal_offset,
        )

        all_pour_pos.append(ee_target_robot)
        all_pour_quat.append(ee_target_quat)
        all_pour_dir.append(np.array(pour_dir_world, dtype=np.float64))
        all_pour_meta.append(
            {
                "status": "ok",
                "selected_centroid_index": int(selected_idx),
                "selected_centroid_local": np.array(selected_pour_local, dtype=np.float64).tolist(),
                "effective_pour_local": np.array(effective_pour_local, dtype=np.float64).tolist(),
                "pour_meta": pour_meta,
                "score": float(candidate_score),
                "score_breakdown": score_breakdown,
                "pour_target_world": np.array(pour_target_world, dtype=np.float64).tolist(),
            }
        )

    context["pour_pose"] = {
        "mode": "auto",
        "target_label": container_label,
        "target_available_mask": list(target_available_mask),
        "positions": np.array(all_pour_pos, dtype=np.float64),
        "orientations": np.array(all_pour_quat, dtype=np.float64),
        "pour_directions": np.array(all_pour_dir, dtype=np.float64),
        "move_hover_height": move_hover_height,
        "tilt_angle_deg": tilt_angle,
        "tilt_lift_height": tilt_lift_height,
        "tilt_horizontal_offset": tilt_horizontal_offset,
        "meta": all_pour_meta,
    }
    print(f"  🫗 [Primitive] Computed pour poses for '{target_label}' -> '{container_label}'")
    return True


def primitive_execute_pour(collector, params: Dict[str, Any], context: Dict[str, Any]) -> bool:
    pour_pose = context.get("pour_pose")
    if pour_pose is None:
        print("[ERROR] Primitive 'execute_pour' requires pour poses first.")
        return False

    prev_success_mask = list(context.get("post_pick_success_mask", [True] * collector.num_envs))
    target_available_mask = list(pour_pose.get("target_available_mask", [True] * collector.num_envs))
    move_hover_height = float(params.get("move_hover_height", pour_pose.get("move_hover_height", 0.15)))
    tilt_angle = float(params.get("tilt_angle_deg", pour_pose.get("tilt_angle_deg", 90.0)))
    tilt_lift_height = float(params.get("tilt_lift_height", pour_pose.get("tilt_lift_height", 0.08)))
    tilt_horizontal_offset = float(params.get("tilt_horizontal_offset", pour_pose.get("tilt_horizontal_offset", 0.05)))
    hold_steps = 10
    tilt_interpolation_steps = 30
    return_lift_height_raw = params.get("return_lift_height", None)
    has_return_lift = return_lift_height_raw is not None
    return_lift_height = float(return_lift_height_raw) if has_return_lift else 0.0

    print(f"  🚀 [Primitive] Executing POUR to '{pour_pose.get('target_label')}'")

    move_targets = np.array(pour_pose["positions"], dtype=np.float64).copy()
    move_targets[:, 2] += move_hover_height
    move_quats = np.array(pour_pose["orientations"], dtype=np.float64)

    move_plans, move_successes = collector.motion_planner.plan_to_pose_batch(
        collector.robot.data.joint_pos[: collector.num_envs].cpu().numpy(),
        collector.robot.data.joint_vel[: collector.num_envs].cpu().numpy(),
        collector.robot.joint_names,
        move_targets,
        move_quats,
        target_prim_path=collector.target_prim_path,
    )
    move_successes = [prev and available and cur for prev, available, cur in zip(prev_success_mask, target_available_mask, move_successes)]
    collector.execute_trajectory_batch(
        move_plans,
        [-1.0 if success else 0.0 for success in move_successes],
        desc="Primitive Move to Pour",
        active_mask=move_successes,
    )
    if not any(move_successes):
        print("  [ABORT] All envs failed at MOVE to pour position.")
        context["task_success_mask"] = [False] * collector.num_envs
        context["post_pour_success_mask"] = [False] * collector.num_envs
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

    all_tilt_quat = []
    all_tilt_pos = []
    for env_idx in range(collector.num_envs):
        if not move_successes[env_idx]:
            all_tilt_quat.append(move_quats[env_idx])
            tilt_pos = np.array(pour_pose["positions"][env_idx], dtype=np.float64).copy()
            tilt_pos[2] += tilt_lift_height
            all_tilt_pos.append(tilt_pos)
            continue

        tilt_quat = compute_tilt_quaternion_from_pour_direction(
            current_quat_wxyz=move_quats[env_idx],
            pour_direction_world=np.array(pour_pose["pour_directions"][env_idx], dtype=np.float64),
            tilt_angle_deg=tilt_angle,
        )
        pour_dir_xy, pour_dir_norm = _normalize_xy(np.array(pour_pose["pour_directions"][env_idx], dtype=np.float64))
        if pour_dir_norm <= 1e-6:
            pour_dir_xy = np.array([1.0, 0.0], dtype=np.float64)

        tilt_pos = np.array(pour_pose["positions"][env_idx], dtype=np.float64).copy()
        tilt_pos[2] += tilt_lift_height
        tilt_pos[:2] += pour_dir_xy * tilt_horizontal_offset

        all_tilt_quat.append(tilt_quat)
        all_tilt_pos.append(tilt_pos)

    # ---- Per-step tilt trajectory sampling (for pour process success check) ----
    held_pick_target = context.get("held_object") or context.get("pick_target_label")
    pour_meta_list = pour_pose.get("meta", [])
    per_step_trajectory: list[list[dict]] = [[] for _ in range(collector.num_envs)]

    def _sample_env_state(env_idx: int) -> None:
        if not move_successes[env_idx]:
            return
        meta = pour_meta_list[env_idx] if env_idx < len(pour_meta_list) else None
        if not isinstance(meta, dict) or meta.get("status") != "ok":
            return
        pos_world, quat_world = _get_object_pose_world_from_scene(
            collector.scene, held_pick_target, env_idx
        )
        if pos_world is None or quat_world is None:
            return
        rot = _quat_wxyz_to_matrix(np.array(quat_world, dtype=np.float64))
        z_axis_world_z = float(rot[2, 2])
        # Use effective_pour_local (the point actually aligned with the container
        # by search_best_pour_rotation). Falls back to selected_centroid_local for
        # backward compatibility if the compute stage didn't store it.
        tip_local_raw = meta.get("effective_pour_local")
        if tip_local_raw is None:
            tip_local_raw = meta.get("selected_centroid_local", [0.0, 0.0, 0.0])
        tip_local = np.array(tip_local_raw, dtype=np.float64)
        tip_world = np.array(pos_world, dtype=np.float64) + rot @ tip_local
        per_step_trajectory[env_idx].append(
            {
                "z_axis_world_z": z_axis_world_z,
                "tip_xy": np.array(tip_world[:2], dtype=np.float64),
            }
        )

    # Seed step 0 sample (state right before tilt motion begins)
    for env_idx in range(collector.num_envs):
        _sample_env_state(env_idx)

    def tilt_step_debug_fn(step: int, total_steps: int, env_successes_snapshot) -> None:
        for env_idx in range(collector.num_envs):
            _sample_env_state(env_idx)

    _, tilt_successes = collector.execute_ik_motion(
        target_positions=all_tilt_pos,
        target_quaternions=all_tilt_quat,
        gripper_states=[-1.0 if s else 0.0 for s in move_successes],
        interpolation_steps=tilt_interpolation_steps,
        desc="Primitive Tilt Pour",
        active_mask=move_successes,
        step_debug_fn=tilt_step_debug_fn,
    )
    tilt_successes = [a and b for a, b in zip(move_successes, tilt_successes)]

    print("  ⏱️ [Primitive] Holding pour pose...")
    action_dim = collector.env.env.action_manager.total_action_dim
    for _ in range(hold_steps):
        action = torch.zeros(collector.num_envs, action_dim, device=collector.device)
        action[:, :] = collector.robot.data.joint_pos[: collector.num_envs, :action_dim]
        for env_idx in range(collector.num_envs):
            action[env_idx, -1] = -1.0 if tilt_successes[env_idx] else 0.0
        collector.env.step(action)

    final_successes = list(tilt_successes)
    step_failure_reason_mask = ["none"] * collector.num_envs
    for env_idx in range(collector.num_envs):
        if not prev_success_mask[env_idx]:
            step_failure_reason_mask[env_idx] = "inactive"
        elif not target_available_mask[env_idx]:
            step_failure_reason_mask[env_idx] = "target_missing"
        elif not tilt_successes[env_idx]:
            step_failure_reason_mask[env_idx] = "planning_failed"
    success_eval_enabled = bool(params.get("success_eval", False))
    if success_eval_enabled:
        pick_target = context.get("held_object") or context.get("pick_target_label")
        pour_pose = context.get("pour_pose") or {}
        pour_target = params.get("target") or pour_pose.get("target_label")
        if not pick_target or not pour_target:
            print("[ERROR] Pour success_eval requires pick target and pour target.")
            context["task_success_mask"] = [False] * collector.num_envs
            context["post_pour_success_mask"] = [False] * collector.num_envs
            context["env_alive_mask"] = [False] * collector.num_envs
            context["step_failure_reason_mask"] = [
                "inactive" if not prev_success_mask[env_idx] else "success_eval_failed"
                for env_idx in range(collector.num_envs)
            ]
            return False

        axis_successes = []
        axis_metrics = []
        for env_idx in range(collector.num_envs):
            env_success, env_metrics = _evaluate_pour_success_with_metrics(
                scene=collector.scene,
                pick_target=pick_target,
                pour_target=pour_target,
                env_idx=env_idx,
            )
            axis_successes.append(env_success)
            axis_metrics.append(env_metrics)

        # ---- Per-step process checks (tip XY range + tilt monotonicity) ----
        process_successes: list[bool] = []
        process_metrics: list[dict] = []
        for env_idx in range(collector.num_envs):
            if not tilt_successes[env_idx]:
                process_successes.append(False)
                process_metrics.append({})
                continue
            meta = pour_meta_list[env_idx] if env_idx < len(pour_meta_list) else None
            pour_target_world = (
                meta.get("pour_target_world") if isinstance(meta, dict) else None
            )
            if pour_target_world is None:
                process_successes.append(False)
                process_metrics.append({})
                continue
            pour_center_xy = np.array(pour_target_world, dtype=np.float64)[:2]
            proc_ok, proc_metric = _evaluate_pour_process_with_metrics(
                per_step_trajectory=per_step_trajectory[env_idx],
                pour_center_world_xy=pour_center_xy,
                tilt_horizontal_offset=tilt_horizontal_offset,
            )
            process_successes.append(proc_ok)
            process_metrics.append(proc_metric)

        final_successes = [
            a and b and c
            for a, b, c in zip(tilt_successes, axis_successes, process_successes)
        ]
        for env_idx in range(collector.num_envs):
            if not prev_success_mask[env_idx]:
                continue
            if tilt_successes[env_idx] and not axis_successes[env_idx]:
                step_failure_reason_mask[env_idx] = "success_eval_failed"
            elif (
                tilt_successes[env_idx]
                and axis_successes[env_idx]
                and not process_successes[env_idx]
            ):
                step_failure_reason_mask[env_idx] = "pour_process_failed"
            elif final_successes[env_idx]:
                step_failure_reason_mask[env_idx] = "none"
        context["task_success_mask"] = list(final_successes)
        context["task_success_metrics"] = list(axis_metrics)
        context["task_process_metrics"] = list(process_metrics)
        for env_idx, metrics in enumerate(axis_metrics):
            z_comp = metrics["object_local_z_world_z"]
            tilt_ok = bool(tilt_successes[env_idx])
            z_ok = np.isfinite(z_comp) and (z_comp <= 0.0)
            pour_target_ok = bool(metrics["pour_target_in_range"] > 0.5)
            proc_ok = bool(process_successes[env_idx])
            proc_metric = process_metrics[env_idx] if env_idx < len(process_metrics) else {}
            tip_in_range = bool(proc_metric.get("pour_tip_in_range", 0.0) > 0.5)
            tilt_monotonic = bool(proc_metric.get("pour_tilt_monotonic", 0.0) > 0.5)
            max_xy_excess = proc_metric.get("max_xy_excess", float("nan"))
            max_tilt_reverse = proc_metric.get("max_tilt_reverse", float("nan"))
            total_tilt_decrease = proc_metric.get("total_tilt_decrease", float("nan"))
            final_ok = bool(final_successes[env_idx])
            z_text = f"{z_comp:.4f}" if np.isfinite(z_comp) else "nan"
            xy_excess_text = (
                f"{max_xy_excess:.4f}" if isinstance(max_xy_excess, float) and np.isfinite(max_xy_excess) else "nan"
            )
            tilt_rev_text = (
                f"{max_tilt_reverse:.4f}" if isinstance(max_tilt_reverse, float) and np.isfinite(max_tilt_reverse) else "nan"
            )
            total_dec_text = (
                f"{total_tilt_decrease:.4f}" if isinstance(total_tilt_decrease, float) and np.isfinite(total_tilt_decrease) else "nan"
            )
            fail_reasons = []
            if not tilt_ok:
                fail_reasons.append("tilt_motion_failed")
            if not z_ok:
                fail_reasons.append("z_axis_condition_failed")
            if not pour_target_ok:
                fail_reasons.append("pour_target_out_of_range")
            if tilt_ok and not tip_in_range:
                fail_reasons.append("pour_tip_out_of_range")
            if tilt_ok and not tilt_monotonic:
                fail_reasons.append("pour_tilt_not_monotonic")
            reason_text = ",".join(fail_reasons) if fail_reasons else "none"
            print(
                f"    [TaskCheck][env {env_idx}] "
                f"tilt_success={'OK' if tilt_ok else 'FAIL'}, "
                f"object_local_z_world_z={z_text} "
                f"(<= 0.0000: {'OK' if z_ok else 'FAIL'}), "
                f"pour_target_in_range={'OK' if pour_target_ok else 'FAIL'}, "
                f"process={'OK' if proc_ok else 'FAIL'} "
                f"(max_xy_excess={xy_excess_text}, max_tilt_reverse={tilt_rev_text}, "
                f"total_tilt_decrease={total_dec_text}), "
                f"final={'OK' if final_ok else 'FAIL'}, "
                f"reason={reason_text}"
            )
        print(
            "  ✅ [Primitive] Task success (pour): "
            f"{sum(final_successes)}/{collector.num_envs}"
        )

    if has_return_lift:
        print("  ↩️ [Primitive] Lifting before returning upright...")
        return_positions = []
        for pos in all_tilt_pos:
            lifted = np.array(pos, dtype=np.float64).copy()
            lifted[2] += return_lift_height
            return_positions.append(lifted)

        collector.execute_ik_motion(
            target_positions=return_positions,
            target_quaternions=all_tilt_quat,
            gripper_states=[-1.0 if s else 0.0 for s in tilt_successes],
            interpolation_steps=5,
            desc="Primitive Lift Before Return",
            active_mask=tilt_successes,
        )
        collector.execute_ik_motion(
            target_positions=return_positions,
            target_quaternions=move_quats,
            gripper_states=[-1.0 if s else 0.0 for s in tilt_successes],
            interpolation_steps=10,
            desc="Primitive Return Upright",
            active_mask=tilt_successes,
        )
    else:
        print("  ↩️ [Primitive] Returning upright...")
        collector.execute_ik_motion(
            target_positions=all_tilt_pos,
            target_quaternions=move_quats,
            gripper_states=[-1.0 if s else 0.0 for s in tilt_successes],
            interpolation_steps=10,
            desc="Primitive Return Upright",
            active_mask=tilt_successes,
        )

    context["post_pour_success_mask"] = list(final_successes)
    context["step_failure_reason_mask"] = list(step_failure_reason_mask)
    context["env_alive_mask"] = list(final_successes)
    context["current_gripper_state"] = "closed"
    context["pour_completed"] = any(final_successes)
    context["primitive_executed_motion"] = True

    print(f"  ✅ [Primitive] Pour success: {sum(final_successes)}/{collector.num_envs}")
    return any(final_successes)
