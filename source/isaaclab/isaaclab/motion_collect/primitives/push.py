from typing import Any, Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from ..success.constants import PUSH_SUCCESS_TILT_THRESHOLD, PUSH_SUCCESS_XY_THRESHOLD
from ..success.evaluators import _evaluate_push_success_with_metrics
from ..tools.affordance_resolver import resolve_affordance_path
from .utils import compute_target_availability_mask, log_target_missing_envs, transform_grasp_to_world

FIXED_GRIPPER_STEPS = 10


def primitive_push(collector, mode: Optional[str], params: Dict[str, Any], context: Dict[str, Any]) -> bool:
    """Composite push primitive: compute push poses -> execute push."""
    del mode
    if not primitive_compute_push_pose(collector, params, context):
        return False
    return primitive_execute_push(collector, params, context)


def load_push_affordance_candidates(npz_path: str) -> list[dict]:
    """Load all push affordance candidates from grasp_result.npz."""
    data = np.load(npz_path)

    num_clusters = int(data["num_clusters"]) if "num_clusters" in data else 0
    candidates = []

    for i in range(num_clusters):
        prefix = f"cluster_{i}_"
        key = prefix + "grasp_center"
        if key not in data:
            continue
        candidates.append(
            {
                "candidate_index": i,
                "grasp_center": np.asarray(data[prefix + "grasp_center"], dtype=np.float64),
                "approach_dir": np.asarray(data[prefix + "approach_dir"], dtype=np.float64),
                "grip_axis": np.asarray(data[prefix + "grip_axis"], dtype=np.float64),
                "binormal": np.asarray(data[prefix + "binormal"], dtype=np.float64),
                "rotation_matrix": np.asarray(data[prefix + "rotation_matrix"], dtype=np.float64),
                "score": float(data[prefix + "score"]) if prefix + "score" in data else 1.0,
            }
        )

    if not candidates and "grasp_center" in data:
        candidates.append(
            {
                "candidate_index": 0,
                "grasp_center": np.asarray(data["grasp_center"], dtype=np.float64),
                "approach_dir": np.asarray(data["approach_dir"], dtype=np.float64),
                "grip_axis": np.asarray(data["grip_axis"], dtype=np.float64),
                "binormal": np.asarray(data["binormal"], dtype=np.float64),
                "rotation_matrix": np.asarray(data["rotation_matrix"], dtype=np.float64),
                "score": float(data["score"]) if "score" in data else 1.0,
            }
        )

    return candidates


def _normalize_xy(vec: np.ndarray) -> tuple[np.ndarray, float]:
    vec_xy = np.asarray(vec[:2], dtype=np.float64)
    norm = float(np.linalg.norm(vec_xy))
    if norm < 1e-8:
        return np.zeros(2, dtype=np.float64), 0.0
    return vec_xy / norm, norm


def select_best_push_candidate(
    candidates_local: list[dict],
    desired_push_delta_env: np.ndarray,
    obj_pos_world: np.ndarray,
    obj_quat_world: np.ndarray,
    env_origin: np.ndarray,
    ee_pos: np.ndarray,
    ee_quat: np.ndarray,
    motion_planner,
    current_joint_pos: np.ndarray,
    current_joint_vel: np.ndarray,
    joint_names: list[str],
    object_scale: np.ndarray,
    pre_height: float,
    pre_push_backoff: float,
    push_contact_retreat: float,
) -> tuple[Optional[dict], dict]:
    """Select the best affordance push pose for the desired push direction."""
    desired_dir_xy, desired_norm = _normalize_xy(desired_push_delta_env)
    if desired_norm < 1e-6:
        return None, {"candidates": [], "reason": "push_target_too_close"}

    rot_obj = Rotation.from_quat(
        [obj_quat_world[1], obj_quat_world[2], obj_quat_world[3], obj_quat_world[0]]
    ).as_matrix()

    scored_candidates = []
    for candidate in candidates_local:
        grasp_pos, grasp_quat = transform_grasp_to_world(
            grasp_center_local=candidate["grasp_center"],
            rotation_matrix_local=candidate["rotation_matrix"],
            obj_pos_world=obj_pos_world,
            obj_quat_world=obj_quat_world,
            env_origin=env_origin,
            reference_quat_wxyz=ee_quat,
            object_scale=object_scale,
        )

        approach_world = rot_obj @ np.asarray(candidate["approach_dir"], dtype=np.float64)
        approach_dir_xy, approach_norm = _normalize_xy(approach_world)
        if approach_norm < 1e-6:
            continue

        alignment = float(np.clip(np.dot(approach_dir_xy, desired_dir_xy), -1.0, 1.0))
        contact_pos = np.array(grasp_pos, dtype=np.float64)
        contact_pos[:2] -= desired_dir_xy * push_contact_retreat
        pre_pos = contact_pos.copy()
        pre_pos[:2] -= desired_dir_xy * pre_push_backoff
        pre_pos[2] += pre_height

        plan_success = False
        try:
            _, plan_success = motion_planner.plan_to_pose(
                current_joint_pos=current_joint_pos,
                current_joint_vel=current_joint_vel,
                joint_names=joint_names,
                target_pos=pre_pos,
                target_quat=grasp_quat,
                max_attempts=2,
            )
        except Exception as exc:
            print(f"    [WARN] Push candidate {candidate['candidate_index']} planning failed: {exc}")

        pos_dist = float(np.linalg.norm(grasp_pos - ee_pos))
        pos_score = max(0.0, 1.0 - pos_dist / 0.6)
        alignment_score = (alignment + 1.0) * 0.5
        total_score = alignment_score * 0.75 + float(plan_success) * 0.20 + pos_score * 0.05

        scored_candidates.append(
            {
                "candidate_index": candidate["candidate_index"],
                "grasp": candidate,
                "grasp_pos": np.asarray(grasp_pos, dtype=np.float64),
                "contact_pos": np.asarray(contact_pos, dtype=np.float64),
                "grasp_quat": np.asarray(grasp_quat, dtype=np.float64),
                "alignment": alignment,
                "alignment_score": alignment_score,
                "pos_score": pos_score,
                "plan_success": bool(plan_success),
                "total_score": float(total_score),
            }
        )

    scored_candidates.sort(
        key=lambda item: (item["plan_success"], item["total_score"]),
        reverse=True,
    )
    best_candidate = scored_candidates[0] if scored_candidates and scored_candidates[0]["plan_success"] else None
    selection_info = {
        "candidates": scored_candidates,
        "best_candidate_index": best_candidate["candidate_index"] if best_candidate else None,
        "desired_push_delta_env": np.asarray(desired_push_delta_env, dtype=np.float64),
    }
    return best_candidate, selection_info


def primitive_compute_push_pose(collector, params: Dict[str, Any], context: Dict[str, Any]) -> bool:
    target_label = params.get("target")
    if not target_label:
        print("[ERROR] Primitive 'compute_push_pose' requires a target object.")
        return False
    collector.target_prim_path = f"/World/envs/env_0/{target_label}"

    affordance_path = params.get("affordance_npz_path") or resolve_affordance_path(target_label, "push")
    if not affordance_path:
        print(
            f"[ERROR] Primitive 'push' could not resolve affordance data for '{target_label}'. "
            "Set params.affordance_npz_path or provide pc_dataset_generate/pc/{object}_push/grasp_result.npz."
        )
        return False

    print(f"  👉 [Primitive] Loading push affordance from: {affordance_path}")
    try:
        affordance_candidates = load_push_affordance_candidates(affordance_path)
    except Exception as exc:
        print(f"[ERROR] Primitive 'push' failed to load affordance data: {exc}")
        return False
    if not affordance_candidates:
        print(f"[ERROR] Primitive 'push' found no valid affordance candidates in: {affordance_path}")
        return False

    env_origins = collector._get_env_origins()
    ee_pos_w, ee_quat_w = collector.get_current_ee_pose()
    prev_success_mask = list(context.get("env_alive_mask", [True] * collector.num_envs))
    target_available_mask = compute_target_availability_mask(collector, target_label)
    log_target_missing_envs("push", target_label, target_available_mask, prev_success_mask)

    pre_height = float(params.get("pre_height", 0.15))
    if "pre_offset" in params:
        pre_offset_raw = params.get("pre_offset")
        pre_offset = np.array(pre_offset_raw, dtype=np.float64).reshape(-1)
        if pre_offset.size != 3:
            print(f"[ERROR] Primitive 'push' expects pre_offset=[x,y,z], got: {pre_offset_raw}")
            return False
        pre_height = float(pre_offset[2])
    pre_height = abs(pre_height)
    pre_push_backoff = float(params.get("pre_push_backoff", 0.08))
    pre_push_backoff = abs(pre_push_backoff)
    push_contact_retreat = float(params.get("push_contact_retreat", 0.02))
    push_contact_retreat = abs(push_contact_retreat)

    push_target_raw = params.get("push_target")
    if push_target_raw is None:
        print("[ERROR] Primitive 'push' requires push_target=[x,y,z].")
        return False
    push_target = np.array(push_target_raw, dtype=np.float64).reshape(-1)
    if push_target.size != 3:
        print(f"[ERROR] Primitive 'push' expects push_target=[x,y,z], got: {push_target_raw}")
        return False
    lift_z_delta_raw = params.get("lift_z_delta")
    should_lift = lift_z_delta_raw is not None
    lift_z_delta = float(lift_z_delta_raw) if should_lift else 0.0

    pre_positions = []
    pre_quats = []
    down_positions = []
    push_positions = []
    push_quats = []
    push_target_positions = []
    lift_positions = []
    lift_quats = []
    selection_infos = []

    object_config = None
    if collector.task_config:
        object_config = collector.task_config.get_object_by_name(target_label)
        if object_config is None:
            object_config = collector.task_config.articulations.get(target_label)
    object_scale = np.array([1.0, 1.0, 1.0], dtype=float)
    if object_config is not None and getattr(object_config, "scale", None) is not None:
        object_scale = np.array(object_config.scale, dtype=float)

    for env_idx in range(collector.num_envs):
        obj_pos, obj_quat = collector._get_object_pose_global(target_label, env_idx)
        if obj_pos is None:
            selection_infos.append(None)
            pre_positions.append(np.array([0.5, 0.0, 0.5], dtype=np.float64))
            pre_quats.append(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64))
            down_positions.append(np.array([0.5, 0.0, 0.4], dtype=np.float64))
            push_positions.append(np.array([0.5, 0.0, 0.4], dtype=np.float64))
            push_quats.append(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64))
            push_target_positions.append(push_target.copy())
            if should_lift:
                lift_positions.append(np.array([0.5, 0.0, 0.5], dtype=np.float64))
                lift_quats.append(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64))
            continue

        obj_pos_env = np.array(obj_pos - env_origins[env_idx], dtype=np.float64)
        desired_push_delta_env = push_target.copy() - obj_pos_env
        desired_push_delta_env[2] = 0.0

        env_ee_pos = ee_pos_w[env_idx] if ee_pos_w.ndim > 1 else ee_pos_w
        env_ee_quat = ee_quat_w[env_idx] if ee_quat_w.ndim > 1 else ee_quat_w

        collector._update_obstacles_for_env(env_idx)
        current_joint_pos = collector.robot.data.joint_pos[env_idx].cpu().numpy()
        current_joint_vel = collector.robot.data.joint_vel[env_idx].cpu().numpy()

        best_candidate, selection_info = select_best_push_candidate(
            candidates_local=affordance_candidates,
            desired_push_delta_env=desired_push_delta_env,
            obj_pos_world=obj_pos,
            obj_quat_world=obj_quat,
            env_origin=env_origins[env_idx],
            ee_pos=env_ee_pos,
            ee_quat=env_ee_quat,
            motion_planner=collector.motion_planner,
            current_joint_pos=current_joint_pos,
            current_joint_vel=current_joint_vel,
            joint_names=collector.robot.joint_names,
            object_scale=object_scale,
            pre_height=pre_height,
            pre_push_backoff=pre_push_backoff,
            push_contact_retreat=push_contact_retreat,
        )
        selection_infos.append(selection_info)

        if best_candidate is None:
            print(f"  ❌ Env {env_idx}: no reachable push affordance candidate matches target direction.")
            pre_positions.append(np.array([0.5, 0.0, 0.5], dtype=np.float64))
            pre_quats.append(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64))
            down_positions.append(np.array([0.5, 0.0, 0.4], dtype=np.float64))
            push_positions.append(np.array([0.5, 0.0, 0.4], dtype=np.float64))
            push_quats.append(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64))
            push_target_positions.append(push_target.copy())
            if should_lift:
                lift_positions.append(np.array([0.5, 0.0, 0.5], dtype=np.float64))
                lift_quats.append(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64))
            continue

        contact_pos = np.array(best_candidate["contact_pos"], dtype=np.float64)
        grasp_pos_world = np.array(best_candidate["grasp_pos"], dtype=np.float64)
        contact_quat = np.array(best_candidate["grasp_quat"], dtype=np.float64)
        push_dir_xy, push_dir_norm = _normalize_xy(desired_push_delta_env)
        pre = contact_pos.copy()
        pre[:2] -= push_dir_xy * pre_push_backoff
        pre[2] += pre_height
        down = contact_pos.copy()
        # Sweep endpoint uses grasp_pos (= object surface position) as the base; otherwise the
        # push_contact_retreat 2cm only closes the contact gap, and the final push falls short by this distance
        push = grasp_pos_world.copy()
        push[:2] += desired_push_delta_env[:2]

        pre_positions.append(pre)
        pre_quats.append(contact_quat)
        down_positions.append(down)
        push_positions.append(push)
        push_quats.append(contact_quat)
        push_target_positions.append(push_target.copy())
        print(
            f"  ✅ Env {env_idx}: selected push candidate {best_candidate['candidate_index']} "
            f"(alignment={best_candidate['alignment']:.3f}, total={best_candidate['total_score']:.3f})"
        )
        if should_lift:
            lift = push.copy()
            lift[2] += lift_z_delta
            lift_positions.append(lift)
            lift_quats.append(contact_quat)

    context["push_pose"] = {
        "target_label": target_label,
        "target_available_mask": list(target_available_mask),
        "pre_positions": np.array(pre_positions, dtype=np.float64),
        "pre_quats": np.array(pre_quats, dtype=np.float64),
        "down_positions": np.array(down_positions, dtype=np.float64),
        "push_positions": np.array(push_positions, dtype=np.float64),
        "push_quats": np.array(push_quats, dtype=np.float64),
        "push_target_positions": np.array(push_target_positions, dtype=np.float64),
        "lift_positions": np.array(lift_positions, dtype=np.float64) if should_lift else None,
        "lift_quats": np.array(lift_quats, dtype=np.float64) if should_lift else None,
        "selection_infos": selection_infos,
    }
    print(f"  👉 [Primitive] Computed push pose for '{target_label}'")
    return True


def primitive_execute_push(collector, params: Dict[str, Any], context: Dict[str, Any]) -> bool:
    push_pose = context.get("push_pose")
    if push_pose is None:
        print("[ERROR] Primitive 'execute_push' requires push poses first.")
        return False

    down_interp_steps = int(params.get("down_interpolation_steps", 5))
    push_interp_steps = int(params.get("push_interpolation_steps", 30))
    lift_interp_steps = int(params.get("lift_interpolation_steps", 10))
    prev_success_mask = list(context.get("env_alive_mask", [True] * collector.num_envs))
    target_available_mask = list(push_pose.get("target_available_mask", [True] * collector.num_envs))

    print(f"  👉 [Primitive] Executing push on '{push_pose.get('target_label')}'")
    collector._update_all_env_obstacles()
    pre_plans, pre_plan_successes = collector.motion_planner.plan_to_pose_batch(
        collector.robot.data.joint_pos[: collector.num_envs].cpu().numpy(),
        collector.robot.data.joint_vel[: collector.num_envs].cpu().numpy(),
        collector.robot.joint_names,
        np.array(push_pose["pre_positions"]),
        np.array(push_pose["pre_quats"]),
    )
    pre_successes = [prev and available and cur for prev, available, cur in zip(prev_success_mask, target_available_mask, pre_plan_successes)]
    collector.execute_trajectory_batch(
        pre_plans,
        [1.0 if s else 0.0 for s in pre_successes],
        desc="Primitive Push Pre",
        active_mask=pre_successes,
    )

    _, down_successes = collector.execute_ik_motion(
        target_positions=np.array(push_pose["down_positions"]),
        target_quaternions=np.array(push_pose["pre_quats"]),
        gripper_states=[1.0 if s else 0.0 for s in pre_successes],
        interpolation_steps=down_interp_steps,
        desc="Primitive Push Down",
        active_mask=pre_successes,
    )
    phase_successes = [a and b for a, b in zip(pre_successes, down_successes)]

    _, push_successes = collector.execute_ik_motion(
        target_positions=np.array(push_pose["push_positions"]),
        target_quaternions=np.array(push_pose["push_quats"]),
        gripper_states=[1.0 if s else 0.0 for s in phase_successes],
        interpolation_steps=push_interp_steps,
        desc="Primitive Push Sweep",
        active_mask=phase_successes,
    )
    phase_successes = [a and b for a, b in zip(phase_successes, push_successes)]

    final_successes = phase_successes
    if push_pose.get("lift_positions") is not None:
        _, lift_successes = collector.execute_ik_motion(
            target_positions=np.array(push_pose["lift_positions"]),
            target_quaternions=np.array(push_pose["lift_quats"]),
            gripper_states=[1.0 if s else 0.0 for s in phase_successes],
            interpolation_steps=lift_interp_steps,
            desc="Primitive Push Lift",
            active_mask=phase_successes,
        )
        final_successes = [a and b for a, b in zip(phase_successes, lift_successes)]

    collector.control_gripper_batch(
        [1.0 if s else 0.0 for s in final_successes],
        steps=FIXED_GRIPPER_STEPS,
        active_mask=final_successes,
    )

    success_eval_enabled = bool(params.get("success_eval", False))
    if success_eval_enabled:
        target_label = push_pose.get("target_label")
        if not target_label:
            print("[ERROR] Push success_eval requires target label.")
        else:
            state_successes = []
            state_metrics = []
            for env_idx in range(collector.num_envs):
                env_success, env_metrics = _evaluate_push_success_with_metrics(
                    scene=collector.scene,
                    push_target_label=target_label,
                    push_target_position_env=np.array(push_pose["push_target_positions"][env_idx], dtype=np.float64),
                    env_idx=env_idx,
                )
                state_successes.append(env_success)
                state_metrics.append(env_metrics)

            final_successes = [motion_ok and state_ok for motion_ok, state_ok in zip(final_successes, state_successes)]
            context["task_success_mask"] = list(final_successes)
            context["task_success_metrics"] = list(state_metrics)
            for env_idx, metrics in enumerate(state_metrics):
                dist_ok = metrics["push_xy_dist"] <= float(PUSH_SUCCESS_XY_THRESHOLD)
                upright_ok = metrics["object_local_z_world_z"] >= float(PUSH_SUCCESS_TILT_THRESHOLD)
                print(
                    f"    [TaskCheck][env {env_idx}] "
                    f"push_xy_dist={metrics['push_xy_dist']:.4f}m ({'OK' if dist_ok else 'FAIL'}), "
                    f"obj_local_z·world_z={metrics['object_local_z_world_z']:.3f} "
                    f"({'OK' if upright_ok else 'FAIL'})"
                )
            print(
                f"  ✅ [Primitive] Task success (push): "
                f"{sum(final_successes)}/{collector.num_envs} "
                f"(push_xy_dist<={PUSH_SUCCESS_XY_THRESHOLD:.3f})"
            )

    context["post_push_success_mask"] = list(final_successes)
    context["step_failure_reason_mask"] = [
        "inactive"
        if not prev_success_mask[env_idx]
        else "target_missing"
        if not target_available_mask[env_idx]
        else "planning_failed"
        if not final_successes[env_idx]
        else "none"
        for env_idx in range(collector.num_envs)
    ]
    context["env_alive_mask"] = list(final_successes)
    context["current_gripper_state"] = "open"
    context["push_completed"] = any(final_successes)
    context["primitive_executed_motion"] = True

    print(f"  ✅ [Primitive] Push success: {sum(final_successes)}/{collector.num_envs}")
    return any(final_successes)
