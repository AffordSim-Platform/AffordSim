from typing import Any, Dict, Optional
from functools import lru_cache

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..success.constants import OPEN_SUCCESS_MIN_ANGLE_RAD, PULL_SUCCESS_MIN_DISPLACEMENT
from ..success.evaluators import _evaluate_open_success_with_metrics, _evaluate_pull_success_with_metrics
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


@lru_cache(maxsize=64)
def _pull_joint_type_from_usd(usd_path: str, joint_name: str) -> str:
    """Use the asset's physical joint type rather than inferring units from its name."""
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise ValueError(f"Cannot inspect pull target USD: {usd_path}")
    matches = [
        prim for prim in stage.Traverse()
        if prim.GetName() == joint_name and prim.IsA(UsdPhysics.Joint)
    ]
    if len(matches) != 1:
        raise ValueError(f"Pull success joint '{joint_name}' must identify exactly one physical joint in {usd_path}")
    joint = matches[0]
    if joint.IsA(UsdPhysics.RevoluteJoint):
        return "revolute"
    if joint.IsA(UsdPhysics.PrismaticJoint):
        return "prismatic"
    raise ValueError(f"Unsupported pull success joint type {joint.GetTypeName()} in {usd_path}")


def _pull_success_threshold(joint_type: str, params: Dict[str, Any]) -> tuple[float, str]:
    """Return a physical threshold and its unit, honoring explicit same-unit values."""
    if joint_type == "revolute":
        keys = ("pull_success_min_angle_rad", "open_success_min_angle_rad", "min_angle_rad")
        incompatible = ("pull_success_min_displacement", "min_displacement")
        default, unit = OPEN_SUCCESS_MIN_ANGLE_RAD, "rad"
    elif joint_type == "prismatic":
        keys = ("pull_success_min_displacement", "min_displacement")
        incompatible = ("pull_success_min_angle_rad", "open_success_min_angle_rad", "min_angle_rad")
        default, unit = PULL_SUCCESS_MIN_DISPLACEMENT, "m"
    else:
        raise ValueError(f"Unsupported pull success joint type: {joint_type}")
    wrong_units = [key for key in incompatible if key in params]
    if wrong_units:
        raise ValueError(f"Pull {joint_type} joint requires a {unit} threshold; incompatible parameters: {wrong_units}")
    threshold = float(next((params[key] for key in keys if key in params), default))
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError(f"Pull success threshold must be finite and positive ({unit})")
    return threshold, unit


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

    n = collector.num_envs
    prev_alive = get_prev_alive_mask(context, n)
    local_active = init_local_active_mask(context, n)
    target_available_mask = list(
        context.get("grasp_pose", {}).get(
            "target_available_mask",
            compute_target_availability_mask(collector, target_label),
        )
    )
    log_target_missing_envs("pull", target_label, target_available_mask, prev_alive)
    for env_idx in range(n):
        if prev_alive[env_idx] and not target_available_mask[env_idx]:
            mark_local_inactive_with_reason(
                context, local_active, env_idx, FAILURE_REASON_TARGET_MISSING
            )
    success_eval_enabled = bool(params.get("success_eval", False))
    success_joint_name = params.get("success_joint_name")
    if success_eval_enabled and not success_joint_name:
        print("[ERROR] Primitive 'pull' with success_eval=true requires `params.success_joint_name` in YAML.")
        for i in range(n):
            local_active[i] = False
        commit_plan_failures(context, prev_alive, local_active)
        commit_task_failures(context, prev_alive, list(local_active), [False] * n)
        context["post_pull_success_mask"] = [False] * n
        context["env_alive_mask"] = [False] * n
        return False

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
        for i in range(n):
            local_active[i] = False
        commit_plan_failures(context, prev_alive, local_active)
        context["post_pull_success_mask"] = [False] * n
        context["env_alive_mask"] = [False] * n
        return False

    articulation = collector.scene.articulations[object_key]
    if success_eval_enabled:
        success_joint_type = _pull_joint_type_from_usd(str(articulation.cfg.spawn.usd_path), success_joint_name)
        success_threshold, success_unit = _pull_success_threshold(success_joint_type, params)

    # Compute world-frame pull direction per env
    all_pull_dirs_world = []
    for env_idx in range(n):
        root_quat_wxyz = articulation.data.root_quat_w[env_idx].cpu().numpy()
        root_rot = R.from_quat([root_quat_wxyz[1], root_quat_wxyz[2], root_quat_wxyz[3], root_quat_wxyz[0]])
        pull_dir_world = root_rot.apply(pull_direction_local)
        norm = np.linalg.norm(pull_dir_world)
        if norm > 1e-6:
            pull_dir_world /= norm
        all_pull_dirs_world.append(pull_dir_world)

    # Compute pre-grasp positions (offset along pull direction from handle)
    all_pre_grasp_pos = []
    for env_idx in range(n):
        pre_pos = all_pick_pos[env_idx] + pre_grasp_offset * all_pull_dirs_world[env_idx]
        all_pre_grasp_pos.append(pre_pos)

    # --- Phase 1: CuRobo plan to pre-grasp ---
    print(f"   [Primitive] Planning pull pre-grasp for '{target_label}'")
    pre_pick_plans, pre_pick_plan_successes = collector.motion_planner.plan_to_pose_batch(
        collector.robot.data.joint_pos[:n].cpu().numpy(),
        collector.robot.data.joint_vel[:n].cpu().numpy(),
        collector.robot.joint_names,
        np.array(all_pre_grasp_pos, dtype=np.float64),
        all_pick_quat,
        target_prim_path=collector.target_prim_path,
    )
    pre_segment = [local_active[i] and bool(pre_pick_plan_successes[i]) for i in range(n)]
    merge_segment_into_local_active(local_active, pre_segment)
    collector.execute_trajectory_batch(
        pre_pick_plans,
        [1.0 if s else 0.0 for s in local_active],
        desc="Primitive Pull Pre-Grasp",
        active_mask=list(local_active),
    )

    # --- Phase 2: IK descend to grasp position ---
    print("   [Primitive] Descending to grasp handle (IK)...")
    _, pick_env_successes = collector.execute_ik_motion(
        target_positions=all_pick_pos,
        target_quaternions=all_pick_quat,
        gripper_states=[1.0 if s else 0.0 for s in local_active],
        interpolation_steps=pick_interp_steps,
        desc="Primitive Pull Grasp Descent",
        active_mask=list(local_active),
    )
    merge_segment_into_local_active(local_active, pick_env_successes)

    # --- Phase 3: Close gripper ---
    collector.control_gripper_batch(
        [-1.0 if s else 0.0 for s in local_active],
        steps=FIXED_GRIPPER_STEPS,
        active_mask=list(local_active),
    )

    # --- Phase 4: IK pull along the pull direction ---
    print(f"   [Primitive] Pulling articulation '{target_label}' (distance={pull_distance:.3f}m)...")

    # Read current EE pose as pull start point
    ee_pos_cur, ee_quat_cur = collector.get_current_ee_pose()
    ee_pos_cur = np.atleast_2d(ee_pos_cur)
    ee_quat_cur = np.atleast_2d(ee_quat_cur)

    pull_targets = []
    pull_quats = []
    for env_idx in range(n):
        if not local_active[env_idx]:
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
        gripper_states=[-1.0 if s else 0.0 for s in local_active],
        interpolation_steps=pull_interp_steps,
        desc="Primitive Pull Drawer",
        active_mask=list(local_active),
    )
    merge_segment_into_local_active(local_active, pull_env_successes)
    plan_active_after = list(local_active)

    # --- Phase 5: Release gripper (optional) ---
    if release_after_pull:
        collector.control_gripper_batch(
            [1.0 if s else 0.0 for s in local_active],
            steps=FIXED_GRIPPER_STEPS,
            active_mask=list(local_active),
        )

    if success_eval_enabled:
        state_successes = []
        state_metrics = []
        for env_idx in range(n):
            if success_joint_type == "revolute":
                env_success, env_metrics = _evaluate_open_success_with_metrics(
                    scene=collector.scene, target_label=target_label,
                    joint_name=success_joint_name, min_angle_rad=success_threshold,
                    env_idx=env_idx,
                )
            else:
                env_success, env_metrics = _evaluate_pull_success_with_metrics(
                    scene=collector.scene, target_label=target_label,
                    joint_name=success_joint_name, min_displacement=success_threshold,
                    env_idx=env_idx,
                )
            env_metrics.update({
                "joint_name": success_joint_name, "joint_type": success_joint_type,
                "joint_unit": success_unit, "success_threshold": success_threshold,
            })
            state_successes.append(env_success)
            state_metrics.append(env_metrics)

        commit_task_failures(context, prev_alive, plan_active_after, list(state_successes))
        context["task_success_metrics"] = list(state_metrics)
        print(
            f"  [Primitive] Task success (pull): "
            f"{sum(1 for i in range(n) if plan_active_after[i] and state_successes[i])}/{n} "
            f"({success_joint_type} |joint_pos|>={success_threshold:.3f} {success_unit})"
        )

    commit_plan_failures(context, prev_alive, local_active)

    plan_mask_final = list(context.get("plan_success_mask", [True] * n))
    task_mask_final = list(context.get("task_success_mask", [True] * n))
    final_alive = [
        bool(prev_alive[i] and plan_mask_final[i] and task_mask_final[i])
        for i in range(n)
    ]
    context["post_pull_success_mask"] = list(final_alive)
    context["env_alive_mask"] = list(final_alive)
    context["current_gripper_state"] = "open" if release_after_pull else "closed"
    context["pull_completed"] = any(final_alive)
    context["primitive_executed_motion"] = True

    print(f"  [Primitive] Pull success (alive): {sum(final_alive)}/{n}")
    return any(final_alive)
