"""
Pi0.5 evaluation script for Isaac Lab environments.

Uses openpi's websocket policy server for remote inference.
Server should be started separately:
    .venv/bin/python scripts/serve_policy.py policy:checkpoint \
        --policy.config=pi05_custom_hdf5 \
        --policy.dir=checkpoints/pi05_custom_hdf5/my_experiment/20000

Usage:
    python play.py --task Pick_Place_banana-v0 \
        --server_host 10.184.17.177 --server_port 8000 \
        --task_config path/to/task_config.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Pi0.5 Eval in Isaac Lab")
parser.add_argument("--task", type=str, default="Pick_Place_banana-v0", help="Environment name")
parser.add_argument("--server_host", type=str, default="10.184.17.177", help="Pi0.5 policy server host")
parser.add_argument("--server_port", type=int, default=8000, help="Pi0.5 policy server port")
parser.add_argument("--prompt", type=str, default="pick banana and place it on the white plate",
                    help="Language instruction for the policy")
parser.add_argument("--horizon", type=int, default=400, help="Max steps per rollout")
parser.add_argument("--num_rollouts", type=int, default=5, help="Number of rollouts")
parser.add_argument("--seed", type=int, default=None, help="Random seed")
parser.add_argument("--action_horizon", type=int, default=10,
                    help="Number of actions to execute per prediction (should match model config)")
parser.add_argument(
    "--task_config", type=str,
    default="source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/task_gen/config/sceneconfig/generated_task_config.yaml",
    help="YAML task config path",
)
parser.add_argument("--output_json", type=str, default=None,
                    help="Path to save JSON results. Default: debug_images/pi05_eval_results.json")
parser.add_argument("--pick_lift_threshold", type=float, default=0.05)
parser.add_argument(
    "--action_type", type=str, default=None, choices=["ee_delta", "joint_delta"],
    help=(
        "Action semantics the policy server returns. ee_delta=6D EE delta+gripper (env 用 IK-rel, 直接下发); "
        "joint_delta=7D joint delta+gripper (env 用 JointPosition 绝对, play 时按 --joint_delta_mode 展开). "
        "未指定则按 task id 自动推断: 含 'joint' → joint_delta, 否则 ee_delta."
    ),
)
parser.add_argument(
    "--joint_delta_mode", type=str, default="cumulative", choices=["current", "cumulative"],
    help=(
        "joint_delta -> 绝对位置的展开方式 (仅 action_type=joint_delta 时生效). "
        "'current'   : arm_target(k) = cur_qpos(k) + delta(k), 每步重新读最新 qpos, 偏差自我修正; "
        "'cumulative': arm_target(k) = arm_target(k-1) + delta(k), 初值=首帧 arm qpos, 与训练数据一阶差分语义一致, 但对执行偏差无反馈."
    ),
)
parser.add_argument(
    "--linger_frames", type=int, default=10,
    help=(
        "整个 play 流程结束 (early_success / terminated / horizon 到) 后, 用 freeze 动作再多跑这么多帧, "
        "用来观察 place/pour/hang 完成后环境是否突变 (物体掉落/翻倒). "
        "freeze 动作: arm 不动 (ee_delta=0 或 joint_delta=cur_qpos), gripper=最后一帧的 gripper 命令."
    ),
)
parser.add_argument(
    "--warmup_action", type=str, default=None, choices=["zero", "hold"],
    help=(
        "rollout 前的 warmup 模式. 'zero' 适合 delta 模式; 'hold' (arm=当前 qpos, gripper=-1/open) 适合 "
        "JointPosition 绝对模式 (否则 zero 会把 robot 命令到全零关节位形). "
        "未指定则按 --action_type 推断 (joint_delta → hold, 其它 → zero)."
    ),
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Auto-infer action_type / warmup_action from task id if not given.
# 约定: task id 字符串里含 "joint" 的都是 *_joint-v0 绑 JointPositionActionCfg 绝对模式.
if args_cli.action_type is None:
    args_cli.action_type = "joint_delta" if "joint" in args_cli.task.lower() else "ee_delta"
    print(f"[EVAL] --action_type 未指定, 按 task='{args_cli.task}' 推断为 '{args_cli.action_type}'")
if args_cli.warmup_action is None:
    args_cli.warmup_action = "hold" if args_cli.action_type == "joint_delta" else "zero"
    print(f"[EVAL] --warmup_action 未指定, 按 action_type='{args_cli.action_type}' 推断为 '{args_cli.warmup_action}'")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Imports after Isaac Sim launch ---
import cv2
import gymnasium as gym
import numpy as np
import random
import torch
import yaml
from PIL import Image

from isaaclab_tasks.utils import parse_env_cfg
import isaaclab_tasks.real2sim.tasks  # noqa: F401
from isaaclab_tasks.real2sim.utils.EpisodeBuffer import replace_background_with_segmentation

from isaaclab.motion_collect.success.evaluators import (
    _evaluate_pick_success_with_metrics,
    _evaluate_place_pose_success_with_metrics,
    _evaluate_place_on_success_with_metrics,
    _evaluate_place_affordance_xy_success_with_metrics,
    _evaluate_pour_success_with_metrics,
    _evaluate_hang_success_with_metrics,
    _evaluate_pull_success_with_metrics,
    _evaluate_open_success_with_metrics,
    _evaluate_close_success_with_metrics,
    _evaluate_push_success_with_metrics,
)
from isaaclab.motion_collect.success.scene_helpers import (
    _get_articulation_joint_pos,
    _get_object_pose_world_from_scene,
    attach_scene_success_meta,
)
from isaaclab.motion_collect.tools.pose_utils import _quat_wxyz_to_matrix
from isaaclab.motion_collect.tools.affordance_resolver import resolve_affordance_path
from isaaclab.motion_collect.primitives.utils import load_affordance_centroid


from isaaclab.motion_collect.success.success import (
    capture_object_initial_z,
)

from openpi_client import websocket_client_policy as _websocket_client_policy


# ============================================================================
# Image processing
# ============================================================================

def load_background_images(task_config_path):
    """Load background images from YAML config."""
    with open(task_config_path) as f:
        cfg = yaml.safe_load(f)
    runtime = cfg.get("runtime", {})
    project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    config_dir = os.path.dirname(os.path.abspath(task_config_path))

    bg_images = {}
    for key, bg_key in [("front", "cam_side_background"), ("wrist", "cam_wrist_background")]:
        bg_path = runtime.get(bg_key)
        if bg_path:
            if os.path.isabs(bg_path):
                full_path = bg_path
            else:
                full_path = os.path.normpath(os.path.join(project_root, bg_path))
                if not os.path.exists(full_path):
                    cwd_path = os.path.abspath(bg_path)
                    full_path = cwd_path if os.path.exists(cwd_path) else os.path.normpath(
                        os.path.join(config_dir, bg_path)
                    )
            img = cv2.imread(full_path)
            if img is not None:
                bg_images[key] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                print(f"[INFO] Loaded background image {key}: {full_path} -> {img.shape}")
            else:
                print(f"[WARN] Failed to load background image {key}: {full_path}")
    return bg_images


def process_image(image_tensor, seg_tensor=None, bg_image=None):
    """Process RGB image: background replace + center crop.
    Returns numpy array (H, W, 3) uint8 RGB.
    """
    image = image_tensor.clone().float()
    C, H, W = image.shape

    if bg_image is not None and seg_tensor is not None:
        seg = seg_tensor.cpu().numpy().reshape(4, H, W).transpose(1, 2, 0)
        rgb_np = image.cpu().numpy().transpose(1, 2, 0)
        composite, _ = replace_background_with_segmentation(rgb_np, seg, bg_image)
        image = torch.from_numpy(composite.transpose(2, 0, 1)).float()

    crop_size = min(H, W)
    y0 = (H - crop_size) // 2
    x0 = (W - crop_size) // 2
    image = image[:, y0:y0 + crop_size, x0:x0 + crop_size]

    img_np = (image.cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    return img_np


def _apply_speckle_noise_u8(img_u8: np.ndarray, std: float) -> np.ndarray:
    """Apply speckle (multiplicative) noise to a uint8 RGB image.

    Matches ``EpisodeBuffer._add_speckle_noise``:
        y = x + x * N(0, std),  x in [0, 1]
    so the eval-time distribution matches train-time data collection.
    """
    if std <= 0 or img_u8 is None:
        return img_u8
    x = img_u8.astype(np.float32) / 255.0
    noise = np.random.normal(0.0, std, x.shape).astype(np.float32)
    y = np.clip(x + x * noise, 0.0, 1.0)
    return (y * 255.0).astype(np.uint8)


# ============================================================================
# Task YAML -> eval stages
# ============================================================================

def _get_object_scale_from_cfg(cfg: dict, object_name: str) -> np.ndarray:
    """Return object scale from YAML config, default [1,1,1]."""
    objects = cfg.get("objects") or {}
    obj_cfg = objects.get(object_name) or {}
    scale = obj_cfg.get("scale")
    if scale is not None:
        return np.asarray(scale, dtype=np.float64)
    return np.ones(3, dtype=np.float64)


def _get_object_height_from_cfg(cfg: dict, object_name: str) -> float:
    """Return object height from YAML config, default 0.1."""
    objects = cfg.get("objects") or {}
    obj_cfg = objects.get(object_name) or {}
    return float(obj_cfg.get("height", 0.1))


def load_eval_stages(task_config_path: str) -> list[dict]:
    with open(task_config_path) as f:
        cfg = yaml.safe_load(f) or {}
    primitives = cfg.get("primitives") or []
    pick_target_name = None
    for entry in primitives:
        if isinstance(entry, dict) and str(entry.get("type", "")).lower() == "pick":
            pick_target_name = (entry.get("params") or {}).get("target")
            break
    stages = []
    for entry in primitives:
        if not isinstance(entry, dict):
            continue
        ptype = str(entry.get("type", "")).lower()
        if ptype not in ("pick", "place", "pour", "hang", "pull", "open", "close", "push"):
            continue
        params = entry.get("params") or {}
        target_name = params.get("target")
        stage = {
            "type": ptype,
            "target": target_name,
            "place_target": target_name if ptype == "place" else None,
            "place_mode": str(entry.get("mode", "")).lower() if ptype == "place" else None,
            "place_position": params.get("position") if ptype == "place" else None,
            "place_orientation": params.get("orientation") if ptype == "place" else None,
            "pick_affordance_npz_path": params.get("pick_affordance_npz_path") if ptype == "place" else None,
            "target_affordance_npz_path": params.get("target_affordance_npz_path") if ptype == "place" else None,
            "pick_object_scale": _get_object_scale_from_cfg(cfg, pick_target_name) if ptype == "place" and pick_target_name else None,
            "place_object_scale": _get_object_scale_from_cfg(cfg, target_name) if ptype == "place" and target_name else None,
            "place_target_height": _get_object_height_from_cfg(cfg, target_name) if ptype == "place" and target_name else None,
            "place_on_height_margin": float(params.get("on_height_margin", 0.01)) if ptype == "place" else None,
            "pour_target": target_name if ptype == "pour" else None,
            "hang_container": target_name if ptype == "hang" else None,
            "hang_container_mount_npz": params.get("container_mount_affordance_npz_path") if ptype == "hang" else None,
            "pull_target": target_name if ptype == "pull" else None,
            "open_target": target_name if ptype == "open" else None,
            "open_joint_name": params.get("joint_name") if ptype == "open" else None,
            "close_target": target_name if ptype == "close" else None,
            "close_joint_name": params.get("joint_name") if ptype == "close" else None,
            "push_target_label": target_name if ptype == "push" else None,
            "push_target_position": params.get("push_target") if ptype == "push" else None,
        }
        stages.append(stage)
    return stages


def resolve_pick_target(stages: list[dict]) -> str | None:
    for s in stages:
        if s["type"] == "pick" and s.get("target"):
            return s["target"]
    return None


def resolve_unique_pick_targets(stages: list[dict]) -> list[str]:
    """Unique pick targets in encounter order (stack tasks pick more than one object)."""
    out: list[str] = []
    for s in stages:
        if s["type"] == "pick":
            tgt = s.get("target")
            if tgt and tgt not in out:
                out.append(tgt)
    return out


# ============================================================================
# Stage evaluation
# ============================================================================

def evaluate_stages_at_episode_end(
    scene,
    stages: list[dict],
    pick_initial_z: dict[str, float] | float | None,
    pick_lift_threshold: float,
    pick_peak_z: dict[str, float] | float | None = None,
    pull_peak_displacement: float | None = None,
    pour_min_tilt_z: float | None = None,
) -> dict[str, dict]:
    # 兼容: 旧签名传 float (单 pick_target). 新签名传 dict[target -> z], 支持 stack 多 pick.
    if not isinstance(pick_initial_z, dict):
        legacy_target = resolve_pick_target(stages)
        pick_initial_z = {legacy_target: pick_initial_z} if legacy_target else {}
    if not isinstance(pick_peak_z, dict):
        legacy_target = resolve_pick_target(stages)
        pick_peak_z = {legacy_target: pick_peak_z} if (legacy_target and pick_peak_z is not None) else {}

    results: dict[str, dict] = {}
    held_target: str | None = None  # 最近一次 pick 抓的物体, place/pour/hang 用它

    for idx, stage in enumerate(stages):
        ptype = stage["type"]
        key = f"{idx:02d}_{ptype}"

        if ptype == "pick":
            stage_target = stage.get("target")
            if stage_target:
                held_target = stage_target
            init_z = pick_initial_z.get(stage_target) if stage_target else None
            if init_z is None or not stage_target:
                results[key] = {"success": False, "reason": "no_initial_z_or_target",
                                "target": stage_target}
                continue
            ok, metrics = _evaluate_pick_success_with_metrics(
                scene=scene,
                pick_target=stage_target,
                initial_z_world=init_z,
                env_idx=0,
                lift_threshold=pick_lift_threshold,
                peak_z_world=pick_peak_z.get(stage_target),
            )
            results[key] = {"success": bool(ok), "metrics": metrics}

        elif ptype == "place":
            place_mode = stage.get("place_mode") or "on"
            place_target = stage.get("place_target")
            if place_mode == "pose":
                target_pos = stage.get("place_position")
                target_quat = stage.get("place_orientation")
                if target_pos is None:
                    results[key] = {"success": False, "reason": "no_place_position"}
                    continue
                ok, metrics = _evaluate_place_pose_success_with_metrics(
                    scene=scene,
                    pick_target=held_target,
                    target_position_env=np.array(target_pos, dtype=float),
                    target_quat_wxyz=np.array(target_quat, dtype=float) if target_quat is not None else None,
                    env_idx=0,
                )
            elif place_mode == "affordance":
                if not place_target or not held_target:
                    results[key] = {"success": False, "reason": "no_place_or_pick_target"}
                    continue
                pick_aff_path = stage.get("pick_affordance_npz_path") or resolve_affordance_path(held_target, "place")
                target_aff_path = stage.get("target_affordance_npz_path") or resolve_affordance_path(place_target, "place")
                if not pick_aff_path or not target_aff_path:
                    results[key] = {"success": False, "reason": "missing_place_affordance_npz",
                                    "pick_aff": pick_aff_path, "target_aff": target_aff_path}
                    continue
                try:
                    pick_aff_local = load_affordance_centroid(pick_aff_path)
                    target_aff_local = load_affordance_centroid(target_aff_path)
                except Exception as e:
                    results[key] = {"success": False, "reason": f"affordance_load_failed:{e}"}
                    continue
                pick_scale = stage.get("pick_object_scale")
                place_scale = stage.get("place_object_scale")
                if pick_scale is not None:
                    pick_aff_local = pick_aff_local * np.asarray(pick_scale, dtype=np.float64)
                if place_scale is not None:
                    target_aff_local = target_aff_local * np.asarray(place_scale, dtype=np.float64)
                ok, metrics = _evaluate_place_affordance_xy_success_with_metrics(
                    scene=scene, pick_target=held_target, place_target=place_target,
                    pick_aff_local=pick_aff_local, target_aff_local=target_aff_local, env_idx=0,
                )
            else:
                if not place_target or not held_target:
                    results[key] = {"success": False, "reason": "no_place_or_pick_target"}
                    continue
                ok, metrics = _evaluate_place_on_success_with_metrics(
                    scene=scene, pick_target=held_target, place_target=place_target,
                    place_target_height=stage.get("place_target_height") or 0.1,
                    on_height_margin=stage.get("place_on_height_margin") or 0.01,
                    env_idx=0,
                )
            results[key] = {"success": bool(ok), "metrics": metrics}

        elif ptype == "pour":
            pour_target = stage.get("pour_target")
            if not pour_target or not held_target:
                results[key] = {"success": False, "reason": "no_pour_or_pick_target"}
                continue
            ok, metrics = _evaluate_pour_success_with_metrics(
                scene=scene,
                pick_target=held_target,
                pour_target=pour_target,
                env_idx=0,
                min_z_component_observed=pour_min_tilt_z,
            )
            results[key] = {"success": bool(ok), "metrics": metrics}

        elif ptype == "hang":
            hang_container = stage.get("hang_container")
            if not hang_container or not held_target:
                results[key] = {"success": False, "reason": "no_hang_container_or_pick_target"}
                continue
            mount_npz = stage.get("hang_container_mount_npz")
            try:
                if not mount_npz:
                    import importlib
                    _afford_mod = importlib.import_module("isaaclab.motion_collect.tools.affordance_resolver")
                    mount_npz = _afford_mod.resolve_affordance_path(hang_container, "hang")
                if not mount_npz:
                    results[key] = {"success": False, "reason": "no_container_mount_affordance"}
                    continue
                npz_data = np.load(mount_npz, allow_pickle=True)
                if "centroids" in npz_data:
                    centroids = np.asarray(npz_data["centroids"], dtype=np.float64)
                    container_mount_local = np.array(centroids[0], dtype=np.float64)
                elif "grasp_center" in npz_data:
                    container_mount_local = np.asarray(npz_data["grasp_center"], dtype=np.float64)
                else:
                    results[key] = {"success": False, "reason": "invalid_mount_npz_keys"}
                    continue
            except Exception as e:
                results[key] = {"success": False, "reason": f"mount_load_failed:{e}"}
                continue

            ok, metrics = _evaluate_hang_success_with_metrics(
                scene=scene,
                held_object_label=held_target,
                container_label=hang_container,
                container_mount_local=container_mount_local,
                env_idx=0,
            )
            results[key] = {"success": bool(ok), "metrics": metrics}

        elif ptype == "pull":
            pull_target = stage.get("pull_target")
            if not pull_target:
                results[key] = {"success": False, "reason": "no_pull_target"}
                continue
            ok, metrics = _evaluate_pull_success_with_metrics(
                scene=scene,
                target_label=pull_target,
                env_idx=0,
                peak_displacement=pull_peak_displacement,
            )
            results[key] = {"success": bool(ok), "metrics": metrics}

        elif ptype == "open":
            open_target = stage.get("open_target")
            if not open_target:
                results[key] = {"success": False, "reason": "no_open_target"}
                continue
            ok, metrics = _evaluate_open_success_with_metrics(
                scene=scene,
                target_label=open_target,
                joint_name=stage.get("open_joint_name"),
                env_idx=0,
            )
            results[key] = {"success": bool(ok), "metrics": metrics}

        elif ptype == "close":
            close_target = stage.get("close_target")
            if not close_target:
                results[key] = {"success": False, "reason": "no_close_target"}
                continue
            ok, metrics = _evaluate_close_success_with_metrics(
                scene=scene,
                target_label=close_target,
                joint_name=stage.get("close_joint_name"),
                env_idx=0,
            )
            results[key] = {"success": bool(ok), "metrics": metrics}

        elif ptype == "push":
            push_label = stage.get("push_target_label")
            push_pos = stage.get("push_target_position")
            if not push_label or push_pos is None:
                results[key] = {"success": False, "reason": "no_push_target_or_position"}
                continue
            ok, metrics = _evaluate_push_success_with_metrics(
                scene=scene,
                push_target_label=push_label,
                push_target_position_env=np.array(push_pos, dtype=float),
                env_idx=0,
            )
            results[key] = {"success": bool(ok), "metrics": metrics}

    return results


# ============================================================================
# GIF utils
# ============================================================================

def frames_to_gif(frames: list[np.ndarray], save_path: str,
                  target_frames: int = 30, target_size: int = 256) -> str | None:
    if not frames:
        return None

    n = len(frames)
    if n > target_frames:
        indices = np.linspace(0, n - 1, target_frames, dtype=int)
    else:
        indices = np.arange(n)
    sampled = [frames[i] for i in indices]

    pil_frames = []
    for f in sampled:
        img = Image.fromarray(f)
        img = img.resize((target_size, target_size), Image.BILINEAR)
        pil_frames.append(img)

    duration_per_frame = max(5000 // len(pil_frames), 50)
    pil_frames[0].save(
        save_path, save_all=True, append_images=pil_frames[1:],
        duration=duration_per_frame, loop=0,
    )
    return save_path


# ============================================================================
# Rollout
# ============================================================================

def _build_warmup_action(mode: str, env, obs_dict, device):
    """(1, D_act) warmup action.

    - "zero": 全零, 适合 delta action 模式 (IK-rel / JointDelta), delta=0 ≡ 不动.
    - "hold": arm ← 当前 qpos, gripper ← -1 (open). 给 JointPosition 绝对模式用,
              否则 zeros() 会把 robot 命令到全零关节位形, 砸坏场景.
    """
    D = env.action_space.shape[1]
    if mode == "zero":
        return torch.zeros(1, D, device=device)
    if mode == "hold":
        qpos = obs_dict["policy"]["qpos"].squeeze().detach().to(device=device, dtype=torch.float32)
        arm_dim = D - 1
        hold_arm = qpos[:arm_dim]
        gripper = torch.tensor([-1.0], device=device, dtype=torch.float32)
        return torch.cat([hold_arm, gripper]).view(1, -1)
    raise ValueError(f"unknown warmup_action mode: {mode}")


def rollout(policy, env, horizon, action_horizon, device, prompt,
            bg_images=None, trial_id=0, initial_obs_dict=None,
            pick_targets=None, pull_target=None, pour_target=None,
            primary_pick_target=None,
            collect_front_images=True,
            image_speckle_noise_std: float = 0.0,
            stages=None, pick_initial_z=None, pick_lift_threshold: float = 0.05,
            early_stop_check_every: int = 10,
            warmup_action: str = "zero", action_type: str = "ee_delta",
            joint_delta_mode: str = "cumulative",
            linger_frames: int = 0):
    """Run one episode with Pi0.5 remote policy.

    pick_targets: 全部 pick 物体名 (stack 任务有多个), 用于 per-target 追踪 peak_z.
    primary_pick_target: 倾倒类任务里用来判断「靠近 pour_target」的主物体, 默认取首个 pick.
    """
    debug_dir = f"debug_images/pi05_trial_{trial_id}"
    os.makedirs(debug_dir, exist_ok=True)

    if initial_obs_dict is None:
        obs_dict, _ = env.reset()
    else:
        obs_dict = initial_obs_dict

    seg_keys = {"front": "semantic_segmentation_side", "wrist": "semantic_segmentation_wrist"}
    env_cam_h, env_cam_w = 480, 640

    pick_targets = list(pick_targets or [])
    if primary_pick_target is None and pick_targets:
        primary_pick_target = pick_targets[0]

    front_frames = []
    pick_peak_z: dict[str, float] = {tgt: None for tgt in pick_targets}  # type: ignore[assignment]
    pull_peak_displacement = None
    pour_min_tilt_z = None  # min z_component (max tilt) when pick obj within 10cm of pour target
    # joint_delta: 一次性记下第一帧 arm qpos, 之后每个 delta 都累加到它上面 (而不是当前 qpos),
    # 即 arm_target(k) = arm_target(k-1) + delta(k), 初值 = 首帧 arm qpos.
    accumulated_arm = None

    POUR_NEAR_RADIUS = 0.15  # 20cm. 之前 10cm 经常漏掉真正的倾倒动作:
    # 壶嘴虽然在容器上方, 但壶身中心可能已经 12-15cm 外,
    # 导致 pour_min_tilt_z 在真正倾倒时没被更新.

    def _update_peak():
        nonlocal pull_peak_displacement, pour_min_tilt_z
        if pull_target:
            from isaaclab.motion_collect.success.constants import PULL_SUCCESS_JOINT_NAME
            jp = _get_articulation_joint_pos(env.scene, pull_target, PULL_SUCCESS_JOINT_NAME, 0)
            if jp is not None:
                disp = abs(float(jp))
                if pull_peak_displacement is None or disp > pull_peak_displacement:
                    pull_peak_displacement = disp
        for tgt in pick_targets:
            pos_w, quat_w = _get_object_pose_world_from_scene(env.scene, tgt, 0)
            if pos_w is None:
                continue
            z = float(pos_w[2])
            cur = pick_peak_z.get(tgt)
            if cur is None or z > cur:
                pick_peak_z[tgt] = z
            # 倾倒判定: 仅对 primary pick (壶/容器) 跟踪倾斜.
            if tgt == primary_pick_target and pour_target and quat_w is not None:
                pour_pos_w, _ = _get_object_pose_world_from_scene(env.scene, pour_target, 0)
                if pour_pos_w is not None:
                    xy_dist = float(np.linalg.norm(pos_w[:2] - pour_pos_w[:2]))
                    if xy_dist <= POUR_NEAR_RADIUS:
                        rot = _quat_wxyz_to_matrix(np.array(quat_w, dtype=float))
                        tilt_z = float(rot[2, 2])
                        if pour_min_tilt_z is None or tilt_z < pour_min_tilt_z:
                            pour_min_tilt_z = tilt_z

    def _check_early_success():
        """全部 stage 都已成功 -> True, 用于早停."""
        if stages is None:
            return False
        res = evaluate_stages_at_episode_end(
            scene=env.scene, stages=stages,
            pick_initial_z=pick_initial_z,
            pick_lift_threshold=pick_lift_threshold,
            pick_peak_z=pick_peak_z,
            pull_peak_displacement=pull_peak_displacement,
            pour_min_tilt_z=pour_min_tilt_z,
        )
        return bool(res) and all(r.get("success") for r in res.values())

    step = 0
    early_success = False
    last_gripper_cmd = -1.0  # 最后一次下发给 env 的 gripper 命令, linger 阶段保留它 (不要松手)
    while step < horizon and not early_success:
        obs = obs_dict["policy"]

        # Skip first frame (renderer not ready)
        if step == 0:
            warmup = _build_warmup_action(warmup_action, env, obs_dict, device)
            obs_dict, _, _, _, _ = env.step(warmup)
            step += 1
            if collect_front_images and "front" in obs_dict["policy"]:
                _fr = obs_dict["policy"]["front"].squeeze().float().reshape(3, env_cam_h, env_cam_w)
                _seg = obs_dict["policy"].get(seg_keys["front"], None)
                if _seg is not None:
                    _seg = _seg.squeeze()
                _bg = bg_images.get("front") if bg_images else None
                front_frames.append(process_image(_fr, _seg, _bg))
            continue

        # --- Process images ---
        front_img = None
        wrist_img = None
        for image_name in ["front", "wrist"]:
            if image_name in obs:
                img_t = obs[image_name].squeeze().float().reshape(3, env_cam_h, env_cam_w)
                seg_key = seg_keys[image_name]
                seg = obs.get(seg_key, None)
                if seg is not None:
                    seg = seg.squeeze()
                bg = bg_images.get(image_name) if bg_images else None
                img_np = process_image(img_t, seg, bg)
                if image_name == "front":
                    front_img = img_np
                else:
                    wrist_img = img_np

        # Save debug images for first few steps
        if step <= 5:
            if front_img is not None:
                Image.fromarray(front_img).save(os.path.join(debug_dir, f"step_{step:04d}_front.png"))
            if wrist_img is not None:
                Image.fromarray(wrist_img).save(os.path.join(debug_dir, f"step_{step:04d}_wrist.png"))

        # Fill missing images with zeros
        if front_img is None:
            front_img = np.zeros((256, 256, 3), dtype=np.uint8)
        if wrist_img is None:
            wrist_img = np.zeros((256, 256, 3), dtype=np.uint8)

        # Apply speckle noise only to policy inputs (GIF frames and debug PNGs
        # were captured before this point, so they remain clean).
        if image_speckle_noise_std > 0:
            front_img = _apply_speckle_noise_u8(front_img, image_speckle_noise_std)
            wrist_img = _apply_speckle_noise_u8(wrist_img, image_speckle_noise_std)

        # --- Build observation for Pi0.5 ---
        qpos = obs["qpos"].squeeze().cpu().numpy().astype(np.float32)
        # qvel = obs["qvel"].squeeze().cpu().numpy().astype(np.float32) if "qvel" in obs else np.zeros_like(qpos)
        # state = np.concatenate([qpos, qvel]).astype(np.float32)
        state = np.concatenate([qpos]).astype(np.float32)

        pi05_obs = {
            "observation/image": front_img,          # (H, W, 3) uint8
            "observation/wrist_image": wrist_img,    # (H, W, 3) uint8
            "observation/state": state,              # (14,) float32
            "prompt": prompt,
        }

        # --- Query remote policy server ---
        result = policy.infer(pi05_obs)
        actions_chunk = np.asarray(result["actions"])  # (action_horizon, A)

        # print(f"[EVAL] policy returned actions_chunk shape={actions_chunk.shape}, "
        #       f"env action_dim={env.action_space.shape[1]}")
        # print(f"[EVAL] actions_chunk[0]={np.array2string(actions_chunk[0], precision=4, suppress_small=True)}")

        # --- Execute actions ---
        n_exec = min(action_horizon, actions_chunk.shape[0], horizon - step)
        terminated = False
        truncated = False
        for idx in range(n_exec):
            action = actions_chunk[idx]  # 7 or 8 dim, auto from server
            # joint_delta: server 返回的是关节角 delta, env 是 JointPositionActionCfg 绝对模式.
            # 两种展开方式 (--joint_delta_mode 控制), gripper 维直通:
            #   "current"   : arm_target(k) = cur_qpos(k) + delta(k), 偏差自我修正
            #   "cumulative": arm_target(k) = arm_target(k-1) + delta(k), 初值=首帧 arm qpos
            if action_type == "joint_delta":
                if joint_delta_mode == "current":
                    cur_qpos = obs_dict["policy"]["qpos"].squeeze().detach().cpu().numpy().astype(action.dtype)
                    arm_abs = cur_qpos[:action.shape[0] - 1] + action[:-1]
                    action_out = np.concatenate([arm_abs, action[-1:]], axis=0)
                else:  # "cumulative"
                    if accumulated_arm is None:
                        cur_qpos = obs_dict["policy"]["qpos"].squeeze().detach().cpu().numpy().astype(action.dtype)
                        accumulated_arm = cur_qpos[:action.shape[0] - 1].copy()
                    accumulated_arm = accumulated_arm + action[:-1]
                    action_out = np.concatenate([accumulated_arm, action[-1:]], axis=0)
            else:
                action_out = action
            action_tensor = torch.from_numpy(action_out).float().to(device).view(1, -1)
            last_gripper_cmd = float(action_out[-1])
            obs_dict, _, terminated, truncated, _ = env.step(action_tensor)
            step += 1
            if step <= 5 or step % 50 == 0:
                print(f"[EVAL] step={step}, action({action_type})={np.array2string(action_out, precision=4, suppress_small=True)}")
            _update_peak()

            if collect_front_images and "front" in obs_dict["policy"]:
                _fr = obs_dict["policy"]["front"].squeeze().float().reshape(3, env_cam_h, env_cam_w)
                _seg = obs_dict["policy"].get(seg_keys["front"], None)
                if _seg is not None:
                    _seg = _seg.squeeze()
                _bg = bg_images.get("front") if bg_images else None
                front_frames.append(process_image(_fr, _seg, _bg))

            if terminated or truncated:
                break

            # 早停: 全部 stage 都成功就立刻退出, 下一个 trial 早点开始
            if step % early_stop_check_every == 0 and _check_early_success():
                print(f"[EVAL] early stop at step={step}: all stages passed")
                early_success = True
                break

            obs = obs_dict["policy"]

        if terminated or truncated or early_success:
            break

    # --- Linger: 整个 play 结束后多跑 N 帧观察是否有突变 (物体掉落 / place 后翻倒等). ---
    # freeze 动作: arm 不动, gripper 维持最后一次命令 (不要松开正在抓的物体).
    #   ee_delta:    arm = 6 维全 0 delta
    #   joint_delta: arm = 当前 qpos (env 走 JointPosition 绝对)
    if linger_frames > 0:
        D = env.action_space.shape[1]
        arm_dim = D - 1
        print(f"[EVAL] play done at step={step}, lingering {linger_frames} frames "
              f"(action_type={action_type}, gripper={last_gripper_cmd:+.2f}) ...")
        for _ in range(linger_frames):
            if action_type == "joint_delta":
                cur_qpos = obs_dict["policy"]["qpos"].squeeze().detach().cpu().numpy()
                arm_part = cur_qpos[:arm_dim].astype(np.float32)
            else:
                arm_part = np.zeros(arm_dim, dtype=np.float32)
            freeze = np.concatenate([arm_part, np.array([last_gripper_cmd], dtype=np.float32)])
            action_tensor = torch.from_numpy(freeze).float().to(device).view(1, -1)
            obs_dict, _, terminated, truncated, _ = env.step(action_tensor)
            _update_peak()
            if collect_front_images and "front" in obs_dict["policy"]:
                _fr = obs_dict["policy"]["front"].squeeze().float().reshape(3, env_cam_h, env_cam_w)
                _seg = obs_dict["policy"].get(seg_keys["front"], None)
                if _seg is not None:
                    _seg = _seg.squeeze()
                _bg = bg_images.get("front") if bg_images else None
                front_frames.append(process_image(_fr, _seg, _bg))
            if terminated or truncated:
                break

    return obs_dict, pick_peak_z, pull_peak_displacement, pour_min_tilt_z, front_frames


# ============================================================================
# Main
# ============================================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Default output path ---
    if args_cli.output_json is None:
        args_cli.output_json = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "..", "debug_images", "pi05_eval_results.json"
        )
    args_cli.output_json = os.path.abspath(args_cli.output_json)

    # --- Connect to Pi0.5 policy server ---
    print(f"[EVAL] Connecting to Pi0.5 server at {args_cli.server_host}:{args_cli.server_port}...")
    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args_cli.server_host,
        port=args_cli.server_port,
    )
    print(f"[EVAL] Connected! Server metadata: {policy.get_server_metadata()}")

    # --- Setup environment ---
    import importlib

    from isaaclab_tasks.real2sim.tasks.pick_r2s_env_cfg import create_pick_env_cfg_from_yaml

    # Resolve env cfg class from the gym registry's `env_cfg_entry_point` so the
    # action space matches the task id (e.g. *_joint-v0 uses JointPosition while
    # the default variants use IK-rel pose deltas).
    _entry = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]
    _mod_name, _cls_name = _entry.split(":")
    env_cfg_class = getattr(importlib.import_module(_mod_name), _cls_name)
    print(f"[EVAL] task={args_cli.task} -> env_cfg_class={_cls_name}")

    env_cfg = create_pick_env_cfg_from_yaml(
        yaml_path=args_cli.task_config,
        env_cfg_class=env_cfg_class,
        num_envs=1,
        device=args_cli.device,
    )
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.terminations.time_out = None
    if hasattr(env_cfg.terminations, "object_dropping"):
        env_cfg.terminations.object_dropping = None
    env_cfg.recorders = None
    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    with open(args_cli.task_config) as f:
        task_yaml = yaml.safe_load(f) or {}
    attach_scene_success_meta(env.scene, task_yaml)

    # Speckle noise applied to policy inputs at eval time, to match the
    # train-time data distribution produced by EpisodeBuffer._add_speckle_noise.
    image_speckle_noise_std = float(task_yaml.get("image_speckle_noise_std", 0.0) or 0.0)
    print(f"[EVAL] image_speckle_noise_std = {image_speckle_noise_std} "
          f"(applied only to front/wrist sent to the policy)")

    # --- Eval stages ---
    stages = load_eval_stages(args_cli.task_config)
    pick_targets = resolve_unique_pick_targets(stages)  # stack 任务有多个 pick
    primary_pick_target = pick_targets[0] if pick_targets else None
    pull_target = next((s.get("pull_target") for s in stages if s["type"] == "pull"), None)
    pour_target = next((s.get("pour_target") for s in stages if s["type"] == "pour"), None)
    print(f"[EVAL] stages={[s['type'] for s in stages]}, pick_targets={pick_targets}, "
          f"pull_target={pull_target}, pour_target={pour_target}")

    # Set seeds
    if args_cli.seed is not None:
        torch.manual_seed(args_cli.seed)
        np.random.seed(args_cli.seed)
        random.seed(args_cli.seed)

    # Load background images
    bg_images = load_background_images(args_cli.task_config)

    # --- Run rollouts ---
    per_rollout_results = []
    gif_dir = os.path.join(os.path.dirname(args_cli.output_json), f"pi05_gifs_{args_cli.task}")
    os.makedirs(gif_dir, exist_ok=True)
    gif_paths = []

    for trial in range(args_cli.num_rollouts):
        print(f"\n[EVAL] === Trial {trial + 1}/{args_cli.num_rollouts} ===")

        initial_obs_dict, _ = env.reset()
        for _ in range(10):
            warmup = _build_warmup_action(args_cli.warmup_action, env, initial_obs_dict, device)
            initial_obs_dict, _, _, _, _ = env.step(warmup)
        # Per-target initial_z (stack 任务里每个 block 都要单独记).
        pick_initial_z = {
            tgt: capture_object_initial_z(env.scene, tgt, env_idx=0)
            for tgt in pick_targets
        }
        pick_initial_z = {k: v for k, v in pick_initial_z.items() if v is not None}

        _, pick_peak_z, pull_peak_displacement, pour_min_tilt_z, front_frames = rollout(
            policy=policy,
            env=env,
            horizon=args_cli.horizon,
            action_horizon=args_cli.action_horizon,
            device=device,
            prompt=args_cli.prompt,
            bg_images=bg_images,
            trial_id=trial,
            initial_obs_dict=initial_obs_dict,
            pick_targets=pick_targets,
            primary_pick_target=primary_pick_target,
            pull_target=pull_target,
            pour_target=pour_target,
            collect_front_images=True,
            image_speckle_noise_std=image_speckle_noise_std,
            stages=stages,
            pick_initial_z=pick_initial_z,
            pick_lift_threshold=args_cli.pick_lift_threshold,
            warmup_action=args_cli.warmup_action,
            action_type=args_cli.action_type,
            joint_delta_mode=args_cli.joint_delta_mode,
            linger_frames=args_cli.linger_frames,
        )

        # Save GIF
        gif_path = os.path.join(gif_dir, f"trial_{trial}.gif")
        saved = frames_to_gif(front_frames, gif_path, target_frames=30, target_size=256)
        if saved:
            gif_paths.append(saved)
            print(f"[EVAL] saved GIF: {saved} ({len(front_frames)} frames -> 30 sampled)")

        # Evaluate stages
        stage_results = evaluate_stages_at_episode_end(
            scene=env.scene,
            stages=stages,
            pick_initial_z=pick_initial_z,
            pick_lift_threshold=args_cli.pick_lift_threshold,
            pick_peak_z=pick_peak_z,
            pull_peak_displacement=pull_peak_displacement,
            pour_min_tilt_z=pour_min_tilt_z,
        )
        per_rollout_results.append(stage_results)
        print(f"[EVAL] trial {trial}: " + ", ".join(
            f"{k}={'OK' if v.get('success') else 'FAIL'}" for k, v in stage_results.items()
        ))
        for k, v in stage_results.items():
            metrics = v.get("metrics", v)
            print(f"  {k} details: {metrics}")

    # --- Aggregate results ---
    stage_keys: list[str] = []
    for r in per_rollout_results:
        for k in r.keys():
            if k not in stage_keys:
                stage_keys.append(k)

    stage_success_rate: dict[str, float] = {}
    for key in stage_keys:
        succ = sum(1 for r in per_rollout_results if r.get(key, {}).get("success"))
        stage_success_rate[key] = succ / max(len(per_rollout_results), 1)

    overall_success = 0
    for r in per_rollout_results:
        if stage_keys and all(r.get(k, {}).get("success") for k in stage_keys):
            overall_success += 1
    overall_rate = overall_success / max(len(per_rollout_results), 1)

    # --- Print results ---
    print(f"\n{'='*60}")
    print(f"[EVAL] ===== EVALUATION RESULTS =====")
    print(f"  server: {args_cli.server_host}:{args_cli.server_port}")
    print(f"  prompt: {args_cli.prompt}")
    print(f"  num_rollouts: {args_cli.num_rollouts}")
    print(f"  horizon: {args_cli.horizon}")
    print(f"")
    print(f"  Stage success rates:")
    for key, rate in stage_success_rate.items():
        succ = sum(1 for r in per_rollout_results if r.get(key, {}).get("success"))
        print(f"    {key}: {succ}/{args_cli.num_rollouts} = {rate:.2%}")
    print(f"")
    print(f"  Overall success rate: {overall_success}/{args_cli.num_rollouts} = {overall_rate:.2%}")
    print(f"{'='*60}")

    # --- Save JSON ---
    summary = {
        "server": f"{args_cli.server_host}:{args_cli.server_port}",
        "prompt": args_cli.prompt,
        "num_rollouts": args_cli.num_rollouts,
        "horizon": args_cli.horizon,
        "stage_success_rate": stage_success_rate,
        "overall_success_rate": overall_rate,
        "per_rollout": per_rollout_results,
        "gif_paths": gif_paths,
    }

    Path(os.path.dirname(args_cli.output_json)).mkdir(parents=True, exist_ok=True)
    with open(args_cli.output_json, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[EVAL] Results saved to {args_cli.output_json}")

    env.close()


if __name__ == "__main__":
    import traceback
    exit_code = 0
    try:
        main()
    except SystemExit as e:
        exit_code = int(e.code) if isinstance(e.code, int) else 1
        print(f"[EVAL] SystemExit(code={e.code})")
    except BaseException as e:
        exit_code = 1
        print(f"[EVAL] ===== UNCAUGHT EXCEPTION =====")
        traceback.print_exc()
        print(f"[EVAL] ===== END EXCEPTION =====")
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass
    sys.exit(exit_code)
