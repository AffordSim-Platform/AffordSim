"""
ACT 策略推理脚本 — 在 Isaac Lab 环境中 play 训练好的 ACT 模型.

推理流程:
1. 编码当前观测 (front, wrist, qpos+qvel)
2. z = 0 (推理时使用先验均值)
3. Transformer Decoder 生成 action chunk (C 步)
4. 执行前 action_horizon 步, 然后重新规划

用法:
    python scripts/imitation_learning/ACT/play_act.py \
        --task Pick_Place_banana-v0 \
        --checkpoint logs/act_xxx/checkpoints/best_eval.pt \
        --task_config /path/to/generated_task_config.yaml \
        --horizon 300 --num_rollouts 10
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play ACT policy in Isaac Lab environment.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to ACT checkpoint (.pt)")
parser.add_argument("--horizon", type=int, default=500, help="Max steps per episode.")
parser.add_argument("--num_rollouts", type=int, default=10, help="Number of rollout trials.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--task_config", type=str, default=None, help="Path to YAML task config file.")
parser.add_argument("--stat", type=str, default=None,
                    help="Path to stat JSON with _minmax_params for denormalization.")
parser.add_argument("--action_horizon", type=int, default=None,
                    help="chunk 模式下实际执行的步数 (默认=chunk_size, 即全部执行).")
parser.add_argument("--action_chunk_size", type=int, default=None,
                    help="action chunk size; 若不指定则从 checkpoint 推断")
parser.add_argument("--save_img_interval", type=int, default=0,
                    help="每隔多少步保存一次调试图像 (0=不保存)")
parser.add_argument("--pick_lift_threshold", type=float, default=0.02,
                    help="Pick 成功所需的最小抬升高度 (米)")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import copy
import json
import os
import random
import sys

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from isaaclab_tasks.real2sim.utils.EpisodeBuffer import replace_background_with_segmentation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from network_act import ACTPolicy
from train_act import minmax_normalize, minmax_denormalize

import isaaclab_tasks.real2sim.tasks  # noqa: F401

from isaaclab_tasks.utils import parse_env_cfg

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
from isaaclab.motion_collect.success.success import (
    capture_object_initial_z,
)
from isaaclab.motion_collect.tools.pose_utils import _quat_wxyz_to_matrix
from isaaclab.motion_collect.tools.affordance_resolver import resolve_affordance_path
from isaaclab.motion_collect.primitives.utils import load_affordance_centroid



# ============================================================================
# Helpers
# ============================================================================

def load_checkpoint(ckpt_path, device, stat=None, action_chunk_size=None):
    """加载 ACT checkpoint, 返回 policy 和 norm_params."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    norm_params = ckpt.get("norm_params", None)
    if norm_params is None:
        norm_json = os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), "norm_params.json")
        if os.path.exists(norm_json):
            with open(norm_json) as f:
                norm_params = json.load(f)
            print(f"[INFO] Loaded norm_params from {norm_json}")

    model_cfg = ckpt.get("model_config", {})
    state_dim = model_cfg.get("state_dim", 14)
    action_dim = model_cfg.get("action_dim", 7)
    hidden_dim = model_cfg.get("hidden_dim", 512)
    latent_dim = model_cfg.get("latent_dim", 32)
    if action_chunk_size is None:
        action_chunk_size = model_cfg.get("action_chunk_size", 16)

    policy = ACTPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        action_chunk_size=action_chunk_size,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
    ).to(device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    # 用 stat 覆盖 norm_params
    if stat:
        with open(stat) as f:
            ds_stat = json.load(f)
        real_minmax = ds_stat.get("_minmax_params", {})
        if real_minmax:
            if "action" in real_minmax:
                if norm_params is None:
                    norm_params = {}
                norm_params["action_ee"] = {
                    "min": real_minmax["action"]["min"][:6],
                    "max": real_minmax["action"]["max"][:6],
                }
            if "qpos" in real_minmax:
                norm_params["qpos"] = real_minmax["qpos"]
            if "qvel" in real_minmax:
                norm_params["qvel"] = real_minmax["qvel"]
            print(f"[INFO] Overriding norm_params with _minmax_params from {stat}")

    print(f"[INFO] Loaded ACT policy: state_dim={state_dim}, action_dim={action_dim}, "
          f"action_chunk_size={action_chunk_size}, hidden_dim={hidden_dim}, latent_dim={latent_dim}")
    return policy, norm_params, action_chunk_size


def load_background_images(task_config_path):
    """从 YAML 配置中加载背景图片."""
    with open(task_config_path) as f:
        cfg = yaml.safe_load(f)
    runtime = cfg.get("runtime", {})
    config_dir = os.path.dirname(os.path.abspath(task_config_path))

    bg_images = {}
    for key, bg_key in [("front", "cam_side_background"), ("wrist", "cam_wrist_background")]:
        bg_path = runtime.get(bg_key)
        if bg_path:
            if os.path.isabs(bg_path):
                full_path = bg_path
            else:
                cwd_path = os.path.abspath(bg_path)
                if os.path.exists(cwd_path):
                    full_path = cwd_path
                else:
                    full_path = os.path.normpath(os.path.join(config_dir, bg_path))
            img = cv2.imread(full_path)
            if img is not None:
                bg_images[key] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                print(f"[INFO] 加载背景图片 {key}: {full_path} -> {img.shape}")
    return bg_images


def preprocess_image(image_tensor, seg_tensor=None, bg_image=None,
                     env_cam_h=480, env_cam_w=640, train_img_size=256):
    """处理环境 RGB 观测 -> 模型输入.

    输入: (C, H, W) float tensor [0, 1]
    输出: (train_img_size, train_img_size, 3) float tensor [0, 1]
    """
    image = image_tensor.clone().float()
    image = image.reshape(3, env_cam_h, env_cam_w)

    if bg_image is not None and seg_tensor is not None:
        seg = seg_tensor.cpu().numpy().reshape(4, env_cam_h, env_cam_w)
        seg_rgba = seg.transpose(1, 2, 0)
        rgb_np = image.cpu().numpy().transpose(1, 2, 0)
        composite, _ = replace_background_with_segmentation(rgb_np, seg_rgba, bg_image)
        image = torch.from_numpy(composite.transpose(2, 0, 1)).float()

    _, h, w = image.shape
    crop_size = min(h, w)
    y0 = (h - crop_size) // 2
    x0 = (w - crop_size) // 2
    image = image[:, y0:y0 + crop_size, x0:x0 + crop_size]

    if crop_size != train_img_size:
        image = image.unsqueeze(0)
        image = F.interpolate(image, size=(train_img_size, train_img_size),
                              mode="bilinear", align_corners=False)
        image = image.squeeze(0)

    image = image.permute(1, 2, 0)  # (H, W, C)
    return image


# ============================================================================
# Task yaml -> 评估阶段
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


def evaluate_stages_at_episode_end(
    scene, stages, pick_initial_z, pick_lift_threshold,
    pick_peak_z=None, pour_min_tilt_z: float | None = None,
    pour_crossed_ok=False, pour_xy_at_cross=None,
    pull_peak_displacement=None,
):
    pick_target = resolve_pick_target(stages)
    results = {}

    for idx, stage in enumerate(stages):
        ptype = stage["type"]
        key = f"{idx:02d}_{ptype}"

        if ptype == "pick":
            if pick_initial_z is None or not pick_target:
                results[key] = {"success": False, "reason": "no_initial_z_or_target"}
                continue
            ok, metrics = _evaluate_pick_success_with_metrics(
                scene=scene, pick_target=pick_target,
                initial_z_world=pick_initial_z, env_idx=0,
                lift_threshold=pick_lift_threshold,
                peak_z_world=pick_peak_z,
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
                    scene=scene, pick_target=pick_target,
                    target_position_env=np.array(target_pos, dtype=float),
                    target_quat_wxyz=np.array(target_quat, dtype=float) if target_quat is not None else None,
                    env_idx=0,
                )
            elif place_mode == "affordance":
                if not place_target or not pick_target:
                    results[key] = {"success": False, "reason": "no_place_or_pick_target"}
                    continue
                pick_aff_path = stage.get("pick_affordance_npz_path") or resolve_affordance_path(pick_target, "place")
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
                    scene=scene, pick_target=pick_target, place_target=place_target,
                    pick_aff_local=pick_aff_local, target_aff_local=target_aff_local, env_idx=0,
                )
            else:
                if not place_target or not pick_target:
                    results[key] = {"success": False, "reason": "no_place_or_pick_target"}
                    continue
                ok, metrics = _evaluate_place_on_success_with_metrics(
                    scene=scene, pick_target=pick_target, place_target=place_target,
                    place_target_height=stage.get("place_target_height") or 0.1,
                    on_height_margin=stage.get("place_on_height_margin") or 0.01,
                    env_idx=0,
                )
            results[key] = {"success": bool(ok), "metrics": metrics}

        elif ptype == "pour":
            pour_target = stage.get("pour_target")
            if not pour_target or not pick_target:
                results[key] = {"success": False, "reason": "no_pour_or_pick_target"}
                continue
            ok, metrics = _evaluate_pour_success_with_metrics(
                scene=scene, pick_target=pick_target,
                pour_target=pour_target, env_idx=0,
                min_z_component_observed=pour_min_tilt_z,
            )
            results[key] = {"success": bool(ok), "metrics": metrics}

        elif ptype == "hang":
            hang_container = stage.get("hang_container")
            if not hang_container or not pick_target:
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
                held_object_label=pick_target,
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

def frames_to_gif(frames, save_path, target_frames=30, target_size=256):
    if not frames:
        return None
    from PIL import Image

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
# Action Post-processing
# ============================================================================

def postprocess_action(action_np, norm_params):
    """
    将模型输出的 action 转换为环境 action.

    ee_delta:    action_np: (7,) — [ee_delta(6) * 100, gripper_continuous]
    joint_delta: action_np: (8,) — [joint_delta(7) * 100, gripper_continuous]
    """
    result = action_np.copy()
    action_type = norm_params.get("action_type", "ee_delta") if norm_params else "ee_delta"

    if action_type == "joint_delta":
        # Joint delta: 训练时放大了 100 倍, 缩回来
        result[:7] = result[:7] / 100.0
        # 反归一化
        if norm_params is not None and "action_ee" in norm_params:
            result[:7] = minmax_denormalize(
                result[:7],
                np.array(norm_params["action_ee"]["min"], dtype=np.float32),
                np.array(norm_params["action_ee"]["max"], dtype=np.float32),
            )
        # Gripper: continuous [0, 1] -> env action (-1=open, +1=close)
        gripper_cont = result[7]
        result[7] = -1.0 if gripper_cont > 0.5 else 1.0
    else:
        # EE delta: 训练时放大了 100 倍, 缩回来
        result[:6] = result[:6] / 100.0
        # 反归一化
        if norm_params is not None and "action_ee" in norm_params:
            result[:6] = minmax_denormalize(
                result[:6],
                np.array(norm_params["action_ee"]["min"], dtype=np.float32),
                np.array(norm_params["action_ee"]["max"], dtype=np.float32),
            )
        # Gripper: continuous [0, 1] -> env action (-1=open, +1=close)
        gripper_cont = result[6]
        result[6] = -1.0 if gripper_cont > 0.5 else 1.0

    return result


# ============================================================================
# Rollout
# ============================================================================

def rollout(policy, env, success_term, horizon, device, norm_params,
            bg_images=None, trial_id=0,
            save_img_interval=0, pick_target=None, pour_target=None,
            pull_target=None,
            action_chunk_size=1, action_horizon=None):
    """执行一次 rollout."""

    _tc = getattr(args_cli, "task_config", None)
    task_label = os.path.splitext(os.path.basename(_tc))[0] if _tc else "unknown"
    debug_img_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "debug_images", task_label, f"trial_{trial_id}",
    )
    if save_img_interval > 0:
        os.makedirs(debug_img_dir, exist_ok=True)

    obs_dict, _ = env.reset()

    pick_peak_z = None
    pour_min_tilt_z = None
    pour_crossed_ok = False
    pour_xy_at_cross = None
    pour_tilt_at_cross = None
    pull_peak_displacement = None
    front_frames = []
    action_chunk_buffer = []

    def _update_peak():
        nonlocal pick_peak_z, pour_min_tilt_z, pour_crossed_ok, pour_xy_at_cross, pour_tilt_at_cross
        nonlocal pull_peak_displacement
        if pull_target:
            from isaaclab.motion_collect.success.constants import PULL_SUCCESS_JOINT_NAME
            jp = _get_articulation_joint_pos(env.scene, pull_target, PULL_SUCCESS_JOINT_NAME, 0)
            if jp is not None:
                disp = abs(float(jp))
                if pull_peak_displacement is None or disp > pull_peak_displacement:
                    pull_peak_displacement = disp
        if not pick_target:
            return
        pos_w, quat_w = _get_object_pose_world_from_scene(env.scene, pick_target, 0)
        if pos_w is None:
            return
        z = float(pos_w[2])
        if pick_peak_z is None or z > pick_peak_z:
            pick_peak_z = z
        if pour_target and quat_w is not None:
            pour_pos_w, _ = _get_object_pose_world_from_scene(env.scene, pour_target, 0)
            if pour_pos_w is not None:
                xy_dist = float(np.linalg.norm(pos_w[:2] - pour_pos_w[:2]))
                if xy_dist <= 0.10:
                    rot = _quat_wxyz_to_matrix(np.array(quat_w, dtype=float))
                    tilt_z = float(rot[2, 2])
                    if pour_min_tilt_z is None or tilt_z < pour_min_tilt_z:
                        pour_min_tilt_z = tilt_z

    seg_keys = {"front": "semantic_segmentation_side", "wrist": "semantic_segmentation_wrist"}

    for step in range(horizon):
        if step == 0:
            obs_dict, _, _, _, _ = env.step(torch.zeros(1, env.action_space.shape[1], device=device))
            if "front" in obs_dict["policy"]:
                _fr = obs_dict["policy"]["front"].squeeze().float().reshape(3, 480, 640)
                _seg = obs_dict["policy"].get(seg_keys["front"], None)
                if _seg is not None:
                    _seg = _seg.squeeze()
                _bg = bg_images.get("front") if bg_images else None
                _processed = preprocess_image(_fr, _seg, _bg)
                front_frames.append((_processed.cpu().numpy() * 255).clip(0, 255).astype(np.uint8))
            continue

        obs = copy.deepcopy(obs_dict["policy"])
        for k in obs:
            obs[k] = torch.squeeze(obs[k])

        # Prepare state
        qpos = obs["qpos"].cpu().numpy()
        qvel = obs["qvel"].cpu().numpy()
        if norm_params is not None:
            qpos = minmax_normalize(qpos,
                                    np.array(norm_params["qpos"]["min"], dtype=np.float32),
                                    np.array(norm_params["qpos"]["max"], dtype=np.float32))
            qvel = minmax_normalize(qvel,
                                    np.array(norm_params["qvel"]["min"], dtype=np.float32),
                                    np.array(norm_params["qvel"]["max"], dtype=np.float32))
        state = np.concatenate([qpos, qvel], axis=-1).astype(np.float32)
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)

        need_inference = len(action_chunk_buffer) == 0

        if need_inference:
            front_img = preprocess_image(
                obs["front"],
                seg_tensor=obs.get(seg_keys["front"]),
                bg_image=bg_images.get("front") if bg_images else None,
            ).unsqueeze(0).to(device)

            wrist_img = preprocess_image(
                obs["wrist"],
                seg_tensor=obs.get(seg_keys["wrist"]),
                bg_image=bg_images.get("wrist") if bg_images else None,
            ).unsqueeze(0).to(device)

            if save_img_interval > 0 and step % save_img_interval == 0:
                f_np = (front_img.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                w_np = (wrist_img.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(debug_img_dir, f"step_{step:04d}_front.png"),
                            cv2.cvtColor(f_np, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(debug_img_dir, f"step_{step:04d}_wrist.png"),
                            cv2.cvtColor(w_np, cv2.COLOR_RGB2BGR))

            # Inference: get_action 返回 (1, C, 7)
            action = policy.get_action(front_img, wrist_img, state_tensor)
            action_raw = action.cpu().numpy().squeeze(0)  # (C, 7)

            if action_raw.ndim == 2:
                ah = action_horizon or action_raw.shape[0]
                for a in action_raw[:ah]:
                    action_chunk_buffer.append(a.copy())
            else:
                action_chunk_buffer.append(action_raw.copy())

        action_np = action_chunk_buffer.pop(0)

        # Post-process
        action_env = postprocess_action(action_np, norm_params)

        # Joint delta: 读当前 qpos + delta -> 绝对目标，通过绝对位置控制执行
        action_type = norm_params.get("action_type", "ee_delta") if norm_params else "ee_delta"
        if action_type == "joint_delta":
            cur_qpos = env.env.scene["robot"].data.joint_pos[0, :7].cpu().numpy()
            abs_target = cur_qpos + action_env[:7]
            action_env = np.concatenate([abs_target, action_env[7:]])

        if step <= 5 or step % 50 == 0:
            print(f"[DEBUG] step {step} raw_action: {np.round(action_np, 4)}")
            print(f"[DEBUG] step {step} env_action: {np.round(action_env, 4)}")

        action_tensor = torch.from_numpy(action_env).to(device).view(1, env.action_space.shape[1])
        obs_dict, _, terminated, truncated, _ = env.step(action_tensor)
        _update_peak()

        # 收集 front 帧
        if "front" in obs_dict["policy"]:
            _fr = obs_dict["policy"]["front"].squeeze().float().reshape(3, 480, 640)
            _seg = obs_dict["policy"].get(seg_keys["front"], None)
            if _seg is not None:
                _seg = _seg.squeeze()
            _bg = bg_images.get("front") if bg_images else None
            _processed = preprocess_image(_fr, _seg, _bg)
            front_frames.append((_processed.cpu().numpy() * 255).clip(0, 255).astype(np.uint8))

        if success_term is not None and bool(success_term.func(env, **success_term.params)[0]):
            print(f"[INFO] 成功! step={step}")
            return True, pick_peak_z, pour_min_tilt_z, pour_crossed_ok, pour_xy_at_cross, pull_peak_displacement, front_frames
        elif terminated or truncated:
            return False, pick_peak_z, pour_min_tilt_z, pour_crossed_ok, pour_xy_at_cross, pull_peak_displacement, front_frames

    return False, pick_peak_z, pour_min_tilt_z, pour_crossed_ok, pour_xy_at_cross, pull_peak_displacement, front_frames


# ============================================================================
# Main
# ============================================================================

def main():
    device = torch.device(args_cli.device if args_cli.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)

    # Load policy
    policy, norm_params, action_chunk_size = load_checkpoint(
        args_cli.checkpoint, device,
        stat=args_cli.stat,
        action_chunk_size=args_cli.action_chunk_size,
    )

    # Create environment
    if args_cli.task_config is not None:
        from isaaclab_tasks.real2sim.tasks.pick_r2s_env_cfg import create_pick_env_cfg_from_yaml
        from isaaclab_tasks.real2sim.tasks.config.franka_fr3.ourgripper.ik_rel_vis_env_cfg import (
            FrankaFr3OurGripperIKRelVisPickEnvCfg,
        )
        env_cfg = create_pick_env_cfg_from_yaml(
            yaml_path=args_cli.task_config,
            env_cfg_class=FrankaFr3OurGripperIKRelVisPickEnvCfg,
            num_envs=1,
            device=args_cli.device,
        )
    else:
        env_cfg = parse_env_cfg(
            args_cli.task, device=args_cli.device, num_envs=1,
            use_fabric=not args_cli.disable_fabric,
        )

    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.terminations.time_out = None
    if hasattr(env_cfg.terminations, "object_dropping"):
        env_cfg.terminations.object_dropping = None
    env_cfg.recorders = None

    success_term = getattr(env_cfg.terminations, "success", None)
    if success_term is not None:
        env_cfg.terminations.success = None

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    if args_cli.task_config is not None:
        with open(args_cli.task_config) as f:
            task_yaml = yaml.safe_load(f) or {}
        attach_scene_success_meta(env.scene, task_yaml)

    # 评估阶段
    stages = []
    pick_target = None
    pour_target = None
    pull_target = None
    if args_cli.task_config is not None:
        stages = load_eval_stages(args_cli.task_config)
        pick_target = resolve_pick_target(stages)
        pour_target = next((s.get("pour_target") for s in stages if s["type"] == "pour"), None)
        pull_target = next((s.get("pull_target") for s in stages if s["type"] == "pull"), None)
    print(f"[INFO] stages={[s['type'] for s in stages]}, pick_target={pick_target}, pour_target={pour_target}, pull_target={pull_target}")

    bg_images = {}
    if args_cli.task_config is not None:
        bg_images = load_background_images(args_cli.task_config)

    action_horizon = args_cli.action_horizon

    # --- Rollout ---
    results = []
    all_stage_results = []
    task_label = os.path.splitext(os.path.basename(args_cli.task_config))[0] if args_cli.task_config else args_cli.task
    gif_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "debug_images", f"act_play_gifs_{task_label}",
    )
    os.makedirs(gif_dir, exist_ok=True)

    for trial in range(args_cli.num_rollouts):
        print(f"\n[INFO] Trial {trial}")
        initial_obs_dict, _ = env.reset()
        for _ in range(10):
            env.step(torch.zeros(1, env.action_space.shape[1], device=device))
        pick_initial_z = capture_object_initial_z(env.scene, pick_target, env_idx=0) if pick_target else None

        success, pick_peak_z, pour_min_tilt_z, pour_crossed_ok, pour_xy_at_cross, pull_peak_displacement, front_frames = rollout(
            policy, env, success_term, args_cli.horizon, device,
            norm_params=norm_params, bg_images=bg_images,
            trial_id=trial,
            save_img_interval=args_cli.save_img_interval,
            pick_target=pick_target,
            pour_target=pour_target,
            pull_target=pull_target,
            action_chunk_size=action_chunk_size,
            action_horizon=action_horizon,
        )
        results.append(success)

        gif_path = os.path.join(gif_dir, f"trial_{trial}.gif")
        saved = frames_to_gif(front_frames, gif_path, target_frames=30, target_size=256)
        if saved:
            print(f"[INFO] saved GIF: {saved} ({len(front_frames)} frames -> 30 sampled)")

        if stages:
            stage_results = evaluate_stages_at_episode_end(
                scene=env.scene, stages=stages,
                pick_initial_z=pick_initial_z,
                pick_lift_threshold=args_cli.pick_lift_threshold,
                pick_peak_z=pick_peak_z,
                pour_min_tilt_z=pour_min_tilt_z,
                pour_crossed_ok=pour_crossed_ok,
                pour_xy_at_cross=pour_xy_at_cross,
                pull_peak_displacement=pull_peak_displacement,
            )
            all_stage_results.append(stage_results)
            stage_summary = ", ".join(
                f"{k}={'OK' if v.get('success') else 'FAIL'}" for k, v in stage_results.items()
            )
            print(f"[INFO] Trial {trial}: term={'Success' if success else 'Fail'} | {stage_summary}")
            for k, v in stage_results.items():
                metrics = {mk: mv for mk, mv in v.items() if mk != "success" and mk != "reason"}
                if metrics:
                    metrics_str = ", ".join(f"{mk}={mv:.4f}" if isinstance(mv, float) else f"{mk}={mv}" for mk, mv in metrics.items())
                    print(f"  [{k}] success={v.get('success')}, {metrics_str}")
                elif "reason" in v:
                    print(f"  [{k}] success={v.get('success')}, reason={v.get('reason')}")
        else:
            print(f"[INFO] Trial {trial}: {'SUCCESS' if success else 'FAIL'}")

    # 汇总
    print(f"\n[INFO] === 结果汇总 ===")
    print(f"Success term: {results.count(True)}/{len(results)} = {results.count(True) / len(results):.2%}")

    if all_stage_results:
        stage_keys = []
        for r in all_stage_results:
            for k in r.keys():
                if k not in stage_keys:
                    stage_keys.append(k)
        for key in stage_keys:
            succ = sum(1 for r in all_stage_results if r.get(key, {}).get("success"))
            print(f"  {key}: {succ}/{len(all_stage_results)} = {succ / len(all_stage_results):.2%}")

        overall = sum(
            1 for r in all_stage_results
            if stage_keys and all(r.get(k, {}).get("success") for k in stage_keys)
        )
        print(f"  overall: {overall}/{len(all_stage_results)} = {overall / len(all_stage_results):.2%}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
