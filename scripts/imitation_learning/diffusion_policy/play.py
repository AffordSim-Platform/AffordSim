"""
Diffusion Policy 推理脚本.

在 Isaac Lab 环境中运行训练好的 Diffusion Policy.

DDPM 采样流程:
1. 编码当前观测 (obs_horizon 帧)
2. 从纯噪声 x_T ~ N(0, I) 开始
3. 逐步去噪 x_T -> x_{T-1} -> ... -> x_0
4. x_0 即为预测的动作序列 (pred_horizon 步)
5. 只执行前 action_horizon 步, 然后重新规划

用法:
    python scripts/imitation_learning/diffusion_policy/play.py \
        --checkpoint /path/to/model_best.pth \
        --stat /path/to/dataset_stat.json \
        --task_config /path/to/generated_task_config.yaml \
        --task Pick_Place_banana-v0 \
        --horizon 300
"""

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path

# Isaac Lab 启动
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Diffusion Policy Inference")
parser.add_argument("--task", type=str, default="Pick_Place_banana-v0", help="环境名称")
parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路径")
parser.add_argument("--stat", type=str, default=None, help="dataset_stat.json 路径")
parser.add_argument("--horizon", type=int, default=500, help="最大步数")
parser.add_argument("--num_rollouts", type=int, default=1, help="Rollout 次数")
parser.add_argument("--seed", type=int, default=None, help="随机种子 (None=随机)")
parser.add_argument("--save_img_interval", type=int, default=1, help="每隔多少步保存一次调试图像 (0=不保存)")
parser.add_argument(
    "--task_config",
    type=str,
    default="source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/task_gen/config/sceneconfig/generated_task_config.yaml",
    help="YAML 任务配置文件路径",
)
parser.add_argument("--pick_lift_threshold", type=float, default=0.05)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- 以下在 Isaac Sim 启动后导入 ---
import cv2
import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn.functional as F
import yaml

from isaaclab_tasks.utils import parse_env_cfg
import isaaclab_tasks.real2sim.tasks  # noqa: F401
from isaaclab_tasks.real2sim.utils.EpisodeBuffer import replace_background_with_segmentation

# 添加 diffusion_policy 目录到 path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from network import DiffusionPolicy
from train import DDPMScheduler

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
# 工具函数
# ============================================================================

def load_norm_params(stat_path):
    """加载 dataset_stat.json 中的 min-max 归一化参数."""
    with open(stat_path) as f:
        stat = json.load(f)
    params = stat["_minmax_params"]
    result = {}
    for key in params:
        result[key] = {
            "min": np.array(params[key]["min"], dtype=np.float32),
            "max": np.array(params[key]["max"], dtype=np.float32),
        }
    return result


def minmax_normalize(data, dmin, dmax):
    drange = dmax - dmin
    drange = np.where(drange < 1e-8, 1.0, drange)
    return 2.0 * (data - dmin) / drange - 1.0


def minmax_denormalize(data, dmin, dmax):
    drange = dmax - dmin
    return (data + 1.0) / 2.0 * drange + dmin


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
                # 优先按 CWD 解析；文件不存在则回退到 yaml 所在目录
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


def process_image(image_tensor, seg_tensor=None, bg_image=None, target_size=256):
    """处理 RGB 图像: 背景替换 + center crop + resize.

    Args:
        image_tensor: (3, H, W) float tensor, 范围 [0, 1]
        seg_tensor: 可选, 语义分割 tensor
        bg_image: 可选, 背景图片 numpy array (H, W, 3)
        target_size: 目标分辨率

    Returns:
        (3, target_size, target_size) float tensor, 范围 [0, 1]
    """
    image = image_tensor.clone().float()
    C, H, W = image.shape

    # 背景替换
    if bg_image is not None and seg_tensor is not None:
        seg = seg_tensor.cpu().numpy().reshape(4, H, W).transpose(1, 2, 0)  # (H, W, 4)
        rgb_np = image.cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
        composite, _ = replace_background_with_segmentation(rgb_np, seg, bg_image)
        image = torch.from_numpy(composite.transpose(2, 0, 1)).float()

    # Center crop to square
    crop_size = min(H, W)
    y0 = (H - crop_size) // 2
    x0 = (W - crop_size) // 2
    image = image[:, y0:y0 + crop_size, x0:x0 + crop_size]

    # Resize
    if crop_size != target_size:
        image = F.interpolate(
            image.unsqueeze(0),
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    return image


# ============================================================================
# task yaml -> 评估阶段
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
    """读取 task yaml 的 primitives 字段, 抽取评估阶段描述."""
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
    scene,
    stages: list[dict],
    pick_initial_z: float | None,
    pick_lift_threshold: float,
    pick_peak_z: float | None = None,
    min_pour_tilt_z: float | None = None,
    pull_peak_displacement: float | None = None,
) -> dict[str, dict]:
    """对每个阶段独立判定成功 (单 env, env_idx=0)."""
    pick_target = resolve_pick_target(stages)
    results: dict[str, dict] = {}

    for idx, stage in enumerate(stages):
        ptype = stage["type"]
        key = f"{idx:02d}_{ptype}"

        if ptype == "pick":
            if pick_initial_z is None or not pick_target:
                results[key] = {"success": False, "reason": "no_initial_z_or_target"}
                continue
            ok, metrics = _evaluate_pick_success_with_metrics(
                scene=scene,
                pick_target=pick_target,
                initial_z_world=pick_initial_z,
                env_idx=0,
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
                    scene=scene,
                    pick_target=pick_target,
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
                scene=scene,
                pick_target=pick_target,
                pour_target=pour_target,
                env_idx=0,
                min_z_component_observed=min_pour_tilt_z,
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

def frames_to_gif(frames: list[np.ndarray], save_path: str,
                  target_frames: int = 30, target_size: int = 256) -> str | None:
    """将采集的帧列表抽帧、缩放后保存为 GIF."""
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
# DDPM 采样
# ============================================================================

@torch.no_grad()
def ddpm_sample(model, scheduler, obs_images, obs_lowdim, device):
    """DDPM 采样: 从噪声生成动作序列.

    Args:
        model: DiffusionPolicy 模型
        scheduler: DDPMScheduler
        obs_images: (1, T_o, 2, 3, H, W) 观测图像
        obs_lowdim: (1, T_o, D_low) 低维观测

    Returns:
        actions: (T_a, D_act) 预测的动作序列 (numpy)
    """
    B = 1
    T_a = model.pred_horizon
    D_act = model.action_dim

    # 编码观测 (只需要做一次)
    global_cond = model.encode_obs(obs_images, obs_lowdim)

    # 从纯噪声开始
    x = torch.randn(B, T_a, D_act, device=device)

    # 逐步去噪 T -> 0
    for t in reversed(range(scheduler.num_steps)):
        timestep = torch.full((B,), t, device=device, dtype=torch.long)
        noise_pred = model.noise_pred_net(x, timestep, global_cond)
        x = scheduler.step(noise_pred, t, x)

    return x.squeeze(0).cpu().numpy()  # (T_a, D_act)


# ============================================================================
# Rollout
# ============================================================================

def rollout(model, scheduler, env, success_term, horizon, action_horizon, device,
            norm_params=None, bg_images=None, obs_horizon=2, trial_id=0, save_img_interval=1,
            pick_target=None, pour_target=None, pull_target=None, initial_obs_dict=None):
    """执行一次 rollout.

    Diffusion Policy 的 receding horizon control:
    - 每次预测 pred_horizon 步动作
    - 只执行前 action_horizon 步
    - 然后重新观测, 重新规划

    Args:
        pick_target: 用于追踪物体 z 峰值 (pick 成功判定).
        initial_obs_dict: 若不为 None 则跳过 reset, 直接使用该 obs 开始 rollout.

    Returns:
        (success_by_term, pick_peak_z, front_frames): success_term 判定结果, 物体 z 峰值, front 帧列表.
    """
    # 创建图像保存目录
    _tc = getattr(args_cli, "task_config", None)
    task_label = os.path.splitext(os.path.basename(_tc))[0] if _tc else "unknown"
    debug_img_dir = f"debug_images/{task_label}/trial_{trial_id}"
    os.makedirs(debug_img_dir, exist_ok=True)

    model.eval()
    if initial_obs_dict is None:
        obs_dict, _ = env.reset()
    else:
        obs_dict = initial_obs_dict

    pick_peak_z = None
    min_pour_tilt_z = None
    pull_peak_displacement = None  # pull 关节 |位移| 峰值
    front_frames = []

    def _update_peak():
        nonlocal pick_peak_z, min_pour_tilt_z
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
        # Track min z_component (max tilt) when pick object is within 10cm XY of pour target
        if pour_target and quat_w is not None:
            pour_pos_w, _ = _get_object_pose_world_from_scene(env.scene, pour_target, 0)
            if pour_pos_w is not None:
                xy_dist = float(np.linalg.norm(pos_w[:2] - pour_pos_w[:2]))
                if xy_dist <= 0.10:
                    rot = _quat_wxyz_to_matrix(np.array(quat_w, dtype=float))
                    tilt_z = float(rot[2, 2])
                    if min_pour_tilt_z is None or tilt_z < min_pour_tilt_z:
                        min_pour_tilt_z = tilt_z

    # 观测缓冲区 (保存最近 obs_horizon 帧)
    obs_img_buffer = []   # list of (2, 3, H, W) tensors
    obs_low_buffer = []   # list of (D_low,) tensors

    seg_keys = {"front": "semantic_segmentation_side", "wrist": "semantic_segmentation_wrist"}
    rgb_keys = ["front", "wrist"]
    env_cam_h, env_cam_w = 480, 640
    train_img_size = 256

    step = 0
    while step < horizon:
        # --- 获取当前观测 ---
        obs = obs_dict["policy"]

        # 跳过第一帧 (渲染器未就绪)
        if step == 0:
            obs_dict, _, _, _, _ = env.step(torch.zeros(1, env.action_space.shape[1], device=device))
            step += 1
            if "front" in obs_dict["policy"]:
                _fr = obs_dict["policy"]["front"].squeeze().float().reshape(3, env_cam_h, env_cam_w)
                _seg = obs_dict["policy"].get(seg_keys["front"], None)
                if _seg is not None:
                    _seg = _seg.squeeze()
                _bg = bg_images.get("front") if bg_images else None
                _processed = process_image(_fr, _seg, _bg, train_img_size)
                front_frames.append((_processed.cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8))
            continue

        # 处理图像
        images_processed = []
        for image_name in rgb_keys:
            if image_name in obs:
                img = obs[image_name].squeeze().float().reshape(3, env_cam_h, env_cam_w)
                seg_key = seg_keys[image_name]
                seg = obs.get(seg_key, None)
                if seg is not None:
                    seg = seg.squeeze()
                bg = bg_images.get(image_name) if bg_images else None
                img = process_image(img, seg, bg, train_img_size)
                images_processed.append(img)

        # (2, 3, H, W)
        obs_img = torch.stack(images_processed, dim=0).to(device)

        # 处理低维观测
        qpos = obs["qpos"].squeeze().cpu().numpy()
        qvel = obs["qvel"].squeeze().cpu().numpy()
        if norm_params is not None:
            qpos = minmax_normalize(qpos, norm_params["qpos"]["min"], norm_params["qpos"]["max"])
            qvel = minmax_normalize(qvel, norm_params["qvel"]["min"], norm_params["qvel"]["max"])
        obs_low = torch.from_numpy(np.concatenate([qpos, qvel])).float().to(device)

        # 更新缓冲区
        obs_img_buffer.append(obs_img)
        obs_low_buffer.append(obs_low)

        # 保持缓冲区长度 = obs_horizon
        if len(obs_img_buffer) > obs_horizon:
            obs_img_buffer.pop(0)
            obs_low_buffer.pop(0)

        # 填充 (如果缓冲区不够长, 重复第一帧)
        while len(obs_img_buffer) < obs_horizon:
            obs_img_buffer.insert(0, obs_img_buffer[0])
            obs_low_buffer.insert(0, obs_low_buffer[0])

        # 组装观测 tensor
        obs_images = torch.stack(obs_img_buffer, dim=0).unsqueeze(0)  # (1, T_o, 2, 3, H, W)
        obs_lowdim = torch.stack(obs_low_buffer, dim=0).unsqueeze(0)  # (1, T_o, D_low)

        # 保存当前观测图像
        if save_img_interval > 0 and step % save_img_interval == 0:
            for cam_idx, cam_name in enumerate(rgb_keys):
                img_save = (images_processed[cam_idx].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(debug_img_dir, f"step_{step:04d}_{cam_name}.png"),
                            cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR))

        # --- DDPM 采样 ---
        action_chunk = ddpm_sample(model, scheduler, obs_images, obs_lowdim, device)
        # action_chunk: (T_a, D_act)

        # --- 执行前 action_horizon 步 ---
        for j in range(min(action_horizon, horizon - step)):
            action = action_chunk[j]  # (D_act,)

            # 反归一化 (前 N-1 维)
            if norm_params is not None and "action" in norm_params:
                action[:-1] = minmax_denormalize(
                    action[:-1],
                    norm_params["action"]["min"][:-1],
                    norm_params["action"]["max"][:-1],
                )

            # Gripper 二值化 (阈值 0)
            action[-1] = 1.0 if action[-1] > 0.0 else -1.0

            # Joint delta: 读当前 qpos + delta -> 绝对目标，通过绝对位置控制执行
            action_type = norm_params.get("action_type", "ee_delta") if norm_params else "ee_delta"
            if action_type == "joint_delta":
                cur_qpos = env.env.scene["robot"].data.joint_pos[0, :7].cpu().numpy()
                abs_target = cur_qpos + action[:-1]
                action = np.concatenate([abs_target, action[-1:]])

            action_tensor = torch.from_numpy(action).float().to(device).view(1, -1)

            if step <= 5 or step % 50 == 0:
                print(f"[DEBUG] step {step} action: {np.round(action, 4)}")

            obs_dict, _, terminated, truncated, _ = env.step(action_tensor)
            step += 1
            _update_peak()

            # 收集 front 帧
            if "front" in obs_dict["policy"]:
                _fr = obs_dict["policy"]["front"].squeeze().float().reshape(3, env_cam_h, env_cam_w)
                _seg = obs_dict["policy"].get(seg_keys["front"], None)
                if _seg is not None:
                    _seg = _seg.squeeze()
                _bg = bg_images.get("front") if bg_images else None
                _processed = process_image(_fr, _seg, _bg, train_img_size)
                front_frames.append((_processed.cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8))

            # 检查成功
            if success_term is not None and bool(success_term.func(env, **success_term.params)[0]):
                print(f"[INFO] 成功! step={step}")
                return True, pick_peak_z, min_pour_tilt_z, pull_peak_displacement, front_frames

            if terminated or truncated:
                return False, pick_peak_z, min_pour_tilt_z, pull_peak_displacement, front_frames

            # 如果不是最后一步, 需要更新观测 (用于下一个 chunk 的缓冲区)
            if j < action_horizon - 1:
                obs = obs_dict["policy"]
                # 处理图像
                images_processed = []
                for image_name in rgb_keys:
                    if image_name in obs:
                        img = obs[image_name].squeeze().float().reshape(3, env_cam_h, env_cam_w)
                        seg_key = seg_keys[image_name]
                        seg = obs.get(seg_key, None)
                        if seg is not None:
                            seg = seg.squeeze()
                        bg = bg_images.get(image_name) if bg_images else None
                        img = process_image(img, seg, bg, train_img_size)
                        images_processed.append(img)
                obs_img = torch.stack(images_processed, dim=0).to(device)

                qpos = obs["qpos"].squeeze().cpu().numpy()
                qvel = obs["qvel"].squeeze().cpu().numpy()
                if norm_params is not None:
                    qpos = minmax_normalize(qpos, norm_params["qpos"]["min"], norm_params["qpos"]["max"])
                    qvel = minmax_normalize(qvel, norm_params["qvel"]["min"], norm_params["qvel"]["max"])
                obs_low = torch.from_numpy(np.concatenate([qpos, qvel])).float().to(device)

                obs_img_buffer.append(obs_img)
                obs_low_buffer.append(obs_low)
                if len(obs_img_buffer) > obs_horizon:
                    obs_img_buffer.pop(0)
                    obs_low_buffer.pop(0)

                # 保存内层循环中的观测图像
                if save_img_interval > 0 and step % save_img_interval == 0:
                    for cam_idx, cam_name in enumerate(rgb_keys):
                        img_save = (images_processed[cam_idx].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                        cv2.imwrite(os.path.join(debug_img_dir, f"step_{step:04d}_{cam_name}.png"),
                                    cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR))

    return False, pick_peak_z, min_pour_tilt_z, pull_peak_displacement, front_frames


# ============================================================================
# Main
# ============================================================================

def main():
    # --- 加载模型 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 加载 checkpoint: {args_cli.checkpoint}")
    ckpt = torch.load(args_cli.checkpoint, map_location=device)

    # 从 checkpoint 中恢复模型参数
    model_args = ckpt.get("args", {})
    model = DiffusionPolicy(
        action_dim=model_args.get("action_dim", 7),
        obs_horizon=model_args.get("obs_horizon", 2),
        pred_horizon=model_args.get("pred_horizon", 16),
        lowdim_dim=model_args.get("lowdim_dim", 14),
        img_feature_dim=model_args.get("img_feature_dim", 256),
        lowdim_feature_dim=model_args.get("lowdim_feature_dim", 64),
        down_dims=tuple(model_args.get("down_dims", [256, 512, 1024])),
        pretrained_resnet=False,  # 推理时不需要重新加载预训练权重
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[INFO] 模型加载完成, epoch={ckpt.get('epoch', 'unknown')}")

    # DDPM 调度器
    scheduler = DDPMScheduler(
        num_diffusion_steps=model_args.get("num_diffusion_steps", 100),
    )

    action_horizon = model_args.get("action_horizon", 3)
    obs_horizon = model_args.get("obs_horizon", 2)

    # --- 环境设置 ---
    from isaaclab_tasks.real2sim.tasks.pick_r2s_env_cfg import create_pick_env_cfg_from_yaml
    from isaaclab_tasks.real2sim.tasks.config.franka_fr3.ourgripper.ik_rel_vis_env_cfg import FrankaFr3OurGripperIKRelVisPickEnvCfg

    env_cfg = create_pick_env_cfg_from_yaml(
        yaml_path=args_cli.task_config,
        env_cfg_class=FrankaFr3OurGripperIKRelVisPickEnvCfg,
        num_envs=1,
        device=args_cli.device,
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

    # 把 task yaml 元信息挂到 scene 上, success 评估需要 position_range
    with open(args_cli.task_config) as f:
        task_yaml = yaml.safe_load(f) or {}
    attach_scene_success_meta(env.scene, task_yaml)

    # 评估阶段
    stages = load_eval_stages(args_cli.task_config)
    pick_target = resolve_pick_target(stages)
    pour_target = next((s.get("pour_target") for s in stages if s["type"] == "pour"), None)
    pull_target = next((s.get("pull_target") for s in stages if s["type"] == "pull"), None)
    print(f"[INFO] stages={[s['type'] for s in stages]}, pick_target={pick_target}, pour_target={pour_target}, pull_target={pull_target}")

    # 设置种子
    if args_cli.seed is not None:
        torch.manual_seed(args_cli.seed)
        np.random.seed(args_cli.seed)
        random.seed(args_cli.seed)

    # 加载归一化参数
    norm_params = None
    if args_cli.stat is not None:
        norm_params = load_norm_params(args_cli.stat)
        print(f"[INFO] 加载归一化参数: {args_cli.stat}")

    # 从 checkpoint args 中获取 action_type，注入 norm_params
    action_type = model_args.get("action_type", "ee_delta")
    if norm_params is not None:
        norm_params["action_type"] = action_type
    print(f"[INFO] action_type: {action_type}")

    # 加载背景图片
    bg_images = load_background_images(args_cli.task_config)

    # --- Rollout ---
    results = []
    all_stage_results = []
    task_label = os.path.splitext(os.path.basename(args_cli.task_config))[0] if args_cli.task_config else args_cli.task
    gif_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "debug_images", f"dp_play_gifs_{task_label}",
    )
    os.makedirs(gif_dir, exist_ok=True)

    for trial in range(args_cli.num_rollouts):
        print(f"\n[INFO] Trial {trial}")
        # reset 并记录 pick baseline
        initial_obs_dict, _ = env.reset()
        for _ in range(10):
            env.step(torch.zeros(1, env.action_space.shape[1], device=device))
        pick_initial_z = capture_object_initial_z(env.scene, pick_target, env_idx=0) if pick_target else None

        success, pick_peak_z, min_pour_tilt_z, pull_peak_displacement, front_frames = rollout(
            model, scheduler, env, success_term,
            horizon=args_cli.horizon,
            action_horizon=action_horizon,
            device=device,
            norm_params=norm_params,
            bg_images=bg_images,
            obs_horizon=obs_horizon,
            trial_id=trial,
            save_img_interval=args_cli.save_img_interval,
            pick_target=pick_target,
            pour_target=pour_target,
            pull_target=pull_target,
            initial_obs_dict=initial_obs_dict,
        )
        results.append(success)

        # 保存 GIF
        gif_path = os.path.join(gif_dir, f"trial_{trial}.gif")
        saved = frames_to_gif(front_frames, gif_path, target_frames=30, target_size=256)
        if saved:
            print(f"[INFO] saved GIF: {saved} ({len(front_frames)} frames -> 30 sampled)")

        # 基于原语的成功判定
        stage_results = evaluate_stages_at_episode_end(
            scene=env.scene,
            stages=stages,
            pick_initial_z=pick_initial_z,
            pick_lift_threshold=args_cli.pick_lift_threshold,
            pick_peak_z=pick_peak_z,
            min_pour_tilt_z=min_pour_tilt_z,
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

    # 汇总
    print(f"\n[INFO] === 结果汇总 ===")
    print(f"Success term: {results.count(True)}/{len(results)} = {results.count(True) / len(results):.2%}")

    # 每个阶段的成功率
    stage_keys: list[str] = []
    for r in all_stage_results:
        for k in r.keys():
            if k not in stage_keys:
                stage_keys.append(k)
    for key in stage_keys:
        succ = sum(1 for r in all_stage_results if r.get(key, {}).get("success"))
        print(f"  {key}: {succ}/{len(all_stage_results)} = {succ / len(all_stage_results):.2%}")

    # 全任务成功率
    overall = sum(
        1 for r in all_stage_results
        if stage_keys and all(r.get(k, {}).get("success") for k in stage_keys)
    )
    print(f"  overall: {overall}/{len(all_stage_results)} = {overall / len(all_stage_results):.2%}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
