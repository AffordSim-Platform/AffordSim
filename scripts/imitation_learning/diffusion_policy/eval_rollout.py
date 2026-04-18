"""
Diffusion Policy 训练期评估子进程脚本.

设计目标:
- 由 train.py 在每次保存 ckpt 后通过 subprocess.run 同步拉起.
- 启动 Isaac Lab 仿真, 加载 ckpt, 跑 N 次 rollout.
- 不依赖 env 自带的 success termination, 而是按 task yaml 中
  ``primitives`` 列表里的 type 字段(pick / place / pour) 作为评估阶段,
  每个 episode 结束时对每个阶段独立判定一次成功.
- 结果以 JSON 写到 ``--output_json`` 路径, 供 train.py 读回写 tensorboard.

用法 (一般由 train.py 自动调用):
    python scripts/imitation_learning/diffusion_policy/eval_rollout.py \
        --checkpoint /path/to/model_epoch_100.pth \
        --stat /path/to/dataset_stat.json \
        --task_config /path/to/generated_task_config.yaml \
        --task Pick_Place_banana-v0 \
        --num_rollouts 5 \
        --horizon 300 \
        --output_json /tmp/eval_epoch_100.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Diffusion Policy Eval Rollout (subprocess)")
parser.add_argument("--task", type=str, default="Pick_Place_banana-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--stat", type=str, default=None)
parser.add_argument("--task_config", type=str, required=True,
                    help="task yaml; primitives 字段中的 type 作为评估阶段")
parser.add_argument("--horizon", type=int, default=300)
parser.add_argument("--num_rollouts", type=int, default=5)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output_json", type=str, required=True)
parser.add_argument("--env_cfg_class", type=str, default=None,
                    help="完整模块路径, 例如 "
                         "isaaclab_tasks.real2sim.tasks.pick.config.sceneconfig.franka_fr3."
                         "ourgripper.ik_rel_vis_env_cfg:FrankaFr3OurGripperIKRelVisPickEnvCfg. "
                         "若不指定则沿用 play.py 默认的 pick env cfg.")
parser.add_argument("--pick_lift_threshold", type=float, default=0.05)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 强制 headless, 子进程评估不需要弹窗
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- 以下在 Isaac Sim 启动后才能 import ---
import cv2
import gymnasium as gym
import importlib
import numpy as np
import random
import torch
import torch.nn.functional as F
import yaml

import isaaclab_tasks.real2sim.tasks  # noqa: F401  保证 task 被注册
from isaaclab_tasks.real2sim.utils.EpisodeBuffer import replace_background_with_segmentation

# 注意: 不 import play.py, 因为 play.py 顶层有自己的 argparse 会吃掉本脚本的 sys.argv.
#       改为在本文件里直接实现需要复用的工具函数 (都很短).
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from network import DiffusionPolicy
from train import DDPMScheduler

from isaaclab.motion_collect.success.evaluators import (
    _evaluate_pick_success_with_metrics,
    _evaluate_place_pose_success_with_metrics,
    _evaluate_place_success_with_metrics,
    _evaluate_pour_success_with_metrics,
    _evaluate_hang_success_with_metrics,
    _evaluate_pull_success_with_metrics,
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


# ============================================================================
# 从 play.py 内联过来的工具函数 (避免 import play.py 触发其 argparse)
# ============================================================================

def load_norm_params(stat_path):
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
                full_path = cwd_path if os.path.exists(cwd_path) else os.path.normpath(
                    os.path.join(config_dir, bg_path)
                )
            img = cv2.imread(full_path)
            if img is not None:
                bg_images[key] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return bg_images


def process_image(image_tensor, seg_tensor=None, bg_image=None, target_size=256):
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
    if crop_size != target_size:
        image = F.interpolate(
            image.unsqueeze(0), size=(target_size, target_size),
            mode="bilinear", align_corners=False,
        ).squeeze(0)
    return image


@torch.no_grad()
def ddpm_sample(model, scheduler, obs_images, obs_lowdim, device):
    B = 1
    T_a = model.pred_horizon
    D_act = model.action_dim
    global_cond = model.encode_obs(obs_images, obs_lowdim)
    x = torch.randn(B, T_a, D_act, device=device)
    for t in reversed(range(scheduler.num_steps)):
        timestep = torch.full((B,), t, device=device, dtype=torch.long)
        noise_pred = model.noise_pred_net(x, timestep, global_cond)
        x = scheduler.step(noise_pred, t, x)
    return x.squeeze(0).cpu().numpy()


# ============================================================================
# task yaml -> 评估阶段
# ============================================================================

def load_eval_stages(task_config_path: str) -> list[dict]:
    """读取 task yaml 的 primitives 字段, 抽取评估阶段描述.

    返回 list, 每项形如:
        {
            "type": "pick",
            "target": "redcup",
            "place_target": None,        # 仅 place/pour 用
            "place_mode": None,          # place 用: "pose"/"on"/"in"/"semi_enclosed"
            "place_position": None,      # place mode=pose 用: [x,y,z] (env-frame)
            "place_orientation": None,   # place mode=pose 用: [w,x,y,z]
        }
    """
    with open(task_config_path) as f:
        cfg = yaml.safe_load(f) or {}
    primitives = cfg.get("primitives") or []
    stages = []
    for entry in primitives:
        if not isinstance(entry, dict):
            continue
        ptype = str(entry.get("type", "")).lower()
        if ptype not in ("pick", "place", "pour", "hang", "pull"):
            continue
        params = entry.get("params") or {}
        stage = {
            "type": ptype,
            "target": params.get("target"),
            "place_target": params.get("target") if ptype == "place" else None,
            "place_mode": str(entry.get("mode", "")).lower() if ptype == "place" else None,
            "place_position": params.get("position") if ptype == "place" else None,
            "place_orientation": params.get("orientation") if ptype == "place" else None,
            "pour_target": params.get("target") if ptype == "pour" else None,
            "hang_container": params.get("target") if ptype == "hang" else None,
            "hang_container_mount_npz": params.get("container_mount_affordance_npz_path") if ptype == "hang" else None,
            "pull_target": params.get("target") if ptype == "pull" else None,
        }
        stages.append(stage)
    return stages


def resolve_pick_target(stages: list[dict]) -> str | None:
    for s in stages:
        if s["type"] == "pick" and s.get("target"):
            return s["target"]
    return None


# ============================================================================
# 阶段判定
# ============================================================================

def evaluate_stages_at_episode_end(
    scene,
    stages: list[dict],
    pick_initial_z: float | None,
    pick_lift_threshold: float,
    pick_peak_z: float | None = None,
    min_pour_tilt_z: float | None = None,
    pour_crossed_ok: bool = False,
    pour_xy_at_cross: float | None = None,
    pull_peak_displacement: float | None = None,
) -> dict[str, dict]:
    """对每个阶段独立判定成功(单 env 评估, env_idx=0)."""
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
            else:
                place_target = stage.get("place_target")
                if not place_target or not pick_target:
                    results[key] = {"success": False, "reason": "no_place_or_pick_target"}
                    continue
                ok, metrics = _evaluate_place_success_with_metrics(
                    scene=scene,
                    pick_target=pick_target,
                    place_target=place_target,
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

    return results


# ============================================================================
# 自定义 rollout (基于 play.py 的实现, 但去掉 success_term 早停)
# ============================================================================

def rollout_until_horizon(model, scheduler, env, horizon, action_horizon, device,
                          norm_params, bg_images, obs_horizon, initial_obs_dict=None,
                          pick_target=None, pour_target=None, pull_target=None,
                          collect_front_images=False):
    """跑满 horizon 或直到 env terminate, 不依赖 env success_term.

    调用方应该在外部先 env.reset() (以便记录 baseline), 然后把返回的 obs_dict
    作为 initial_obs_dict 传进来; 这里就不会再 reset, 避免物体初始位置被重新随机化.
    """
    model.eval()
    if initial_obs_dict is None:
        obs_dict, _ = env.reset()
    else:
        obs_dict = initial_obs_dict

    obs_img_buffer = []
    obs_low_buffer = []
    seg_keys = {"front": "semantic_segmentation_side", "wrist": "semantic_segmentation_wrist"}
    rgb_keys = ["front", "wrist"]
    env_cam_h, env_cam_w = 480, 640
    train_img_size = 256

    front_frames = []  # 收集 front 图像用于生成 GIF
    pick_peak_z = None
    min_pour_tilt_z = None
    pour_crossed_ok = False
    pour_xy_at_cross = None
    pour_tilt_at_cross = None
    pull_peak_displacement = None

    def _update_peak():
        nonlocal pick_peak_z, min_pour_tilt_z, pour_crossed_ok, pour_xy_at_cross, pour_tilt_at_cross
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
        # 追踪 pour：当倾斜角度首次超过阈值时，检查 XY 距离
        if quat_w is not None:
            from isaaclab.motion_collect.success.constants import POUR_SUCCESS_TILT_THRESHOLD, POUR_SUCCESS_XY_THRESHOLD
            rot = _quat_wxyz_to_matrix(np.array(quat_w, dtype=float))
            tilt_z = float(rot[2, 2])
            if min_pour_tilt_z is None or tilt_z < min_pour_tilt_z:
                min_pour_tilt_z = tilt_z
            if not pour_crossed_ok and pour_tilt_at_cross is None and tilt_z <= POUR_SUCCESS_TILT_THRESHOLD:
                if pour_target:
                    pour_pos_w, _ = _get_object_pose_world_from_scene(env.scene, pour_target, 0)
                    if pour_pos_w is not None:
                        xy_dist = float(np.linalg.norm(pos_w[:2] - pour_pos_w[:2]))
                        pour_xy_at_cross = xy_dist
                        pour_tilt_at_cross = tilt_z
                        pour_crossed_ok = xy_dist <= POUR_SUCCESS_XY_THRESHOLD

    step = 0
    while step < horizon:
        obs = obs_dict["policy"]

        if step == 0:
            obs_dict, _, _, _, _ = env.step(torch.zeros(1, env.action_space.shape[1], device=device))
            step += 1
            if collect_front_images and "front" in obs_dict["policy"]:
                _fr = obs_dict["policy"]["front"].squeeze().float().reshape(3, env_cam_h, env_cam_w)
                _seg = obs_dict["policy"].get(seg_keys["front"], None)
                if _seg is not None:
                    _seg = _seg.squeeze()
                _bg = bg_images.get("front") if bg_images else None
                _processed = process_image(_fr, _seg, _bg, train_img_size)
                front_frames.append((_processed.cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8))
            continue

        images_processed = []
        for image_name in rgb_keys:
            if image_name in obs:
                img = obs[image_name].squeeze().float().reshape(3, env_cam_h, env_cam_w)
                seg = obs.get(seg_keys[image_name], None)
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
        while len(obs_img_buffer) < obs_horizon:
            obs_img_buffer.insert(0, obs_img_buffer[0])
            obs_low_buffer.insert(0, obs_low_buffer[0])

        obs_images = torch.stack(obs_img_buffer, dim=0).unsqueeze(0)
        obs_lowdim = torch.stack(obs_low_buffer, dim=0).unsqueeze(0)

        action_chunk = ddpm_sample(model, scheduler, obs_images, obs_lowdim, device)

        terminated = False
        truncated = False
        for j in range(min(action_horizon, horizon - step)):
            action = action_chunk[j]
            if norm_params is not None and "action" in norm_params:
                action[:-1] = minmax_denormalize(
                    action[:-1],
                    norm_params["action"]["min"][:-1],
                    norm_params["action"]["max"][:-1],
                )
            action[-1] = 1.0 if action[-1] > 0.0 else -1.0
            action_tensor = torch.from_numpy(action).float().to(device).view(1, -1)
            obs_dict, _, terminated, truncated, _ = env.step(action_tensor)
            step += 1
            _update_peak()
            if collect_front_images and "front" in obs_dict["policy"]:
                _fr = obs_dict["policy"]["front"].squeeze().float().reshape(3, env_cam_h, env_cam_w)
                _seg = obs_dict["policy"].get(seg_keys["front"], None)
                if _seg is not None:
                    _seg = _seg.squeeze()
                _bg = bg_images.get("front") if bg_images else None
                _processed = process_image(_fr, _seg, _bg, train_img_size)
                front_frames.append((_processed.cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8))
            if terminated or truncated:
                break

            if j < action_horizon - 1:
                obs = obs_dict["policy"]
                images_processed = []
                for image_name in rgb_keys:
                    if image_name in obs:
                        img = obs[image_name].squeeze().float().reshape(3, env_cam_h, env_cam_w)
                        seg = obs.get(seg_keys[image_name], None)
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

        if terminated or truncated:
            break

    return obs_dict, pick_peak_z, min_pour_tilt_z, pour_crossed_ok, pour_xy_at_cross, pull_peak_displacement, front_frames


# ============================================================================
# Main
# ============================================================================

def frames_to_gif(frames: list[np.ndarray], save_path: str,
                  target_frames: int = 30, target_size: int = 256) -> str | None:
    """将采集的帧列表抽帧、缩放后保存为 GIF.

    Args:
        frames: list of (H, W, 3) uint8 numpy arrays
        save_path: 保存路径
        target_frames: 目标帧数 (从原始帧中均匀采样)
        target_size: 缩放到的正方形尺寸
    Returns:
        保存路径, 或 None 表示失败
    """
    if not frames:
        return None
    from PIL import Image

    # 均匀抽帧
    n = len(frames)
    if n > target_frames:
        indices = np.linspace(0, n - 1, target_frames, dtype=int)
    else:
        indices = np.arange(n)
    sampled = [frames[i] for i in indices]

    # 缩放并转 PIL
    pil_frames = []
    for f in sampled:
        img = Image.fromarray(f)
        img = img.resize((target_size, target_size), Image.BILINEAR)
        pil_frames.append(img)

    # 保存 GIF, duration 按总时长 ~5s 分配
    duration_per_frame = max(5000 // len(pil_frames), 50)
    pil_frames[0].save(
        save_path, save_all=True, append_images=pil_frames[1:],
        duration=duration_per_frame, loop=0,
    )
    return save_path


def _import_env_cfg_class(spec: str):
    """spec 形如 'pkg.mod:ClassName'."""
    if ":" not in spec:
        raise ValueError(f"--env_cfg_class 必须是 'pkg.mod:ClassName' 形式, 收到: {spec}")
    mod_path, cls_name = spec.split(":", 1)
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args_cli.output_json = os.path.abspath(args_cli.output_json)
    print(f"[EVAL] ===== eval_rollout main started =====")
    print(f"[EVAL] device={device}, ckpt={args_cli.checkpoint}")
    print(f"[EVAL] cwd={os.getcwd()}")
    print(f"[EVAL] output_json(abs)={args_cli.output_json}")

    # --- 加载 ckpt + 模型 ---
    ckpt = torch.load(args_cli.checkpoint, map_location=device)
    model_args = ckpt.get("args", {})
    model = DiffusionPolicy(
        action_dim=model_args.get("action_dim", 7),
        obs_horizon=model_args.get("obs_horizon", 2),
        pred_horizon=model_args.get("pred_horizon", 16),
        lowdim_dim=model_args.get("lowdim_dim", 14),
        img_feature_dim=model_args.get("img_feature_dim", 256),
        lowdim_feature_dim=model_args.get("lowdim_feature_dim", 64),
        down_dims=tuple(model_args.get("down_dims", [256, 512, 1024])),
        pretrained_resnet=False,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    scheduler = DDPMScheduler(num_diffusion_steps=model_args.get("num_diffusion_steps", 100))
    action_horizon = model_args.get("action_horizon", 3)
    obs_horizon = model_args.get("obs_horizon", 2)

    # --- env ---
    if args_cli.env_cfg_class:
        env_cfg_cls = _import_env_cfg_class(args_cli.env_cfg_class)
    else:
        from isaaclab_tasks.real2sim.tasks.config.franka_fr3.ourgripper.ik_rel_vis_env_cfg import (
            FrankaFr3OurGripperIKRelVisPickEnvCfg,
        )
        env_cfg_cls = FrankaFr3OurGripperIKRelVisPickEnvCfg

    from isaaclab_tasks.real2sim.tasks.pick_r2s_env_cfg import create_pick_env_cfg_from_yaml
    env_cfg = create_pick_env_cfg_from_yaml(
        yaml_path=args_cli.task_config,
        env_cfg_class=env_cfg_cls,
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

    # 把 task yaml 元信息挂到 scene 上, place success 评估需要 position_range
    with open(args_cli.task_config) as f:
        task_yaml = yaml.safe_load(f) or {}
    attach_scene_success_meta(env.scene, task_yaml)

    # 评估阶段
    stages = load_eval_stages(args_cli.task_config)
    pick_target = resolve_pick_target(stages)
    pour_target = next((s.get("pour_target") for s in stages if s["type"] == "pour"), None)
    pull_target = next((s.get("pull_target") for s in stages if s["type"] == "pull"), None)
    print(f"[EVAL] stages={[s['type'] for s in stages]}, pick_target={pick_target}, pour_target={pour_target}, pull_target={pull_target}")

    # 种子
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)

    norm_params = load_norm_params(args_cli.stat) if args_cli.stat else None
    bg_images = load_background_images(args_cli.task_config)

    # --- N 次 rollout ---
    per_rollout_results = []
    gif_dir = os.path.join(os.path.dirname(os.path.abspath(args_cli.output_json)), "gifs")
    os.makedirs(gif_dir, exist_ok=True)
    gif_paths = []
    for trial in range(args_cli.num_rollouts):
        print(f"[EVAL] trial {trial + 1}/{args_cli.num_rollouts}")
        # 这一次 reset 既用于记录 pick baseline, 也作为 rollout 的起始 obs,
        # 避免被再次 reset 重新随机化物体位置导致 baseline 对不上 rollout 实际状态.
        initial_obs_dict, _ = env.reset()
        for _ in range(10):
            env.step(torch.zeros(1, env.action_space.shape[1], device=device))
        pick_initial_z = capture_object_initial_z(env.scene, pick_target, env_idx=0) if pick_target else None

        _, pick_peak_z, min_pour_tilt_z, pour_crossed_ok, pour_xy_at_cross, pull_peak_displacement, front_frames = rollout_until_horizon(
            model=model,
            scheduler=scheduler,
            env=env,
            horizon=args_cli.horizon,
            action_horizon=action_horizon,
            device=device,
            norm_params=norm_params,
            bg_images=bg_images,
            obs_horizon=obs_horizon,
            initial_obs_dict=initial_obs_dict,
            pick_target=pick_target,
            pour_target=pour_target,
            pull_target=pull_target,
            collect_front_images=True,
        )

        # 保存 GIF (抽帧到 30 帧, 缩放到 256x256)
        gif_path = os.path.join(gif_dir, f"trial_{trial}.gif")
        saved = frames_to_gif(front_frames, gif_path, target_frames=30, target_size=256)
        if saved:
            gif_paths.append(saved)
            print(f"[EVAL] saved GIF: {saved} ({len(front_frames)} frames -> 30 sampled)")

        stage_results = evaluate_stages_at_episode_end(
            scene=env.scene,
            stages=stages,
            pick_initial_z=pick_initial_z,
            pick_lift_threshold=args_cli.pick_lift_threshold,
            pick_peak_z=pick_peak_z,
            min_pour_tilt_z=min_pour_tilt_z,
            pour_crossed_ok=pour_crossed_ok,
            pour_xy_at_cross=pour_xy_at_cross,
            pull_peak_displacement=pull_peak_displacement,
        )
        per_rollout_results.append(stage_results)
        stage_summary = ", ".join(
            f"{k}={'OK' if v.get('success') else 'FAIL'}" for k, v in stage_results.items()
        )
        print(f"[EVAL] trial {trial}: {stage_summary}")
        for k, v in stage_results.items():
            metrics = {mk: mv for mk, mv in v.items() if mk != "success" and mk != "reason"}
            if metrics:
                metrics_str = ", ".join(f"{mk}={mv:.4f}" if isinstance(mv, float) else f"{mk}={mv}" for mk, mv in metrics.items())
                print(f"  [{k}] success={v.get('success')}, {metrics_str}")

    # --- 聚合: 每个阶段的成功率 ---
    stage_keys: list[str] = []
    for r in per_rollout_results:
        for k in r.keys():
            if k not in stage_keys:
                stage_keys.append(k)

    stage_success_rate: dict[str, float] = {}
    for key in stage_keys:
        succ = sum(1 for r in per_rollout_results if r.get(key, {}).get("success"))
        stage_success_rate[key] = succ / max(len(per_rollout_results), 1)

    # 全任务成功率: 所有阶段都成功才算
    overall_success = 0
    for r in per_rollout_results:
        if stage_keys and all(r.get(k, {}).get("success") for k in stage_keys):
            overall_success += 1
    overall_rate = overall_success / max(len(per_rollout_results), 1)

    summary = {
        "checkpoint": args_cli.checkpoint,
        "num_rollouts": args_cli.num_rollouts,
        "stage_success_rate": stage_success_rate,
        "overall_success_rate": overall_rate,
        "per_rollout": per_rollout_results,
        "gif_paths": gif_paths,
    }

    Path(os.path.dirname(os.path.abspath(args_cli.output_json))).mkdir(parents=True, exist_ok=True)
    with open(args_cli.output_json, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[EVAL] wrote {args_cli.output_json}")
    print(f"[EVAL] stage_success_rate={stage_success_rate}, overall={overall_rate:.2%}")

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
