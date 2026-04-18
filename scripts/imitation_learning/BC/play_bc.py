"""
BC 策略推理脚本 — 在 Isaac Lab 环境中 play 训练好的 BC Gaussian 模型.

用法:
  python scripts/imitation_learning/BC/play_bc.py \
    --task Pick_Place_banana-v0 \
    --checkpoint runs/bc_xxx/checkpoints/best_eval.pt \
    --task_config source/isaaclab_tasks/.../pick_banana_place_in_plate.yaml \
    --horizon 300 --num_rollouts 10
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play BC policy in Isaac Lab environment.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to BC checkpoint (.pt)")
parser.add_argument("--horizon", type=int, default=500, help="Max steps per episode.")
parser.add_argument("--num_rollouts", type=int, default=10, help="Number of rollout trials.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--task_config", type=str, default=None, help="Path to YAML task config file.")
parser.add_argument("--dataset_stat", type=str, default=None,
                    help="Path to dataset_stat JSON with _minmax_params for denormalization.")
parser.add_argument("--save_inputs", type=str, default=None,
                    help="保存 play 时模型输入到 HDF5 文件. 例如: /tmp/play_inputs.hdf5")
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
from train_bc import BCGaussianPolicy, minmax_normalize, minmax_denormalize

import isaaclab_tasks.real2sim.tasks.pick  # noqa: F401
import isaaclab_tasks.real2sim.tasks.task_gen  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from isaaclab.motion_collect.success.evaluators import (
    _evaluate_pick_success_with_metrics,
    _evaluate_place_pose_success_with_metrics,
    _evaluate_place_success_with_metrics,
    _evaluate_pour_success_with_metrics,
)
from isaaclab.motion_collect.success.scene_helpers import (
    _get_object_pose_world_from_scene,
    attach_scene_success_meta,
)
from isaaclab.motion_collect.success.success import (
    capture_object_initial_z,
)
from isaaclab.motion_collect.tools.pose_utils import _quat_wxyz_to_matrix


# ============================================================================
# Helpers
# ============================================================================

def load_checkpoint(ckpt_path, device, dataset_stat=None, action_chunk_size=None):
    """加载 BC Gaussian checkpoint, 返回 policy 和 norm_params."""
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
    if action_chunk_size is None:
        action_chunk_size = model_cfg.get("action_chunk_size", 1)

    policy = BCGaussianPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        action_chunk_size=action_chunk_size,
    ).to(device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    # 用 dataset_stat 覆盖 norm_params
    if dataset_stat:
        with open(dataset_stat) as f:
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
            print(f"[INFO] Overriding norm_params with _minmax_params from {dataset_stat}")

    print(f"[INFO] Loaded policy: state_dim={state_dim}, action_dim={action_dim}, "
          f"action_chunk_size={action_chunk_size}")
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
    """处理环境 RGB 观测 → 模型输入.

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
# Task yaml → 评估阶段
# ============================================================================

def load_eval_stages(task_config_path: str) -> list[dict]:
    with open(task_config_path) as f:
        cfg = yaml.safe_load(f) or {}
    primitives = cfg.get("primitives") or []
    stages = []
    for entry in primitives:
        if not isinstance(entry, dict):
            continue
        ptype = str(entry.get("type", "")).lower()
        if ptype not in ("pick", "place", "pour"):
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
    pick_peak_z=None, min_pour_tilt_z=None,
    pour_crossed_ok=False, pour_xy_at_cross=None,
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
            else:
                place_target = stage.get("place_target")
                if not place_target or not pick_target:
                    results[key] = {"success": False, "reason": "no_place_or_pick_target"}
                    continue
                ok, metrics = _evaluate_place_success_with_metrics(
                    scene=scene, pick_target=pick_target,
                    place_target=place_target, env_idx=0,
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

    action_np: (7,) — [ee_delta(6) * 100, gripper_continuous]
    返回: (7,) — [ee_delta(6), gripper(-1 or +1)]
    """
    result = action_np.copy()

    # EE: 训练时放大了 100 倍, 缩回来
    result[:6] = result[:6] / 100.0

    # EE: 反归一化
    if norm_params is not None and "action_ee" in norm_params:
        result[:6] = minmax_denormalize(
            result[:6],
            np.array(norm_params["action_ee"]["min"], dtype=np.float32),
            np.array(norm_params["action_ee"]["max"], dtype=np.float32),
        )

    # Gripper: continuous [0, 1] → env action (-1=open, +1=close)
    # 0 = close → +1, 1 = open → -1
    gripper_cont = result[6]
    result[6] = -1.0 if gripper_cont > 0.5 else 1.0

    return result


# ============================================================================
# Rollout
# ============================================================================

def rollout(policy, env, success_term, horizon, device, norm_params,
            bg_images=None, save_inputs_path=None, trial_id=0,
            save_img_interval=0, pick_target=None, pour_target=None,
            action_chunk_size=1, action_horizon=None, initial_obs_dict=None):
    """执行一次 rollout."""

    _tc = getattr(args_cli, "task_config", None)
    task_label = os.path.splitext(os.path.basename(_tc))[0] if _tc else "unknown"
    debug_img_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "debug_images", task_label, f"trial_{trial_id}",
    )
    if save_img_interval > 0:
        os.makedirs(debug_img_dir, exist_ok=True)

    # 使用外部传入的 obs_dict, 避免重复 reset
    if initial_obs_dict is not None:
        obs_dict = initial_obs_dict
    else:
        obs_dict, _ = env.reset()

    pick_peak_z = None
    min_pour_tilt_z = None
    pour_crossed_ok = False
    pour_xy_at_cross = None
    pour_tilt_at_cross = None
    front_frames = []
    action_chunk_buffer = []

    _save_fronts, _save_wrists, _save_states = [], [], []

    def _update_peak():
        nonlocal pick_peak_z, min_pour_tilt_z, pour_crossed_ok, pour_xy_at_cross, pour_tilt_at_cross
        if not pick_target:
            return
        pos_w, quat_w = _get_object_pose_world_from_scene(env.scene, pick_target, 0)
        if pos_w is None:
            return
        z = float(pos_w[2])
        if pick_peak_z is None or z > pick_peak_z:
            pick_peak_z = z
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

            if save_inputs_path is not None:
                _save_fronts.append((front_img.squeeze(0).cpu().numpy() * 255).astype(np.uint8))
                _save_wrists.append((wrist_img.squeeze(0).cpu().numpy() * 255).astype(np.uint8))
                _save_states.append(state.copy())

            # Inference: get_action 返回 (1, C, 7) 均值
            action = policy.get_action(front_img, wrist_img, state_tensor)
            action_raw = action.cpu().numpy().squeeze(0)  # (C, 7)

            if action_raw.ndim == 2:
                ah = action_horizon or action_raw.shape[0]
                for a in action_raw[:ah]:
                    action_chunk_buffer.append(a.copy())
            else:
                action_chunk_buffer.append(action_raw.copy())

        action_np = action_chunk_buffer.pop(0)

        # Post-process: denormalize EE, threshold gripper
        action_env = postprocess_action(action_np, norm_params)

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
            _dump_play_inputs(save_inputs_path, _save_fronts, _save_wrists, _save_states)
            return True, pick_peak_z, min_pour_tilt_z, pour_crossed_ok, pour_xy_at_cross, front_frames
        elif terminated or truncated:
            _dump_play_inputs(save_inputs_path, _save_fronts, _save_wrists, _save_states)
            return False, pick_peak_z, min_pour_tilt_z, pour_crossed_ok, pour_xy_at_cross, front_frames

    _dump_play_inputs(save_inputs_path, _save_fronts, _save_wrists, _save_states)
    return False, pick_peak_z, min_pour_tilt_z, pour_crossed_ok, pour_xy_at_cross, front_frames


def _dump_play_inputs(save_path, fronts, wrists, states):
    if save_path is None or len(fronts) == 0:
        return
    import h5py
    N = len(fronts)
    with h5py.File(save_path, "w") as f:
        grp = f.create_group("data/demo_0")
        grp.create_dataset("obs/front", data=np.stack(fronts))
        grp.create_dataset("obs/wrist", data=np.stack(wrists))
        grp.create_dataset("obs/qpos", data=np.stack([s[:7] for s in states]))
        grp.create_dataset("obs/qvel", data=np.stack([s[7:] for s in states]))
        dummy_actions = np.zeros((N, 7), dtype=np.float32)
        dummy_actions[:, -1] = 1.0
        grp.create_dataset("actions", data=dummy_actions)
    print(f"\n[SAVED] Play inputs ({N} steps) -> {save_path}")


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
        dataset_stat=args_cli.dataset_stat,
        action_chunk_size=args_cli.action_chunk_size,
    )

    # Create environment
    if args_cli.task_config is not None:
        from isaaclab_tasks.real2sim.tasks.pick.pick_r2s_env_cfg import create_pick_env_cfg_from_yaml
        from isaaclab_tasks.real2sim.tasks.pick.config.sceneconfig.franka_fr3.ourgripper.ik_rel_vis_env_cfg import (
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
    if args_cli.task_config is not None:
        stages = load_eval_stages(args_cli.task_config)
        pick_target = resolve_pick_target(stages)
        pour_target = next((s.get("pour_target") for s in stages if s["type"] == "pour"), None)
    print(f"[INFO] stages={[s['type'] for s in stages]}, pick_target={pick_target}, pour_target={pour_target}")

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
        "..", "..", "..", "debug_images", f"bc_play_gifs_{task_label}",
    )
    os.makedirs(gif_dir, exist_ok=True)

    for trial in range(args_cli.num_rollouts):
        print(f"\n[INFO] Trial {trial}")
        initial_obs_dict, _ = env.reset()
        for _ in range(10):
            warmup_obs, _, _, _, _ = env.step(torch.zeros(1, env.action_space.shape[1], device=device))
        pick_initial_z = capture_object_initial_z(env.scene, pick_target, env_idx=0) if pick_target else None

        save_path = args_cli.save_inputs if trial == 0 else None
        success, pick_peak_z, min_pour_tilt_z, pour_crossed_ok, pour_xy_at_cross, front_frames = rollout(
            policy, env, success_term, args_cli.horizon, device,
            norm_params=norm_params, bg_images=bg_images,
            save_inputs_path=save_path,
            trial_id=trial,
            save_img_interval=args_cli.save_img_interval,
            pick_target=pick_target,
            pour_target=pour_target,
            action_chunk_size=action_chunk_size,
            action_horizon=action_horizon,
            initial_obs_dict=warmup_obs,
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
                min_pour_tilt_z=min_pour_tilt_z,
                pour_crossed_ok=pour_crossed_ok,
                pour_xy_at_cross=pour_xy_at_cross,
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
