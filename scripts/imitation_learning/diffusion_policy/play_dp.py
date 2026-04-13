"""
Diffusion Policy inference script.

Runs a trained Diffusion Policy in an Isaac Lab environment.

DDPM sampling procedure:
1. Encode current observations (obs_horizon frames)
2. Start from pure noise x_T ~ N(0, I)
3. Iteratively denoise x_T -> x_{T-1} -> ... -> x_0
4. x_0 is the predicted action sequence (pred_horizon steps)
5. Execute only the first action_horizon steps, then re-plan

Usage:
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

# Isaac Lab launch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Diffusion Policy Inference")
parser.add_argument("--task", type=str, default="Pick_Place_banana-v0", help="Environment name")
parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
parser.add_argument("--stat", type=str, default=None, help="dataset_stat.json path")
parser.add_argument("--horizon", type=int, default=500, help="Max steps")
parser.add_argument("--num_rollouts", type=int, default=1, help="Number of rollouts")
parser.add_argument("--seed", type=int, default=None, help="Random seed (None=random)")
parser.add_argument("--save_img_interval", type=int, default=1, help="Save debug images every N steps (0=disabled)")
parser.add_argument(
    "--task_config",
    type=str,
    default="source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/task_gen/config/sceneconfig/generated_task_config.yaml",
    help="YAML task configuration file path",
)
parser.add_argument("--pick_lift_threshold", type=float, default=0.05)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Imports below require Isaac Sim to be running ---
import cv2
import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn.functional as F
import yaml

from isaaclab_tasks.utils import parse_env_cfg
import isaaclab_tasks.real2sim.tasks.pick  # noqa: F401
from isaaclab_tasks.real2sim.utils.EpisodeBuffer import replace_background_with_segmentation

# Add diffusion_policy directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from network import DiffusionPolicy
from scripts.imitation_learning.diffusion_policy.train_dp import DDPMScheduler

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
# Utility functions
# ============================================================================

def load_norm_params(stat_path):
    """Load min-max normalization parameters from dataset_stat.json."""
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
    """Load background images from YAML config."""
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
                # Resolve relative to CWD first; fall back to yaml directory if not found
                cwd_path = os.path.abspath(bg_path)
                if os.path.exists(cwd_path):
                    full_path = cwd_path
                else:
                    full_path = os.path.normpath(os.path.join(config_dir, bg_path))
            img = cv2.imread(full_path)
            if img is not None:
                bg_images[key] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                print(f"[INFO] Loaded background image {key}: {full_path} -> {img.shape}")
    return bg_images


def process_image(image_tensor, seg_tensor=None, bg_image=None, target_size=256):
    """Process RGB image: background replacement + center crop + resize.

    Args:
        image_tensor: (3, H, W) float tensor, range [0, 1]
        seg_tensor: optional semantic segmentation tensor
        bg_image: optional background image numpy array (H, W, 3)
        target_size: target resolution

    Returns:
        (3, target_size, target_size) float tensor, range [0, 1]
    """
    image = image_tensor.clone().float()
    C, H, W = image.shape

    # Background replacement
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
# task yaml -> evaluation stages
# ============================================================================

def load_eval_stages(task_config_path: str) -> list[dict]:
    """Read the primitives field from task yaml and extract evaluation stage descriptions."""
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
    scene,
    stages: list[dict],
    pick_initial_z: float | None,
    pick_lift_threshold: float,
    pick_peak_z: float | None = None,
    min_pour_tilt_z: float | None = None,
    pour_crossed_ok: bool = False,
    pour_xy_at_cross: float | None = None,
) -> dict[str, dict]:
    """Independently evaluate success for each stage (single env, env_idx=0)."""
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

    return results


# ============================================================================
# GIF utils
# ============================================================================

def frames_to_gif(frames: list[np.ndarray], save_path: str,
                  target_frames: int = 30, target_size: int = 256) -> str | None:
    """Subsample and resize collected frames, then save as GIF."""
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
# DDPM Sampling
# ============================================================================

@torch.no_grad()
def ddpm_sample(model, scheduler, obs_images, obs_lowdim, device):
    """DDPM sampling: generate action sequence from noise.

    Args:
        model: DiffusionPolicy model
        scheduler: DDPMScheduler
        obs_images: (1, T_o, 2, 3, H, W) observation images
        obs_lowdim: (1, T_o, D_low) low-dimensional observations

    Returns:
        actions: (T_a, D_act) predicted action sequence (numpy)
    """
    B = 1
    T_a = model.pred_horizon
    D_act = model.action_dim

    # Encode observation (only needs to be done once)
    global_cond = model.encode_obs(obs_images, obs_lowdim)

    # Start from pure noise
    x = torch.randn(B, T_a, D_act, device=device)

    # Iterative denoising T -> 0
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
            pick_target=None, pour_target=None, initial_obs_dict=None):
    """Execute a single rollout.

    Diffusion Policy receding horizon control:
    - Predict pred_horizon steps of actions at a time
    - Execute only the first action_horizon steps
    - Then re-observe and re-plan

    Args:
        pick_target: used to track object z peak (for pick success evaluation).
        initial_obs_dict: if not None, skip reset and start rollout from this obs.

    Returns:
        (success_by_term, pick_peak_z, front_frames): success_term result, object z peak, front frame list.
    """
    # Create image save directory
    _tc = getattr(args_cli, "task_config", None)
    task_label = os.path.splitext(os.path.basename(_tc))[0] if _tc else "unknown"
    debug_img_dir = f"/home/lmy/code/Real2Sim2Real-Pipeline-auto_task/debug_images/{task_label}/trial_{trial_id}"
    os.makedirs(debug_img_dir, exist_ok=True)

    model.eval()
    if initial_obs_dict is None:
        obs_dict, _ = env.reset()
    else:
        obs_dict = initial_obs_dict

    pick_peak_z = None
    min_pour_tilt_z = None
    pour_crossed_ok = False  # Whether XY distance is satisfied when tilt angle first exceeds threshold
    pour_xy_at_cross = None  # XY distance at first threshold crossing
    pour_tilt_at_cross = None  # tilt_z at first threshold crossing
    front_frames = []

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
        # Track pour: check XY distance when tilt angle first exceeds threshold
        if quat_w is not None:
            from isaaclab.motion_collect.success.constants import POUR_SUCCESS_TILT_THRESHOLD, POUR_SUCCESS_XY_THRESHOLD
            rot = _quat_wxyz_to_matrix(np.array(quat_w, dtype=float))
            tilt_z = float(rot[2, 2])
            if min_pour_tilt_z is None or tilt_z < min_pour_tilt_z:
                min_pour_tilt_z = tilt_z
            # Record when tilt threshold is first exceeded
            if not pour_crossed_ok and pour_tilt_at_cross is None and tilt_z <= POUR_SUCCESS_TILT_THRESHOLD:
                if pour_target:
                    pour_pos_w, _ = _get_object_pose_world_from_scene(env.scene, pour_target, 0)
                    if pour_pos_w is not None:
                        xy_dist = float(np.linalg.norm(pos_w[:2] - pour_pos_w[:2]))
                        pour_xy_at_cross = xy_dist
                        pour_tilt_at_cross = tilt_z
                        pour_crossed_ok = xy_dist <= POUR_SUCCESS_XY_THRESHOLD

    # Observation buffer (keeps the most recent obs_horizon frames)
    obs_img_buffer = []   # list of (2, 3, H, W) tensors
    obs_low_buffer = []   # list of (D_low,) tensors

    seg_keys = {"front": "semantic_segmentation_side", "wrist": "semantic_segmentation_wrist"}
    rgb_keys = ["front", "wrist"]
    env_cam_h, env_cam_w = 480, 640
    train_img_size = 256

    step = 0
    while step < horizon:
        # --- Get current observation ---
        obs = obs_dict["policy"]

        # Skip first frame (renderer not ready)
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

        # Process images
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

        # Process low-dimensional observations
        qpos = obs["qpos"].squeeze().cpu().numpy()
        qvel = obs["qvel"].squeeze().cpu().numpy()
        if norm_params is not None:
            qpos = minmax_normalize(qpos, norm_params["qpos"]["min"], norm_params["qpos"]["max"])
            qvel = minmax_normalize(qvel, norm_params["qvel"]["min"], norm_params["qvel"]["max"])
        obs_low = torch.from_numpy(np.concatenate([qpos, qvel])).float().to(device)

        # Update buffer
        obs_img_buffer.append(obs_img)
        obs_low_buffer.append(obs_low)

        # Keep buffer length = obs_horizon
        if len(obs_img_buffer) > obs_horizon:
            obs_img_buffer.pop(0)
            obs_low_buffer.pop(0)

        # Pad buffer (repeat first frame if buffer is too short)
        while len(obs_img_buffer) < obs_horizon:
            obs_img_buffer.insert(0, obs_img_buffer[0])
            obs_low_buffer.insert(0, obs_low_buffer[0])

        # Assemble observation tensors
        obs_images = torch.stack(obs_img_buffer, dim=0).unsqueeze(0)  # (1, T_o, 2, 3, H, W)
        obs_lowdim = torch.stack(obs_low_buffer, dim=0).unsqueeze(0)  # (1, T_o, D_low)

        # Save current observation images
        if save_img_interval > 0 and step % save_img_interval == 0:
            for cam_idx, cam_name in enumerate(rgb_keys):
                img_save = (images_processed[cam_idx].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(debug_img_dir, f"step_{step:04d}_{cam_name}.png"),
                            cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR))

        # --- DDPM sampling ---
        action_chunk = ddpm_sample(model, scheduler, obs_images, obs_lowdim, device)
        # action_chunk: (T_a, D_act)

        # --- Execute the first action_horizon steps ---
        for j in range(min(action_horizon, horizon - step)):
            action = action_chunk[j]  # (D_act,)

            # Denormalize ee_delta (first 6 dims)
            if norm_params is not None and "action" in norm_params:
                action[:-1] = minmax_denormalize(
                    action[:-1],
                    norm_params["action"]["min"][:-1],
                    norm_params["action"]["max"][:-1],
                )

            # Gripper binarization (threshold 0)
            action[-1] = 1.0 if action[-1] > 0.0 else -1.0

            action_tensor = torch.from_numpy(action).float().to(device).view(1, -1)

            if step <= 5 or step % 50 == 0:
                print(f"[DEBUG] step {step} action: {np.round(action, 4)}")

            obs_dict, _, terminated, truncated, _ = env.step(action_tensor)
            step += 1
            _update_peak()

            # Collect front frames
            if "front" in obs_dict["policy"]:
                _fr = obs_dict["policy"]["front"].squeeze().float().reshape(3, env_cam_h, env_cam_w)
                _seg = obs_dict["policy"].get(seg_keys["front"], None)
                if _seg is not None:
                    _seg = _seg.squeeze()
                _bg = bg_images.get("front") if bg_images else None
                _processed = process_image(_fr, _seg, _bg, train_img_size)
                front_frames.append((_processed.cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8))

            # Check success
            if success_term is not None and bool(success_term.func(env, **success_term.params)[0]):
                print(f"[INFO] Success! step={step}")
                return True, pick_peak_z, min_pour_tilt_z, pour_crossed_ok, pour_xy_at_cross, front_frames

            if terminated or truncated:
                return False, pick_peak_z, min_pour_tilt_z, pour_crossed_ok, pour_xy_at_cross, front_frames

            # If not the last step, update observations (for next chunk's buffer)
            if j < action_horizon - 1:
                obs = obs_dict["policy"]
                # Process images
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

                # Save observation images from inner loop
                if save_img_interval > 0 and step % save_img_interval == 0:
                    for cam_idx, cam_name in enumerate(rgb_keys):
                        img_save = (images_processed[cam_idx].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                        cv2.imwrite(os.path.join(debug_img_dir, f"step_{step:04d}_{cam_name}.png"),
                                    cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR))

    return False, pick_peak_z, min_pour_tilt_z, pour_crossed_ok, pour_xy_at_cross, front_frames


# ============================================================================
# Main
# ============================================================================

def main():
    # --- Load model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
    ckpt = torch.load(args_cli.checkpoint, map_location=device)

    # Restore model parameters from checkpoint
    model_args = ckpt.get("args", {})
    model = DiffusionPolicy(
        action_dim=model_args.get("action_dim", 7),
        obs_horizon=model_args.get("obs_horizon", 2),
        pred_horizon=model_args.get("pred_horizon", 16),
        lowdim_dim=model_args.get("lowdim_dim", 14),
        img_feature_dim=model_args.get("img_feature_dim", 256),
        lowdim_feature_dim=model_args.get("lowdim_feature_dim", 64),
        down_dims=tuple(model_args.get("down_dims", [256, 512, 1024])),
        pretrained_resnet=False,  # No need to reload pretrained weights during inference
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[INFO] Model loaded, epoch={ckpt.get('epoch', 'unknown')}")

    # DDPM scheduler
    scheduler = DDPMScheduler(
        num_diffusion_steps=model_args.get("num_diffusion_steps", 100),
    )

    action_horizon = model_args.get("action_horizon", 3)
    obs_horizon = model_args.get("obs_horizon", 2)

    # --- Environment setup ---
    from isaaclab_tasks.real2sim.tasks.pick.pick_r2s_env_cfg import create_pick_env_cfg_from_yaml
    from isaaclab_tasks.real2sim.tasks.pick.config.sceneconfig.franka_fr3.ourgripper.ik_rel_vis_env_cfg import FrankaFr3OurGripperIKRelVisPickEnvCfg

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

    # Attach task yaml metadata to scene; success evaluation needs position_range
    with open(args_cli.task_config) as f:
        task_yaml = yaml.safe_load(f) or {}
    attach_scene_success_meta(env.scene, task_yaml)

    # Evaluation stages
    stages = load_eval_stages(args_cli.task_config)
    pick_target = resolve_pick_target(stages)
    pour_target = next((s.get("pour_target") for s in stages if s["type"] == "pour"), None)
    print(f"[INFO] stages={[s['type'] for s in stages]}, pick_target={pick_target}, pour_target={pour_target}")

    # Set random seed
    if args_cli.seed is not None:
        torch.manual_seed(args_cli.seed)
        np.random.seed(args_cli.seed)
        random.seed(args_cli.seed)

    # Load normalization parameters
    norm_params = None
    if args_cli.stat is not None:
        norm_params = load_norm_params(args_cli.stat)
        print(f"[INFO] Loaded normalization parameters: {args_cli.stat}")

    # Load background images
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
        # Reset and record pick baseline
        initial_obs_dict, _ = env.reset()
        for _ in range(10):
            env.step(torch.zeros(1, env.action_space.shape[1], device=device))
        pick_initial_z = capture_object_initial_z(env.scene, pick_target, env_idx=0) if pick_target else None

        success, pick_peak_z, min_pour_tilt_z, pour_crossed_ok, pour_xy_at_cross, front_frames = rollout(
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
            initial_obs_dict=initial_obs_dict,
        )
        results.append(success)

        # Save GIF
        gif_path = os.path.join(gif_dir, f"trial_{trial}.gif")
        saved = frames_to_gif(front_frames, gif_path, target_frames=30, target_size=256)
        if saved:
            print(f"[INFO] saved GIF: {saved} ({len(front_frames)} frames -> 30 sampled)")

        # Primitive-based success evaluation
        stage_results = evaluate_stages_at_episode_end(
            scene=env.scene,
            stages=stages,
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

    # Summary
    print(f"\n[INFO] === Results Summary ===")
    print(f"Success term: {results.count(True)}/{len(results)} = {results.count(True) / len(results):.2%}")

    # Per-stage success rate
    stage_keys: list[str] = []
    for r in all_stage_results:
        for k in r.keys():
            if k not in stage_keys:
                stage_keys.append(k)
    for key in stage_keys:
        succ = sum(1 for r in all_stage_results if r.get(key, {}).get("success"))
        print(f"  {key}: {succ}/{len(all_stage_results)} = {succ / len(all_stage_results):.2%}")

    # Overall task success rate
    overall = sum(
        1 for r in all_stage_results
        if stage_keys and all(r.get(k, {}).get("success") for k in stage_keys)
    )
    print(f"  overall: {overall}/{len(all_stage_results)} = {overall / len(all_stage_results):.2%}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
