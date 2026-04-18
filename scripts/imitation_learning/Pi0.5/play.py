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

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

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
)
from isaaclab.motion_collect.success.scene_helpers import (
    _get_articulation_joint_pos,
    _get_object_pose_world_from_scene,
    attach_scene_success_meta,
)
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
        if ptype not in ("pick", "place", "pour", "hang", "pull"):
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
        }
        stages.append(stage)
    return stages


def resolve_pick_target(stages: list[dict]) -> str | None:
    for s in stages:
        if s["type"] == "pick" and s.get("target"):
            return s["target"]
    return None


# ============================================================================
# Stage evaluation
# ============================================================================

def evaluate_stages_at_episode_end(
    scene,
    stages: list[dict],
    pick_initial_z: float | None,
    pick_lift_threshold: float,
    pick_peak_z: float | None = None,
    pull_peak_displacement: float | None = None,
) -> dict[str, dict]:
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

def rollout(policy, env, horizon, action_horizon, device, prompt,
            bg_images=None, trial_id=0, initial_obs_dict=None,
            pick_target=None, pull_target=None, collect_front_images=True,
            image_speckle_noise_std: float = 0.0):
    """Run one episode with Pi0.5 remote policy."""
    debug_dir = f"debug_images/pi05_trial_{trial_id}"
    os.makedirs(debug_dir, exist_ok=True)

    if initial_obs_dict is None:
        obs_dict, _ = env.reset()
    else:
        obs_dict = initial_obs_dict

    seg_keys = {"front": "semantic_segmentation_side", "wrist": "semantic_segmentation_wrist"}
    env_cam_h, env_cam_w = 480, 640

    front_frames = []
    pick_peak_z = None
    pull_peak_displacement = None

    def _update_peak():
        nonlocal pick_peak_z, pull_peak_displacement
        if pull_target:
            from isaaclab.motion_collect.success.constants import PULL_SUCCESS_JOINT_NAME
            jp = _get_articulation_joint_pos(env.scene, pull_target, PULL_SUCCESS_JOINT_NAME, 0)
            if jp is not None:
                disp = abs(float(jp))
                if pull_peak_displacement is None or disp > pull_peak_displacement:
                    pull_peak_displacement = disp
        if not pick_target:
            return
        pos_w, _ = _get_object_pose_world_from_scene(env.scene, pick_target, 0)
        if pos_w is None:
            return
        z = float(pos_w[2])
        if pick_peak_z is None or z > pick_peak_z:
            pick_peak_z = z

    step = 0
    while step < horizon:
        obs = obs_dict["policy"]

        # Skip first frame (renderer not ready)
        if step == 0:
            obs_dict, _, _, _, _ = env.step(torch.zeros(1, env.action_space.shape[1], device=device))
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
        qvel = obs["qvel"].squeeze().cpu().numpy().astype(np.float32) if "qvel" in obs else np.zeros_like(qpos)
        state = np.concatenate([qpos, qvel]).astype(np.float32)

        pi05_obs = {
            "observation/image": front_img,          # (H, W, 3) uint8
            "observation/wrist_image": wrist_img,    # (H, W, 3) uint8
            "observation/state": state,              # (14,) float32
            "prompt": prompt,
        }

        # --- Query remote policy server ---
        result = policy.infer(pi05_obs)
        actions_chunk = np.asarray(result["actions"])  # (action_horizon, 7)

        # --- Execute actions ---
        n_exec = min(action_horizon, actions_chunk.shape[0], horizon - step)
        terminated = False
        truncated = False
        for idx in range(n_exec):
            action_7d = actions_chunk[idx]
            action_tensor = torch.from_numpy(action_7d).float().to(device).view(1, -1)
            obs_dict, _, terminated, truncated, _ = env.step(action_tensor)
            step += 1
            if step <= 5 or step % 50 == 0:
                print(f"[EVAL] step={step}, action={np.array2string(action_7d, precision=4, suppress_small=True)}")
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

            obs = obs_dict["policy"]

        if terminated or truncated:
            break

    return obs_dict, pick_peak_z, pull_peak_displacement, front_frames


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
    pick_target = resolve_pick_target(stages)
    pull_target = next((s.get("pull_target") for s in stages if s["type"] == "pull"), None)
    print(f"[EVAL] stages={[s['type'] for s in stages]}, pick_target={pick_target}, pull_target={pull_target}")

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
            env.step(torch.zeros(1, env.action_space.shape[1], device=device))
        pick_initial_z = capture_object_initial_z(env.scene, pick_target, env_idx=0) if pick_target else None

        _, pick_peak_z, pull_peak_displacement, front_frames = rollout(
            policy=policy,
            env=env,
            horizon=args_cli.horizon,
            action_horizon=args_cli.action_horizon,
            device=device,
            prompt=args_cli.prompt,
            bg_images=bg_images,
            trial_id=trial,
            initial_obs_dict=initial_obs_dict,
            pick_target=pick_target,
            pull_target=pull_target,
            collect_front_images=True,
            image_speckle_noise_std=image_speckle_noise_std,
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
