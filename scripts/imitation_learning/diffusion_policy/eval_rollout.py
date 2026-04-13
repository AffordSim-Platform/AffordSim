"""
Diffusion Policy evaluation subprocess script (invoked during training).

Design goals:
- Launched synchronously by train.py via subprocess.run after each checkpoint save.
- Starts Isaac Lab simulation, loads checkpoint, runs N rollouts.
- Does not rely on the env's built-in success termination; instead uses the
  ``type`` field (pick / place / pour) in the ``primitives`` list from the task
  yaml as evaluation stages, judging success independently for each stage at
  the end of every episode.
- Writes results as JSON to ``--output_json`` for train.py to read back into
  TensorBoard.

Usage (typically called automatically by train.py):
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
                    help="task yaml; the 'type' field in primitives defines evaluation stages")
parser.add_argument("--horizon", type=int, default=300)
parser.add_argument("--num_rollouts", type=int, default=5)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output_json", type=str, required=True)
parser.add_argument("--env_cfg_class", type=str, default=None,
                    help="Full module path, e.g. "
                         "isaaclab_tasks.real2sim.tasks.pick.config.sceneconfig.franka_fr3."
                         "ourgripper.ik_rel_vis_env_cfg:FrankaFr3OurGripperIKRelVisPickEnvCfg. "
                         "If not specified, falls back to the default pick env cfg from play.py.")
parser.add_argument("--pick_lift_threshold", type=float, default=0.05)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Force headless mode; subprocess evaluation does not need a GUI window
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Imports below require Isaac Sim to be running ---
import cv2
import gymnasium as gym
import importlib
import numpy as np
import random
import torch
import torch.nn.functional as F
import yaml

import isaaclab_tasks.real2sim.tasks.pick  # noqa: F401  ensure task is registered
from isaaclab_tasks.real2sim.utils.EpisodeBuffer import replace_background_with_segmentation

# Note: do not import play.py, because its top-level argparse would consume
#       this script's sys.argv. Instead, reusable helpers are implemented
#       directly in this file (they are short).
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
# Utility functions inlined from play.py (to avoid triggering its argparse)
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
# task yaml -> evaluation stages
# ============================================================================

def load_eval_stages(task_config_path: str) -> list[dict]:
    """Read the primitives field from the task yaml and extract evaluation stage descriptions.

    Returns a list, each item of the form:
        {
            "type": "pick",
            "target": "redcup",
            "place_target": None,        # only used for place/pour
            "place_mode": None,          # for place: "pose"/"on"/"in"/"semi_enclosed"
            "place_position": None,      # for place mode=pose: [x,y,z] (env-frame)
            "place_orientation": None,   # for place mode=pose: [w,x,y,z]
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


# ============================================================================
# Stage evaluation
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
# Custom rollout (based on play.py, but without success_term early stopping)
# ============================================================================

def rollout_until_horizon(model, scheduler, env, horizon, action_horizon, device,
                          norm_params, bg_images, obs_horizon, initial_obs_dict=None,
                          pick_target=None, pour_target=None, collect_front_images=False):
    """Run until horizon or env termination, without relying on env success_term.

    The caller should call env.reset() externally first (to record baselines),
    then pass the returned obs_dict as initial_obs_dict; this function will not
    reset again, avoiding re-randomization of object initial positions.
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

    front_frames = []  # Collect front images for GIF generation
    pick_peak_z = None
    min_pour_tilt_z = None
    pour_crossed_ok = False
    pour_xy_at_cross = None
    pour_tilt_at_cross = None

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

    return obs_dict, pick_peak_z, min_pour_tilt_z, pour_crossed_ok, pour_xy_at_cross, front_frames


# ============================================================================
# Main
# ============================================================================

def frames_to_gif(frames: list[np.ndarray], save_path: str,
                  target_frames: int = 30, target_size: int = 256) -> str | None:
    """Subsample and resize collected frames, then save as GIF.

    Args:
        frames: list of (H, W, 3) uint8 numpy arrays
        save_path: output file path
        target_frames: target frame count (uniformly sampled from original frames)
        target_size: square resize dimension
    Returns:
        save path, or None on failure
    """
    if not frames:
        return None
    from PIL import Image

    # Uniformly subsample frames
    n = len(frames)
    if n > target_frames:
        indices = np.linspace(0, n - 1, target_frames, dtype=int)
    else:
        indices = np.arange(n)
    sampled = [frames[i] for i in indices]

    # Resize and convert to PIL
    pil_frames = []
    for f in sampled:
        img = Image.fromarray(f)
        img = img.resize((target_size, target_size), Image.BILINEAR)
        pil_frames.append(img)

    # Save GIF, allocating duration to target ~5s total
    duration_per_frame = max(5000 // len(pil_frames), 50)
    pil_frames[0].save(
        save_path, save_all=True, append_images=pil_frames[1:],
        duration=duration_per_frame, loop=0,
    )
    return save_path


def _import_env_cfg_class(spec: str):
    """spec should be in the form 'pkg.mod:ClassName'."""
    if ":" not in spec:
        raise ValueError(f"--env_cfg_class must be in 'pkg.mod:ClassName' form, got: {spec}")
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

    # --- Load checkpoint + model ---
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

    # --- Environment ---
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

    # Attach task yaml metadata to scene; place success evaluation needs position_range
    with open(args_cli.task_config) as f:
        task_yaml = yaml.safe_load(f) or {}
    attach_scene_success_meta(env.scene, task_yaml)

    # Evaluation stages
    stages = load_eval_stages(args_cli.task_config)
    pick_target = resolve_pick_target(stages)
    pour_target = next((s.get("pour_target") for s in stages if s["type"] == "pour"), None)
    print(f"[EVAL] stages={[s['type'] for s in stages]}, pick_target={pick_target}, pour_target={pour_target}")

    # Seed
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)

    norm_params = load_norm_params(args_cli.stat) if args_cli.stat else None
    bg_images = load_background_images(args_cli.task_config)

    # --- N rollouts ---
    per_rollout_results = []
    gif_dir = os.path.join(os.path.dirname(os.path.abspath(args_cli.output_json)), "gifs")
    os.makedirs(gif_dir, exist_ok=True)
    gif_paths = []
    for trial in range(args_cli.num_rollouts):
        print(f"[EVAL] trial {trial + 1}/{args_cli.num_rollouts}")
        # This reset both records the pick baseline and serves as the starting obs
        # for the rollout, avoiding a second reset that would re-randomize object
        # positions and cause baseline mismatch with the actual rollout state.
        initial_obs_dict, _ = env.reset()
        for _ in range(10):
            env.step(torch.zeros(1, env.action_space.shape[1], device=device))
        pick_initial_z = capture_object_initial_z(env.scene, pick_target, env_idx=0) if pick_target else None

        _, pick_peak_z, min_pour_tilt_z, pour_crossed_ok, pour_xy_at_cross, front_frames = rollout_until_horizon(
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
            collect_front_images=True,
        )

        # Save GIF (subsample to 30 frames, resize to 256x256)
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

    # --- Aggregate: per-stage success rate ---
    stage_keys: list[str] = []
    for r in per_rollout_results:
        for k in r.keys():
            if k not in stage_keys:
                stage_keys.append(k)

    stage_success_rate: dict[str, float] = {}
    for key in stage_keys:
        succ = sum(1 for r in per_rollout_results if r.get(key, {}).get("success"))
        stage_success_rate[key] = succ / max(len(per_rollout_results), 1)

    # Overall task success rate: all stages must succeed
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
