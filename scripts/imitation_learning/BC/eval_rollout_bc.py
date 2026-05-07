"""
BC .

 train_bc.py  save_checkpoint  subprocess.run .
 Isaac Lab,  BC Gaussian ckpt,  N  rollout.
 JSON  --output_json,  train_bc.py  tensorboard.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="BC Eval Rollout (subprocess)")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--task_config", type=str, required=True,
                    help="task yaml; primitives  type ")
parser.add_argument("--dataset_stat", type=str, default=None,
                    help="dataset_stat.json (,  norm_params)")
parser.add_argument("--horizon", type=int, default=300)
parser.add_argument("--num_rollouts", type=int, default=5)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output_json", type=str, required=True)
parser.add_argument("--env_cfg_class", type=str, default=None,
                    help=",  'pkg.mod:ClassName'.")
parser.add_argument("--pick_lift_threshold", type=float, default=0.05)
parser.add_argument("--action_horizon", type=int, default=None,
                    help="chunk ; ")
parser.add_argument("--action_chunk_size", type=int, default=None,
                    help="action chunk size;  checkpoint ")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---  import ---
import copy
import cv2
import gymnasium as gym
import importlib
import numpy as np
import random
import torch
import torch.nn.functional as F
import yaml

import isaaclab_tasks.real2sim.tasks  # noqa: F401
from isaaclab_tasks.real2sim.utils.EpisodeBuffer import replace_background_with_segmentation

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from train_bc import BCGaussianPolicy, minmax_normalize, minmax_denormalize

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

def load_bc_checkpoint(ckpt_path, device, dataset_stat, action_chunk_size=None):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    norm_params = ckpt.get("norm_params", None)
    if norm_params is None:
        norm_json = os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), "norm_params.json")
        if os.path.exists(norm_json):
            with open(norm_json) as f:
                norm_params = json.load(f)

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

    print(f"[EVAL_BC] loaded ckpt: state_dim={state_dim}, action_dim={action_dim}, "
          f"action_chunk_size={action_chunk_size}")
    return policy, norm_params, action_chunk_size


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
                #  CWD () 
                cwd_path = os.path.abspath(bg_path)
                if os.path.exists(cwd_path):
                    full_path = cwd_path
                else:
                    full_path = os.path.normpath(os.path.join(config_dir, bg_path))
            img = cv2.imread(full_path)
            if img is not None:
                bg_images[key] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                print(f"[EVAL_BC]  {key}: {full_path}")
            else:
                print(f"[EVAL_BC] :  {key}: {full_path}")
    return bg_images


def preprocess_image(image_tensor, seg_tensor=None, bg_image=None,
                     env_cam_h=480, env_cam_w=640, train_img_size=256):
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
        image = F.interpolate(image.unsqueeze(0), size=(train_img_size, train_img_size),
                              mode="bilinear", align_corners=False).squeeze(0)
    image = image.permute(1, 2, 0)
    return image


def postprocess_action(action_np, norm_params):
    """  to   action."""
    result = action_np.copy()
    result[:6] = result[:6] / 100.0
    if norm_params is not None and "action_ee" in norm_params:
        result[:6] = minmax_denormalize(
            result[:6],
            np.array(norm_params["action_ee"]["min"], dtype=np.float32),
            np.array(norm_params["action_ee"]["max"], dtype=np.float32),
        )
    # Gripper: continuous [0, 1]  to  env (-1=open, +1=close)
    gripper_cont = result[6]
    result[6] = -1.0 if gripper_cont > 0.5 else 1.0
    return result


# ============================================================================
# Task yaml  to  
# ============================================================================

def load_eval_stages(task_config_path):
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
        stages.append({
            "type": ptype,
            "target": params.get("target"),
            "place_target": params.get("target") if ptype == "place" else None,
            "place_mode": str(entry.get("mode", "")).lower() if ptype == "place" else None,
            "place_position": params.get("position") if ptype == "place" else None,
            "place_orientation": params.get("orientation") if ptype == "place" else None,
            "pour_target": params.get("target") if ptype == "pour" else None,
        })
    return stages


def resolve_pick_target(stages):
    for s in stages:
        if s["type"] == "pick" and s.get("target"):
            return s["target"]
    return None


def evaluate_stages_at_episode_end(scene, stages, pick_initial_z, pick_lift_threshold,
                                    pick_peak_z=None, min_pour_tilt_z=None,
                                    pour_crossed_ok=False, pour_xy_at_cross=None):
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
                min_tilt_z=min_pour_tilt_z,
                pour_crossed_ok=pour_crossed_ok,
                pour_xy_at_cross=pour_xy_at_cross,
            )
            results[key] = {"success": bool(ok), "metrics": metrics}
    return results


# ============================================================================
# GIF
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
# Rollout
# ============================================================================

def rollout_until_horizon(policy, env, horizon, device, norm_params, bg_images,
                          action_chunk_size, action_horizon, initial_obs_dict,
                          pick_target=None, pour_target=None,
                          collect_front_images=False):
    obs_dict = initial_obs_dict
    action_chunk_buffer = []
    front_frames = []
    seg_keys = {"front": "semantic_segmentation_side", "wrist": "semantic_segmentation_wrist"}
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

    for step in range(horizon):
        if step == 0:
            obs_dict, _, _, _, _ = env.step(torch.zeros(1, env.action_space.shape[1], device=device))
            continue

        obs = copy.deepcopy(obs_dict["policy"])
        for k in obs:
            obs[k] = torch.squeeze(obs[k])

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

        front_processed = preprocess_image(
            obs["front"], seg_tensor=obs.get(seg_keys["front"]),
            bg_image=bg_images.get("front") if bg_images else None,
        )
        if collect_front_images:
            front_raw = (front_processed.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            front_frames.append(front_raw)

        need_inference = len(action_chunk_buffer) == 0
        if need_inference:
            front_img = front_processed.unsqueeze(0).to(device)
            wrist_img = preprocess_image(
                obs["wrist"], seg_tensor=obs.get(seg_keys["wrist"]),
                bg_image=bg_images.get("wrist") if bg_images else None,
            ).unsqueeze(0).to(device)

            # get_action  (1, C, 7) 
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
            print(f"  [DBG step={step}] ee={action_env[:3]}... gripper_raw={action_np[6]:.3f} gripper_env={action_env[6]:.0f}")

        action_tensor = torch.from_numpy(action_env).to(device).view(1, env.action_space.shape[1])
        obs_dict, _, terminated, truncated, _ = env.step(action_tensor)
        _update_peak()
        if terminated or truncated:
            break

    return obs_dict, pick_peak_z, min_pour_tilt_z, pour_crossed_ok, pour_xy_at_cross, front_frames


# ============================================================================
# Main
# ============================================================================

def _import_env_cfg_class(spec):
    if ":" not in spec:
        raise ValueError(f"--env_cfg_class  'pkg.mod:ClassName' , : {spec}")
    mod_path, cls_name = spec.split(":", 1)
    return getattr(importlib.import_module(mod_path), cls_name)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args_cli.output_json = os.path.abspath(args_cli.output_json)
    print(f"[EVAL_BC] ===== eval_rollout_bc main started =====")
    print(f"[EVAL_BC] device={device}, ckpt={args_cli.checkpoint}")

    policy, norm_params, action_chunk_size = load_bc_checkpoint(
        args_cli.checkpoint, device, args_cli.dataset_stat,
        action_chunk_size=args_cli.action_chunk_size,
    )

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

    with open(args_cli.task_config) as f:
        task_yaml = yaml.safe_load(f) or {}
    attach_scene_success_meta(env.scene, task_yaml)

    stages = load_eval_stages(args_cli.task_config)
    pick_target = resolve_pick_target(stages)
    pour_target = next((s.get("pour_target") for s in stages if s["type"] == "pour"), None)
    print(f"[EVAL_BC] stages={[s['type'] for s in stages]}, pick_target={pick_target}, pour_target={pour_target}")

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)

    bg_images = load_background_images(args_cli.task_config)

    per_rollout_results = []
    gif_dir = os.path.join(os.path.dirname(os.path.abspath(args_cli.output_json)), "gifs")
    os.makedirs(gif_dir, exist_ok=True)
    gif_paths = []
    for trial in range(args_cli.num_rollouts):
        print(f"[EVAL_BC] trial {trial + 1}/{args_cli.num_rollouts}")
        initial_obs_dict, _ = env.reset()
        for _ in range(10):
            env.step(torch.zeros(1, env.action_space.shape[1], device=device))
        pick_initial_z = capture_object_initial_z(env.scene, pick_target, env_idx=0) if pick_target else None

        _, pick_peak_z, min_pour_tilt_z, pour_crossed_ok, pour_xy_at_cross, front_frames = rollout_until_horizon(
            policy=policy, env=env, horizon=args_cli.horizon, device=device,
            norm_params=norm_params, bg_images=bg_images,
            action_chunk_size=action_chunk_size,
            action_horizon=args_cli.action_horizon,
            initial_obs_dict=initial_obs_dict,
            pick_target=pick_target,
            pour_target=pour_target,
            collect_front_images=True,
        )

        gif_path = os.path.join(gif_dir, f"trial_{trial}.gif")
        saved = frames_to_gif(front_frames, gif_path, target_frames=30, target_size=256)
        if saved:
            gif_paths.append(saved)
            print(f"[EVAL_BC] saved GIF: {saved} ({len(front_frames)} frames  to  30 sampled)")

        stage_results = evaluate_stages_at_episode_end(
            scene=env.scene, stages=stages,
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
        print(f"[EVAL_BC] trial {trial}: {stage_summary}")
        for k, v in stage_results.items():
            metrics = {mk: mv for mk, mv in v.items() if mk != "success" and mk != "reason"}
            if metrics:
                metrics_str = ", ".join(f"{mk}={mv:.4f}" if isinstance(mv, float) else f"{mk}={mv}" for mk, mv in metrics.items())
                print(f"  [{k}] success={v.get('success')}, {metrics_str}")

    # 
    stage_keys = []
    for r in per_rollout_results:
        for k in r.keys():
            if k not in stage_keys:
                stage_keys.append(k)
    stage_success_rate = {}
    for key in stage_keys:
        succ = sum(1 for r in per_rollout_results if r.get(key, {}).get("success"))
        stage_success_rate[key] = succ / max(len(per_rollout_results), 1)

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
    Path(os.path.dirname(args_cli.output_json)).mkdir(parents=True, exist_ok=True)
    with open(args_cli.output_json, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[EVAL_BC] wrote {args_cli.output_json}")
    print(f"[EVAL_BC] stage_success_rate={stage_success_rate}, overall={overall_rate:.2%}")

    env.close()


if __name__ == "__main__":
    import traceback
    exit_code = 0
    try:
        main()
    except SystemExit as e:
        exit_code = int(e.code) if isinstance(e.code, int) else 1
    except BaseException:
        exit_code = 1
        print(f"[EVAL_BC] ===== UNCAUGHT EXCEPTION =====")
        traceback.print_exc()
        print(f"[EVAL_BC] ===== END EXCEPTION =====")
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass
    sys.exit(exit_code)
