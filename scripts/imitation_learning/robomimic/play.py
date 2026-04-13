# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play and evaluate a trained policy from robomimic.

This script loads a robomimic policy and plays it in an Isaac Lab environment.

Args:
    task: Name of the environment.
    checkpoint: Path to the robomimic policy checkpoint.
    horizon: If provided, override the step horizon of each rollout.
    num_rollouts: If provided, override the number of rollouts.
    seed: If provided, overeride the default random seed.
    norm_factor_min: If provided, minimum value of the action space normalization factor.
    norm_factor_max: If provided, maximum value of the action space normalization factor.
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate robomimic policy for Isaac Lab environment.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Pytorch model checkpoint to load.")
parser.add_argument("--horizon", type=int, default=500, help="Step horizon of each rollout.")
parser.add_argument("--num_rollouts", type=int, default=1, help="Number of rollouts.")
parser.add_argument("--seed", type=int, default=101, help="Random seed.")
parser.add_argument(
    "--norm_factor_min", type=float, default=None, help="Optional: minimum value of the normalization factor."
)
parser.add_argument(
    "--norm_factor_max", type=float, default=None, help="Optional: maximum value of the normalization factor."
)
parser.add_argument("--enable_pinocchio", default=False, action="store_true", help="Enable Pinocchio.")
parser.add_argument(
    "--dataset_stat", type=str,
    default=None,
    help="Path to dataset_stat.json with min-max normalization params.",
)
parser.add_argument("--config", type=str, default=None, help="Path to task YAML config file for task_gen environments.")
path_Pour_bean = Path("source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/task_gen/config/sceneconfig/generated_task_config.yaml")
parser.add_argument(
    "--task_config", 
    type=str, 
    default=path_Pour_bean, 
    help="Path to YAML task config file (e.g., tasks/pick_up_cube.yaml)"
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and not the one installed by Isaac Sim
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import copy
import gymnasium as gym
import json
import numpy as np
import random
import torch

import cv2
import os
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import torch.nn.functional as F

from isaaclab_tasks.real2sim.utils.EpisodeBuffer import replace_background_with_segmentation


# ============================================================================
# 归一化/反归一化工具
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
    """min-max 归一化到 [-1, 1]."""
    drange = dmax - dmin
    drange = np.where(drange < 1e-8, 1.0, drange)
    return 2.0 * (data - dmin) / drange - 1.0


def minmax_denormalize(data, dmin, dmax):
    """从 [-1, 1] 反归一化回原始尺度."""
    drange = dmax - dmin
    return (data + 1.0) / 2.0 * drange + dmin

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401

from isaaclab_tasks.utils import parse_env_cfg
import isaaclab_tasks.real2sim.tasks.task_gen  # noqa: F401


def load_background_images(task_config_path):
    """从 YAML 配置中加载背景图片."""
    import yaml
    with open(task_config_path) as f:
        cfg = yaml.safe_load(f)
    runtime = cfg.get("runtime", {})
    config_dir = os.path.dirname(os.path.abspath(task_config_path))

    bg_images = {}
    for key, bg_key in [("front", "cam_side_background"), ("wrist", "cam_wrist_background")]:
        bg_path = runtime.get(bg_key)
        if bg_path:
            full_path = os.path.normpath(os.path.join(config_dir, bg_path))
            img = cv2.imread(full_path)
            if img is not None:
                bg_images[key] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                print(f"[INFO] 加载背景图片 {key}: {full_path} -> {img.shape}")
            else:
                print(f"[WARN] 无法加载背景图片: {full_path}")
    return bg_images


def rollout(policy, env, success_term, horizon, device, norm_params=None, bg_images=None):
    """Perform a single rollout of the policy in the environment.

    Args:
        policy: The robomimic policy to play.
        env: The environment to play in.
        horizon: The step horizon of each rollout.
        device: The device to run the policy on.
        norm_params: min-max normalization parameters from dataset_stat.json.
        bg_images: dict of background images for background replacement.

    Returns:
        terminated: Whether the rollout terminated.
        traj: The trajectory of the rollout.
    """
    policy.start_episode()
    obs_dict, _ = env.reset()
    traj = dict(actions=[])

    for i in range(horizon):
        # Skip first frame (renderer not ready, produces garbage image)
        if i == 0:
            obs_dict, _, _, _, _ = env.step(torch.zeros(1, env.action_space.shape[1], device=device))
            continue

        # Prepare observations
        obs = copy.deepcopy(obs_dict["policy"])
        for ob in obs:
            obs[ob] = torch.squeeze(obs[ob])

        # 对低维 obs 做 min-max 归一化 (和训练数据一致)
        if norm_params is not None:
            for key in ["qpos", "qvel"]:
                if key in obs and key in norm_params:
                    raw = obs[key].cpu().numpy()
                    normed = minmax_normalize(raw, norm_params[key]["min"], norm_params[key]["max"])
                    obs[key] = torch.from_numpy(normed).to(device=device)

        # Process RGB image observations
        rgb_keys = ["front", "wrist"]
        seg_keys = {"front": "semantic_segmentation_side", "wrist": "semantic_segmentation_wrist"}
        channels = 3
        env_cam_h, env_cam_w = 480, 640
        train_img_size = 256

        for image_name in rgb_keys:
            if image_name in obs:
                image = obs[image_name].clone().float()
                image = image.reshape(channels, env_cam_h, env_cam_w)  # (C, H, W)

                # 背景替换: 用语义分割 mask + 真实背景图片
                seg_key = seg_keys[image_name]
                if bg_images and image_name in bg_images and seg_key in obs:
                    seg = obs[seg_key].cpu().numpy()  # flattened
                    seg = seg.reshape(4, env_cam_h, env_cam_w)  # (4, H, W) RGBA
                    seg_rgba = seg.transpose(1, 2, 0)  # (H, W, 4)

                    rgb_np = image.cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
                    composite, _ = replace_background_with_segmentation(
                        rgb_np, seg_rgba, bg_images[image_name]
                    )
                    # composite 是 float [0, 1], shape (H, W, 3)
                    image = torch.from_numpy(composite.transpose(2, 0, 1)).float()  # (C, H, W)

                # Center crop to square, then resize to training resolution
                _, h, w = image.shape
                crop_size = min(h, w)
                y0 = (h - crop_size) // 2
                x0 = (w - crop_size) // 2
                image = image[:, y0:y0+crop_size, x0:x0+crop_size]  # (C, crop, crop)
                if crop_size != train_img_size:
                    image = image.unsqueeze(0)
                    image = F.interpolate(image, size=(train_img_size, train_img_size), mode="bilinear", align_corners=False)
                    image = image.squeeze(0)  # (C, 256, 256)
                obs[image_name] = image

        # 移除语义分割 obs (不送入模型)
        for seg_key in seg_keys.values():
            obs.pop(seg_key, None)

        # Save model input images for first 3 frames only (debug)
        if i <= 3:
            debug_dir = "/home/lmy/code/Real2Sim2Real-Pipeline-auto_task/debug_images"
            os.makedirs(debug_dir, exist_ok=True)
            for image_name in rgb_keys:
                if image_name in obs:
                    vis = obs[image_name].detach().cpu().numpy()  # (C, H, W)
                    vis = np.transpose(vis, (1, 2, 0))  # (H, W, C) RGB
                    vis = np.clip(vis * 255, 0, 255).astype(np.uint8)
                    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                    path = os.path.join(debug_dir, f"model_input_{image_name}_step{i}.png")
                    cv2.imwrite(path, vis_bgr)

        # Compute actions
        with torch.no_grad():
            actions = policy(obs)
        print(f"[DEBUG] step {i} raw actions: {np.round(actions, 4)}")

        # 反归一化 ee_delta (前6维), gripper 单独处理
        if norm_params is not None and "action" in norm_params:
            actions[:-1] = minmax_denormalize(
                actions[:-1], norm_params["action"]["min"][:-1], norm_params["action"]["max"][:-1]
            )

        # Gripper 二值化: 模型输出的是 logit (BCEWithLogitsLoss 训练)
        # sigmoid(logit) > 0.5 → close (+1), 否则 → open (-1)
        gripper_prob = 1.0 / (1.0 + np.exp(-actions[-1]))  # sigmoid
        actions[-1] = 1.0 if gripper_prob > 0.5 else -1.0
        print(f"[DEBUG] step {i} gripper prob: {gripper_prob:.4f}, action: {actions[-1]}")

        actions = torch.from_numpy(actions).to(device=device).view(1, env.action_space.shape[1])

        # Apply actions
        obs_dict, _, terminated, truncated, _ = env.step(actions)
        print(f"[DEBUG] step {i} rendered actions: {np.round(actions.cpu().numpy(), 4)}")
        obs = obs_dict["policy"]

        # Record trajectory
        traj["actions"].append(actions.cpu().tolist())

        # Check if rollout was successful
        if success_term is not None and bool(success_term.func(env, **success_term.params)[0]):
            return True, traj
        elif terminated or truncated:
            return False, traj

    return False, traj


def main():
    """Run a trained policy from robomimic with Isaac Lab environment."""
    # parse configuration
    if args_cli.task_config is not None:
        # Use YAML config to create environment (for task_gen environments)
        from isaaclab_tasks.real2sim.tasks.task_gen.gen_r2s_env_cfg import create_gen_env_cfg_from_yaml
        from isaaclab_tasks.real2sim.tasks.task_gen.config.sceneconfig.franka_fr3.ourgripper.ik_rel_vis_env_cfg import FrankaFr3OurGripperIKRelVisGenEnvCfg
        env_cfg = create_gen_env_cfg_from_yaml(
            yaml_path=args_cli.task_config,
            env_cfg_class=FrankaFr3OurGripperIKRelVisGenEnvCfg,
            num_envs=1,
            device=args_cli.device,
        )
    else:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)

    # Set observations to dictionary mode for Robomimic
    env_cfg.observations.policy.concatenate_terms = False

    # Disable termination conditions for play
    env_cfg.terminations.time_out = None
    if hasattr(env_cfg.terminations, "object_dropping"):
        env_cfg.terminations.object_dropping = None

    # Disable recorder
    env_cfg.recorders = None

    # Extract success checking function (if defined)
    success_term = getattr(env_cfg.terminations, "success", None)
    if success_term is not None:
        env_cfg.terminations.success = None

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Set seed
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)
    env.seed(args_cli.seed)

    # Acquire device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # 加载归一化参数
    norm_params = None
    if args_cli.dataset_stat is not None:
        norm_params = load_norm_params(args_cli.dataset_stat)
        print(f"[INFO] 加载归一化参数: {args_cli.dataset_stat}")
    else:
        print("[WARN] 未指定 --dataset_stat, 不做归一化/反归一化")

    # 加载背景图片 (用于背景替换)
    bg_images = {}
    if args_cli.task_config is not None:
        bg_images = load_background_images(args_cli.task_config)

    # Run policy
    results = []
    for trial in range(args_cli.num_rollouts):
        print(f"[INFO] Starting trial {trial}")
        policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=args_cli.checkpoint, device=device)
        terminated, traj = rollout(policy, env, success_term, args_cli.horizon, device, norm_params=norm_params, bg_images=bg_images)
        results.append(terminated)
        print(f"[INFO] Trial {trial}: {terminated}\n")

    print(f"\nSuccessful trials: {results.count(True)}, out of {len(results)} trials")
    print(f"Success rate: {results.count(True) / len(results)}")
    print(f"Trial Results: {results}\n")

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
