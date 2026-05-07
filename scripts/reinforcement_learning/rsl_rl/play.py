# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
from pathlib import Path

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")

path_Pour_bean = Path("/home/hk/limingyang/code/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/pour/config/sceneconfig/pour_beans.yaml")
parser.add_argument(
    "--task_config", 
    type=str, 
    default=path_Pour_bean, 
    help="Path to YAML task config file (e.g., tasks/pick_up_cube.yaml)"
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
print("[INFO] RSL-RL CLI arguments added to the parser.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
from tensordict import TensorDict

from rsl_rl.runners import DistillationRunner, OnPolicyRunner
from isaaclab_rl.rsl_rl.ViT_BC.runners import OnPolicyRunnerViTBC, OnPolicyRunnerViT
from isaaclab_tasks.real2sim.utils.EpisodeBuffer import replace_background_with_segmentation
# from isaaclab_rl.rsl_rl.ViT_BC.runners.on_policy_runner_vitbc import unnormalize_actions
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)

# Global variables for background replacement
_bg_side_image = None
_bg_wrist_image = None
_bg_save_counter = 0
_log_dir = None

def replace_observation_background(obs: TensorDict, log_dir: str = None) -> TensorDict:
        """Replace background in RGB observations using semantic segmentation
        
        Args:
            obs: Observation TensorDict from environment
            log_dir: Directory to save example images
            
        Returns:
            Modified observation with background replaced in RGB images
        """
        import numpy as np
        import cv2
        from pathlib import Path
        
        global _bg_side_image, _bg_wrist_image, _bg_save_counter, _log_dir
        
        # Update log directory
        if log_dir is not None:
            _log_dir = log_dir
        
        # Load background images if not already loaded
        if _bg_side_image is None:
            bg_side_path = "/home/hk/limingyang/code/IsaacLab/backgrounds/rightsidebackground_realworld_128x128.png"
            bg_img = cv2.imread(bg_side_path)
            _bg_side_image = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
            
        if _bg_wrist_image is None:
            bg_wrist_path = "/home/hk/limingyang/code/IsaacLab/backgrounds/wristbackground_realworld.jpg"
            bg_img = cv2.imread(bg_wrist_path)
            _bg_wrist_image = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)

            h, w = bg_img.shape[:2]
            crop_height = int(h * 2 / 3)  # Keep top 2/3
            bg_img_cropped = bg_img[:crop_height, :]
            _bg_wrist_image = cv2.resize(bg_img_cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        
        ee_pose_obs = False
        if ee_pose_obs :
            batch_size = obs['policy'].shape[0]
            # Extract components from flattened observation
            # Order: arm_ee_pose(7), gripper_pos(2), arm_ee_vel(6), seg_side(65536), seg_wrist(65536), ext_cam(49152), wrist_cam(49152)
            start_idx = 0
            arm_ee_pose = obs['policy'][:, start_idx:start_idx+7]
            start_idx += 7
            gripper_pos = obs['policy'][:, start_idx:start_idx+2]
            start_idx += 2
            arm_ee_vel = obs['policy'][:, start_idx:start_idx+6]
            start_idx += 6
            seg_side = obs['policy'][:, start_idx:start_idx+65536]  # 4 * 128 * 128 = 65536 (RGBA)
            start_idx += 65536
            seg_wrist = obs['policy'][:, start_idx:start_idx+65536]  # 4 * 128 * 128 = 65536 (RGBA)
            start_idx += 65536
            ext_cam = obs['policy'][:, start_idx:start_idx+49152]  # 3 * 128 * 128 = 49152 (RGB)
            start_idx += 49152
            wrist_cam = obs['policy'][:, start_idx:start_idx+49152]  # 3 * 128 * 128 = 49152 (RGB)
        else:
            batch_size = obs['policy'].shape[0]
            # Extract components from flattened observation
            # Order: joint_pos(9), joint_vel(9), actions(7), seg_side(65536), seg_wrist(65536), ext_cam(49152), wrist_cam(49152)
            start_idx = 0
            joint_pos = obs['policy'][:, start_idx:start_idx+9]
            start_idx += 9
            joint_vel = obs['policy'][:, start_idx:start_idx+9]
            start_idx += 9
            seg_side = obs['policy'][:, start_idx:start_idx+65536]  # 4 * 128 * 128 = 65536 (RGBA)
            start_idx += 65536
            seg_wrist = obs['policy'][:, start_idx:start_idx+65536]  # 4 * 128 * 128 = 65536 (RGBA)
            start_idx += 65536
            ext_cam = obs['policy'][:, start_idx:start_idx+49152]  # 3 * 128 * 128 = 49152 (RGB)
            start_idx += 49152
            wrist_cam = obs['policy'][:, start_idx:start_idx+49152]  # 3 * 128 * 128 = 49152 (RGB)
            
        # Process each sample in batch
        new_ext_cam_list = []
        new_wrist_cam_list = []
        new_ext_cam_front_list = []
        
        for i in range(batch_size):
            # Process external camera
            rgb_img = ext_cam[i].reshape(3, 128, 128).permute(1, 2, 0).cpu().numpy()
            seg_rgba = seg_side[i].reshape(4, 128, 128).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            
            composite_img = replace_background_with_segmentation(
                rgb_img, seg_rgba, _bg_side_image
            ) # TODO: 255
            new_ext_cam_list.append(torch.from_numpy(composite_img.transpose(2, 0, 1).flatten()).float())
            
            # Process wrist camera
            rgb_img_wrist = wrist_cam[i].reshape(3, 128, 128).permute(1, 2, 0).cpu().numpy()
            seg_rgba_wrist = seg_wrist[i].reshape(4, 128, 128).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            
            composite_img_wrist = replace_background_with_segmentation(
                rgb_img_wrist, seg_rgba_wrist, _bg_wrist_image
            )
            new_wrist_cam_list.append(torch.from_numpy(composite_img_wrist.transpose(2, 0, 1).flatten()).float())

            # Save example images every 50 iterations, only for first sample in batch
            # if self._bg_save_counter % 50 == 0 and i == 0:
            #     save_dir = Path(self.log_dir) / "background_replacement_examples"
            #     save_dir.mkdir(exist_ok=True, parents=True)
                
            #     # Convert to uint8 for saving
            #     rgb_save = (rgb_img * 255).astype(np.uint8) if rgb_img.max() <= 1.0 else rgb_img.astype(np.uint8)
            #     composite_save = (composite_img * 255).astype(np.uint8) if composite_img.max() <= 1.0 else composite_img.astype(np.uint8)
            #     rgb_wrist_save = (rgb_img_wrist * 255).astype(np.uint8) if rgb_img_wrist.max() <= 1.0 else rgb_img_wrist.astype(np.uint8)
            #     composite_wrist_save = (composite_img_wrist * 255).astype(np.uint8) if composite_img_wrist.max() <= 1.0 else composite_img_wrist.astype(np.uint8)
                
            #     iter_num = self._bg_save_counter
            #     cv2.imwrite(str(save_dir / f"side_original_iter{iter_num:06d}.png"), cv2.cvtColor(rgb_save, cv2.COLOR_RGB2BGR))
            #     cv2.imwrite(str(save_dir / f"side_composite_iter{iter_num:06d}.png"), cv2.cvtColor(composite_save, cv2.COLOR_RGB2BGR))
            #     cv2.imwrite(str(save_dir / f"wrist_original_iter{iter_num:06d}.png"), cv2.cvtColor(rgb_wrist_save, cv2.COLOR_RGB2BGR))
            #     cv2.imwrite(str(save_dir / f"wrist_composite_iter{iter_num:06d}.png"), cv2.cvtColor(composite_wrist_save, cv2.COLOR_RGB2BGR))

            #     print(f"[INFO] Saved background replacement examples (iter {iter_num}) to: {save_dir}")
        
        _bg_save_counter += 1
        
        # Stack and move to device
        device = obs['policy'].device
        new_ext_cam_tensor = torch.stack(new_ext_cam_list).to(device)
        new_wrist_cam_tensor = torch.stack(new_wrist_cam_list).to(device)
        
        # Reconstruct observation
        if ee_pose_obs :
            obs['policy'] = torch.cat([
                arm_ee_pose,
                gripper_pos,
                arm_ee_vel,
                new_ext_cam_tensor,
                new_wrist_cam_tensor,
            ], dim=-1)
        else:
            obs['policy'] = torch.cat([
                joint_pos,
                joint_vel,
                new_ext_cam_tensor,
                new_wrist_cam_tensor,
            ], dim=-1)
        
        return obs

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")
    background_replace = True

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", args_cli.task)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        print(f"[INFO] Loading checkpoint from: {args_cli.checkpoint}")
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Resuming from checkpoint path: {resume_path}")
    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    # env_cfg.log_dir = log_dir

    # create isaac environment
    if "PPOBC" in args_cli.task:
        if "pick" in str(args_cli.task_config):
            from isaaclab_tasks.real2sim.tasks.pick.pick_r2s_env_cfg import (
                create_pick_env_cfg_from_yaml,
                get_pick_env_cfg_class_by_name,
            )
            create_env_cfg_from_yaml = create_pick_env_cfg_from_yaml
            get_env_cfg_class_by_name = get_pick_env_cfg_class_by_name
            
        elif "pour" in str(args_cli.task_config):
            from isaaclab_tasks.real2sim.tasks.pour.pour_r2s_env_cfg import (
                create_pour_env_cfg_from_yaml,
                get_pour_env_cfg_class_by_name,
            )
            create_env_cfg_from_yaml = create_pour_env_cfg_from_yaml
            get_env_cfg_class_by_name = get_pour_env_cfg_class_by_name
            
        else:
            raise ValueError(f"Unsupported task config path: {args_cli.task_config}")
        # 
        env_cfg_class = get_env_cfg_class_by_name(args_cli.task)
        
        # YAML
        env_cfg = create_env_cfg_from_yaml(
            yaml_path=args_cli.task_config,
            env_cfg_class=env_cfg_class,
            num_envs=args_cli.num_envs,
            device=args_cli.device,
            seed=agent_cfg.seed,
        )
        env_cfg.log_dir = log_dir
        
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    else:
        env =  gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    print(f"[INFO]: Log directory: {agent_cfg.class_name}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunnerViTBC":
        runner = OnPolicyRunnerViTBC(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device,bc_data_path = agent_cfg.bc_data_path)
    elif agent_cfg.class_name == "OnPolicyRunnerViT":
        runner = OnPolicyRunnerViT(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic
    
    # Force evaluation mode to ensure consistent behavior with single environment
    policy_nn.eval()
    for module in policy_nn.modules():
        if isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.BatchNorm1d):
            module.track_running_stats = False  # Disable running stats for batch norm

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt
    # reset environment
    obs = env.get_observations()
    # print(obs.shape)
    if background_replace:
        obs = replace_observation_background(obs, log_dir)
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # print('actions before unnormalize:', actions)
            actions = runner.unnormalize_actions(actions)
            # print('actions after unnormalize:', actions)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            if background_replace:
                obs = replace_observation_background(obs, log_dir)
                # obs = replace_observation_background(obs)
            # print("obs", obs)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
            if dones:
                print("*"*200)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
