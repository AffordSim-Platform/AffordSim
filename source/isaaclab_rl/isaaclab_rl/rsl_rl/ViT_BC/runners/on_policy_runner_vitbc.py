# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
import torch
import warnings
from collections import deque
from tensordict import TensorDict
import rsl_rl
from isaaclab_rl.rsl_rl.ViT_BC.algorithms import PPOBC
from rsl_rl.env import VecEnv
from isaaclab_rl.rsl_rl.ViT_BC.modules import ActorCriticViT, ActorCriticCNN
from isaaclab_rl.rsl_rl.ViT_BC.storage import BCBuffer
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, resolve_rnd_config, resolve_symmetry_config
from rsl_rl.utils import resolve_obs_groups, store_code_state
from isaaclab_tasks.real2sim.utils.EpisodeBuffer import replace_background_with_segmentation


class OnPolicyRunnerViTBC:
    """On-policy runner for training and evaluation of actor-critic methods."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu", bc_data_path: str | None = None) -> None:
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        self.bc_data_path = bc_data_path
        self.bc_buffer = None
        self.USE_BACKGROUND_REPLACE = True
        # self.ee_pose_obs = False
        self.ee_pose_obs = getattr(env.unwrapped.cfg, 'ee_pose_obs', False)
        print(f"ee_pose_obs: {self.ee_pose_obs}")
        
        # Load BC data if path provided
        print(f"BC Data Path: {bc_data_path}")
        if bc_data_path is not None:
            self._load_bc_data(bc_data_path)

        # Check if multi-GPU is enabled
        self._configure_multi_gpu()

        # Store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # Query observations from environment for algorithm construction
        obs = self.env.get_observations()
        default_sets = ["critic"]
        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], default_sets)
        self.log_dir = log_dir
        # Create the algorithm
        if self.USE_BACKGROUND_REPLACE:
            obs = self._replace_observation_background(obs)
        self.alg = self._construct_algorithm(obs)

        # Decide whether to disable logging
        # Note: We only log from the process with rank 0 (main process)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0

        # Logging
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        # Initialize writer
        self._prepare_logging_writer()

        # Randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Start learning
        obs = self.env.get_observations().to(self.device)
        
        if self.USE_BACKGROUND_REPLACE:
            obs = self._replace_observation_background(obs)
        self.train_mode()  # switch to train mode (for dropout for example)
        
        # Get BC batch size from config if available
        # self.bc_batch_size = self.alg_cfg.get('bc_batch_size', 32)

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Create buffers for logging extrinsic and intrinsic rewards
        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            # self.env.unwrapped.current_learning_iteration_for_curriculums = it
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Sample actions
                    actions = self.alg.act(obs)
                    # print(f"policy actions before unnormalize: {actions}")
                    actions = self._unnormailize_actions(actions)
                    # print(f"policy actions after unnormalize: {actions}")
                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    
                    # Apply background replacement if enabled
                    if self.USE_BACKGROUND_REPLACE:
                        obs = self._replace_observation_background(obs)
                    
                    # Process the step
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    # Extract intrinsic rewards (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None
                    # Book keeping
                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        # Update rewards
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                # Compute returns
                self.alg.compute_returns(obs)

            # Update policy (pass BC buffer directly to algorithm)zh
            loss_dict = self.alg.update(bc_buffer=self.bc_buffer)

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())
                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()
            # Save code state
            if it == start_iter and not self.disable_logs:
                # Obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # If possible store them to wandb or neptune
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        # Compute the collection size
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        # Update total time-steps and time
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # Log episode information
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # Handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # Log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f"Mean episode {key}:":>{pad}} {value:.4f}\n"""

        mean_std = self.alg.policy.action_std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        # Log losses
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])

        # Log noise std
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        # Log performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # Log training
        if len(locs["rewbuffer"]) > 0:
            # Separate logging for intrinsic and extrinsic rewards
            if hasattr(self.alg, "rnd") and self.alg.rnd:
                self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(locs["erewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(locs["irewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/weight", self.alg.rnd.weight, locs["it"])
            # Everything else
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{"#" * width}\n"""
                f"""{str.center(width, " ")}\n\n"""
                f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection: {locs["collection_time"]:.3f}s, learning {
                    locs["learn_time"]:.3f}s)\n"""
                f"""{"Mean action noise std:":>{pad}} {mean_std.item():.2f}\n"""
            )
            # Print losses
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f"Mean {key} loss:":>{pad}} {value:.4f}\n"""
            # Print rewards
            if hasattr(self.alg, "rnd") and self.alg.rnd:
                log_string += (
                    f"""{"Mean extrinsic reward:":>{pad}} {statistics.mean(locs["erewbuffer"]):.2f}\n"""
                    f"""{"Mean intrinsic reward:":>{pad}} {statistics.mean(locs["irewbuffer"]):.2f}\n"""
                )
            log_string += f"""{"Mean reward:":>{pad}} {statistics.mean(locs["rewbuffer"]):.2f}\n"""
            # Print episode information
            log_string += f"""{"Mean episode length:":>{pad}} {statistics.mean(locs["lenbuffer"]):.2f}\n"""
        else:
            log_string = (
                f"""{"#" * width}\n"""
                f"""{str.center(width, " ")}\n\n"""
                f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection: {locs["collection_time"]:.3f}s, learning {
                    locs["learn_time"]:.3f}s)\n"""
                f"""{"Mean action noise std:":>{pad}} {mean_std.item():.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""

        log_string += ep_string
        log_string += (
            f"""{"-" * width}\n"""
            f"""{"Total timesteps:":>{pad}} {self.tot_timesteps}\n"""
            f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
            f"""{"Time elapsed:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{"ETA:":>{pad}} {
                time.strftime(
                    "%H:%M:%S",
                    time.gmtime(
                        self.tot_time
                        / (locs["it"] - locs["start_iter"] + 1)
                        * (locs["start_iter"] + locs["num_learning_iterations"] - locs["it"])
                    ),
                )
            }\n"""
        )
        print(log_string)

    def save(self, path: str, infos: dict | None = None) -> None:
        # Save model
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        # Save RND model if used
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        
        # Print normalizer statistics for debugging
        if hasattr(self.alg.policy, 'actor_obs_normalizer') and hasattr(self.alg.policy.actor_obs_normalizer, 'running_mean'):
            print(f"[DEBUG] Saving actor normalizer - mean: {self.alg.policy.actor_obs_normalizer.running_mean.mean().item():.4f}, "
                  f"var: {self.alg.policy.actor_obs_normalizer.running_var.mean().item():.4f}")
        
        torch.save(saved_dict, path)

        # Upload model to external logging service
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        # Load model (this returns a _IncompatibleKeys object, not a boolean)
        missing_keys = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"], strict=False)
        
        # Print information about loaded normalizer states
        if hasattr(self.alg.policy, 'actor_obs_normalizer'):
            print(f"[INFO] Actor obs normalizer loaded from checkpoint")
            if hasattr(self.alg.policy.actor_obs_normalizer, 'running_mean'):
                print(f"[DEBUG] Loaded actor normalizer - mean: {self.alg.policy.actor_obs_normalizer.running_mean.mean().item():.4f}, "
                      f"var: {self.alg.policy.actor_obs_normalizer.running_var.mean().item():.4f}, "
                      f"count: {self.alg.policy.actor_obs_normalizer.count.item()}")
        if hasattr(self.alg.policy, 'critic_obs_normalizer'):
            print(f"[INFO] Critic obs normalizer loaded from checkpoint")
        
        # Load RND model if used
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        
        # Load optimizer (always load if resuming training)
        if load_optimizer:
            # Algorithm optimizer
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            print(f"[INFO] Optimizer state loaded - learning rate: {self.alg.optimizer.param_groups[0]['lr']:.2e}")
            # RND optimizer if used
            if hasattr(self.alg, "rnd") and self.alg.rnd:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        
        # Load current learning iteration (always update when loading a checkpoint)
        self.current_learning_iteration = loaded_dict["iter"]
        print(f"[INFO] Resumed from iteration {self.current_learning_iteration}")
        
        return loaded_dict["infos"]
    # def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
    #     loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
    #     # Load model
    #     resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
    #     # Load RND model if used
    #     if hasattr(self.alg, "rnd") and self.alg.rnd:
    #         self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
    #     # Load optimizer if used
    #     if load_optimizer and resumed_training:
    #         # Algorithm optimizer
    #         self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
    #         # RND optimizer if used
    #         if hasattr(self.alg, "rnd") and self.alg.rnd:
    #             self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
    #     # Load current learning iteration
    #     if resumed_training:
    #         self.current_learning_iteration = loaded_dict["iter"]
    #     return loaded_dict["infos"]

    def get_inference_policy(self, device: str | None = None) -> callable:
        self.eval_mode()  # Switch to evaluation mode (e.g. for dropout)
        if device is not None:
            self.alg.policy.to(device)
        return self.alg.policy.act_inference

    def train_mode(self) -> None:
        # PPO
        self.alg.policy.train()
        # RND
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.alg.rnd.train()

    def eval_mode(self) -> None:
        # PPO
        self.alg.policy.eval()
        # RND
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.alg.rnd.eval()

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        self.git_status_repos.append(repo_file_path)

    def _configure_multi_gpu(self) -> None:
        """Configure multi-gpu training."""
        # Check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # If not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        # Get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # Make a configuration dictionary
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,  # Rank of the main process
            "local_rank": self.gpu_local_rank,  # Rank of the current process
            "world_size": self.gpu_world_size,  # Total number of processes
        }

        # Check if user has device specified for local rank
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        # Validate multi-gpu configuration
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        # Initialize torch distributed
        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        # Set device to the local rank
        torch.cuda.set_device(self.gpu_local_rank)

    def _construct_algorithm(self, obs: TensorDict) -> PPOBC:
        """Construct the actor-critic algorithm."""
        # Resolve RND config
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)

        # Resolve symmetry config
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        # Resolve deprecated normalization config
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and "
                "`critic_obs_normalization` as part of the `policy` configuration instead.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        # Initialize the policy
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))
        actor_critic: ActorCriticCNN | ActorCriticViT | ActorCritic | ActorCriticRecurrent = actor_critic_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        alg_class = eval(self.alg_cfg.pop("class_name"))
        alg: PPOBC = alg_class(actor_critic, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg)

        # Initialize the storage
        alg.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )

        return alg

    def _prepare_logging_writer(self) -> None:
        """Prepare the logging writers."""
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            # Launch either Tensorboard or Neptune or Tensorboard summary writer, default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")
    
    def _load_bc_data(self, data_path: str) -> None:
        """Load BC demonstration data from directory of trajectory files and create BCBuffer."""
        import numpy as np
        import os
        from pathlib import Path
        
        print(f"[INFO] Loading BC data from: {data_path}")
        
        # Check if data_path is a directory or a single file
        if os.path.isdir(data_path):
            # Load all trajectory files from directory
            trajectory_files = sorted(Path(data_path).glob('*.npy'))
            print(f"[INFO] Found {len(trajectory_files)} trajectory files")
            
            # Lists to store all state-action pairs from all trajectories
            if self.ee_pose_obs :
                all_arm_ee_pose = []
                all_gripper_pos = []
                all_arm_ee_vel = []
            else:
                all_joint_pos = []
                all_joint_vel = []
            # all_actions_obs = []
            # all_semantic_seg_side = []
            # all_semantic_seg_wrist = []
            
            all_external_cam = []
            all_wrist_cam = []
            all_actions = []
            
            # Load each trajectory and convert to state-action pairs
            for traj_file in trajectory_files:
                print(f"[INFO] Loading trajectory: {traj_file.name}")
                traj_data = np.load(traj_file, allow_pickle=True).item()
                
                # Extract trajectory data
                if self.ee_pose_obs :
                    arm_ee_pose = traj_data['obs_policy_arm_ee_pose']
                    gripper_pos = traj_data['obs_policy_gripper_pos']
                    arm_ee_vel = traj_data['obs_policy_arm_ee_vel']
                else:
                    joint_pos = traj_data['obs_policy_joint_pos']
                    joint_vel = traj_data['obs_policy_joint_vel']
                # actions_obs = traj_data['obs_policy_actions']
                # semantic_seg_side = traj_data['obs_policy_semantic_segmentation_side']
                # semantic_seg_wrist = traj_data['obs_policy_semantic_segmentation_wrist']
                external_cam = traj_data['obs_policy_external_cam_side']
                wrist_cam = traj_data['obs_policy_wrist_cam']
                actions = traj_data['actions']
                
                # Get trajectory length
                traj_len = len(actions)
                
                # Convert trajectory to state-action pairs
                # Each timestep becomes one (state, action) pair
                if self.ee_pose_obs :
                    for t in range(traj_len):
                        all_arm_ee_pose.append(arm_ee_pose[t])
                        all_gripper_pos.append(gripper_pos[t])
                        all_arm_ee_vel.append(arm_ee_vel[t])
                        all_external_cam.append(external_cam[t])
                        all_wrist_cam.append(wrist_cam[t])
                        all_actions.append(actions[t])
                else:
                    for t in range(traj_len):
                        all_joint_pos.append(joint_pos[t])
                        all_joint_vel.append(joint_vel[t])
                        all_external_cam.append(external_cam[t])
                        all_wrist_cam.append(wrist_cam[t])
                        all_actions.append(actions[t])
            
            # Convert lists to numpy arrays
            if self.ee_pose_obs :
                bc_data_dict = {
                    'arm_ee_pose':  np.array(all_arm_ee_pose),
                    'gripper_pos': np.array(all_gripper_pos),
                    'arm_ee_vel':  np.array(all_arm_ee_vel),
                    'external_cam': np.array(all_external_cam),
                    'wrist_cam': np.array(all_wrist_cam),
                    'actions': np.array(all_actions),
                }
            else:
                bc_data_dict = {
                    'joint_pos': np.array(all_joint_pos),
                    'joint_vel': np.array(all_joint_vel),
                    'external_cam': np.array(all_external_cam),
                    'wrist_cam': np.array(all_wrist_cam),
                    'actions': np.array(all_actions),
                }
            
            total_samples = len(all_actions)
            print(f"[INFO] Converted {len(trajectory_files)} trajectories into {total_samples} state-action pairs")
            
        else:
            # Single file mode (for backward compatibility)
            print(f"[INFO] Loading single trajectory file")
            data = np.load(data_path, allow_pickle=True)
            
            # Extract required observations and actions
            if self.ee_pose_obs :
                bc_data_dict = {
                    'arm_ee_pose': data['obs_policy_arm_ee_pose'],
                    'gripper_pos': data['obs_policy_gripper_pos'],
                    'arm_ee_vel': data['obs_policy_arm_ee_vel'],
                    'external_cam': data['obs_policy_external_cam_front'],
                    'wrist_cam': data['obs_policy_wrist_cam'],
                    'actions': data['actions'],
                }
            else:
                bc_data_dict = {
                    'joint_pos': data['obs_policy_joint_pos'],
                    'joint_vel': data['obs_policy_joint_vel'],
                    'external_cam': data['obs_policy_external_cam_front'],
                    'wrist_cam': data['obs_policy_wrist_cam'],
                    'actions': data['actions'],
                }
        
        # Create BCBuffer with all state-action pairs
        self.bc_buffer = BCBuffer(bc_data_dict, device=self.device, ee_pose_obs=self.ee_pose_obs)
    
    def _replace_observation_background(self, obs: TensorDict) -> TensorDict:
        """Replace background in RGB observations using semantic segmentation
        
        Args:
            obs: Observation TensorDict from environment
            
        Returns:
            Modified observation with background replaced in RGB images
        """
        import numpy as np
        import cv2
        from pathlib import Path
        
        # Load background images if not already loaded
        if not hasattr(self, '_bg_side_image'):
            bg_side_path = "/home/hk/limingyang/code/IsaacLab/backgrounds/rightsidebackground_realworld_128x128.png"
            bg_img = cv2.imread(bg_side_path)
            self._bg_side_image = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
            
        if not hasattr(self, '_bg_wrist_image'):
            bg_wrist_path = "/home/hk/limingyang/code/IsaacLab/backgrounds/wristbackground_realworld.jpg"
            bg_img = cv2.imread(bg_wrist_path)
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
            
            # Crop bottom 1/3 and resize back to original size
            h, w = bg_img.shape[:2]
            crop_height = int(h * 2 / 3)  # Keep top 2/3
            bg_img_cropped = bg_img[:crop_height, :]
            self._bg_wrist_image = cv2.resize(bg_img_cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Initialize counter for saving example images
        if not hasattr(self, '_bg_save_counter'):
            self._bg_save_counter = 0
        
        # Process policy observations
        if self.ee_pose_obs :
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
                rgb_img, seg_rgba, self._bg_side_image
            ) # TODO: 输出的图像已经乘以255
            new_ext_cam_list.append(torch.from_numpy(composite_img.transpose(2, 0, 1).flatten()).float())
            
            # Process wrist camera
            rgb_img_wrist = wrist_cam[i].reshape(3, 128, 128).permute(1, 2, 0).cpu().numpy()
            seg_rgba_wrist = seg_wrist[i].reshape(4, 128, 128).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            
            composite_img_wrist = replace_background_with_segmentation(
                rgb_img_wrist, seg_rgba_wrist, self._bg_wrist_image
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
        
        self._bg_save_counter += 1
        
        # Stack and move to device
        new_ext_cam_tensor = torch.stack(new_ext_cam_list).to(self.device)
        new_wrist_cam_tensor = torch.stack(new_wrist_cam_list).to(self.device)
        
        # Reconstruct observation
        if self.ee_pose_obs :
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

    def _unnormailize_actions(self, actions: torch.Tensor) -> torch.Tensor:
            # actions_clamped = torch.clamp(actions, -1.0, 1.0)
            # return torch.concat([actions_clamped[:, :6] * self.bc_buffer.max_abs, actions[:, 6:]], dim=-1)
            return torch.concat([actions[:, :6] * self.bc_buffer.max_abs, actions[:, 6:]], dim=-1)
    
    def unnormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
            # actions_clamped = torch.clamp(actions, -1.0, 1.0)
            # return torch.concat([actions_clamped[:, :6] * self.bc_buffer.max_abs, actions[:, 6:]], dim=-1)
            return torch.concat([actions[:, :6] * self.bc_buffer.max_abs, actions[:, 6:]], dim=-1)
        