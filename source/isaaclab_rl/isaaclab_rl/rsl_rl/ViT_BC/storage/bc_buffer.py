# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import numpy as np
from tensordict import TensorDict


class BCBuffer:
    """Buffer for storing and sampling behavior cloning demonstration data."""

    def __init__(self, data_dict: dict, device: str = "cpu", ee_pose_obs: bool = False):
        """Initialize BC buffer from loaded data.
        
        Args:
            data_dict: Dictionary containing demonstration data with keys:
                - 'joint_pos': Joint positions
                - 'joint_vel': Joint velocities  
                - 'actions_obs': Previous actions
                - 'external_cam': External camera observations
                - 'wrist_cam': Wrist camera observations
                - 'actions': Expert actions
            device: Device to store tensors on
        """
        self.device = device
        # self.ee_pose_obs = False
        self.ee_pose_obs = ee_pose_obs
        
        # Convert numpy arrays to torch tensors and move to device
        if self.ee_pose_obs:
            self.arm_ee_pose = torch.from_numpy(data_dict['arm_ee_pose']).float().to(device)
            self.gripper_pos = torch.from_numpy(data_dict['gripper_pos']).float().to(device)
            self.arm_ee_vel = torch.from_numpy(data_dict['arm_ee_vel']).float().to(device)
        else:
            self.joint_pos = torch.from_numpy(data_dict['joint_pos']).float().to(device)
            self.joint_vel = torch.from_numpy(data_dict['joint_vel']).float().to(device)
        self.external_cam = torch.from_numpy(data_dict['external_cam']).float().to(device)
        self.wrist_cam = torch.from_numpy(data_dict['wrist_cam']).float().to(device)
        self.actions = torch.from_numpy(data_dict['actions']).float().to(device)
        self._post_process_actions()
        # Get total number of samples
        self.num_samples = len(self.actions)
        
        # Print shapes for debugging
        print(f"[BCBuffer] Data shapes:")
        if self.ee_pose_obs:
            print(f"  arm_ee_pose: {self.arm_ee_pose.shape}")
            print(f"  gripper_pos: {self.gripper_pos.shape}")
            print(f"  arm_ee_vel: {self.arm_ee_vel.shape}")
        else:
            print(f"  joint_pos: {self.joint_pos.shape}")
            print(f"  joint_vel: {self.joint_vel.shape}")
        print(f"  external_cam: {self.external_cam.shape}")
        print(f"  wrist_cam: {self.wrist_cam.shape}")
        
        # Initialize shuffled indices
        self.indices = torch.arange(self.num_samples, device=device)
        self.shuffle()
        
        # Track current position in buffer
        self.current_idx = 0
        
        print(f"[BCBuffer] Initialized with {self.num_samples} samples")
    
    def shuffle(self):
        """Shuffle the indices for randomized sampling."""
        self.indices = torch.randperm(self.num_samples, device=self.device)
        self.current_idx = 0
    
    def get_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Get next batch of data.
        
        Args:
            batch_size: Number of samples in the batch
            
        Returns:
            Tuple of (flattened_observations, actions) or None if no more data
        """
        # Check if we've gone through all data
        if self.current_idx >= self.num_samples:
            return None
        
        # Get batch indices
        end_idx = min(self.current_idx + batch_size, self.num_samples)
        batch_indices = self.indices[self.current_idx:end_idx]
        
        # Update current index
        self.current_idx = end_idx
        
        # Flatten all observations into a single vector
        # Order must match environment: joint_pos, joint_vel, actions_obs, semantic_seg_side, semantic_seg_wrist, external_cam, wrist_cam
        if self.ee_pose_obs:
            obs_batch = torch.cat([
                self.arm_ee_pose[batch_indices],
                self.gripper_pos[batch_indices],
                self.arm_ee_vel[batch_indices],
                self.external_cam[batch_indices],  # Already flattened (N, 49152)
                self.wrist_cam[batch_indices],  # Already flattened (N, 49152)
            ], dim=-1)
        else:   
            obs_batch = torch.cat([
                self.joint_pos[batch_indices],
                self.joint_vel[batch_indices],
                self.external_cam[batch_indices],  # Already flattened (N, 49152)
                self.wrist_cam[batch_indices],  # Already flattened (N, 49152)
            ], dim=-1)
        
        # Get actions
        actions_batch = self.actions[batch_indices]
        
        return obs_batch, actions_batch
    
    def __len__(self):
        """Return number of samples in buffer."""
        return self.num_samples
    
    def __iter__(self):
        """Make buffer iterable."""
        self.current_idx = 0
        return self
    
    def __next__(self):
        """Get next batch when iterating."""
        raise NotImplementedError("Use get_batch_generator() for iteration with specific batch size")
    
    # def _post_process_actions(self):
    #     """Symmetric Log Transform (Symlog)
    #         f(x) = sign(x) * log(|x| * C + 1)
    #     """
    #     # C = 1.0e7
    #     # M = 15
    #     # self.actions[:, :6] = torch.sign(self.actions[:, :6]) * torch.log(torch.abs(self.actions[:, :6]) * C + 1.0) / M
    #     # self.actions[:, 6:] = self.actions[:, 6:] * 0.04
    #     # Normalize each of the first 6 dimensions by its own maximum absolute value
    #     self.max_abs_before = self.actions[:, :6].abs().max(dim=0, keepdim=True).values 
    #     # self.max_abs = self.actions[:, :6].abs().max(dim=0, keepdim=True).values * 10.0 # shape: (1, 6)
    #     self.actions[:, :6] = self.actions[:, :6] / self.max_abs_before
    #     print(f"[BCBuffer] Action normalization max abs values: {self.actions[:, :6]}")
    #     self.actions[:, :6] = self.actions[:, :6] * 10.0
    #     print(f"[BCBuffer] Action scaling applied.{self.actions[:, :6]}")
    #     self.actions[:, 6:] = self.actions[:, 6:] * 0.04

    def _post_process_actions(self):
        """Symmetric Log Transform (Symlog)
            f(x) = sign(x) * log(|x| * C + 1)
        """
        # C = 1.0e7
        # M = 15
        # self.actions[:, :6] = torch.sign(self.actions[:, :6]) * torch.log(torch.abs(self.actions[:, :6]) * C + 1.0) / M
        # self.actions[:, 6:] = self.actions[:, 6:] * 0.04
        # Normalize each of the first 6 dimensions by its own maximum absolute value
        self.max_abs = self.actions[:, :6].abs().max(dim=0, keepdim=True).values * 10.0 # shape: (1, 6) #10.0
        # print(f"{self.max_abs}")
        # quit()
        # self.max_abs = self.actions[:, :6].abs().max(dim=0, keepdim=True).values
        self.actions[:, :6] = self.actions[:, :6] / self.max_abs
        self.actions[:, 6:] = self.actions[:, 6:] * 0.04
    # tensor([[0.0934, 0.1765, 0.2140, 0.2007, 0.3891, 0.6785]]
