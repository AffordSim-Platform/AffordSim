# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms, quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, 
    minimal_height: float, 
    maximum_height: float, 
    object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height.
    The reward increases linearly with height within the valid range [minimal_height, maximum_height].
    """
    object: RigidObject = env.scene[object_cfg.name]
    minimal_height = env.object_init_z + minimal_height
    # print("minimal_height", minimal_height)
    object_z = object.data.root_pos_w[:, 2]

    
    # print("object_z", object_z)
    # print("*"*20)
    # print("delta_z", object_z - minimal_height)
    # print("*"*20)
    condition = (object_z > minimal_height) & (object_z < maximum_height)
    reward = torch.where(condition, 1.0, 0.0)
    # print(f"object_is_lifted reward: {reward}")
    # reward = reward * (1.0 - arm_joint_limits_reached(env))
    return reward


# def object_ee_distance(
#     env: ManagerBasedRLEnv,
#     std: float,
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
#     ee_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# )  to  torch.Tensor:
#     """Reward the agent for reaching the object using tanh-kernel."""
#     # extract the used quantities (to enable type-hinting)
#     object: RigidObject = env.scene[object_cfg.name]
#     cube_pos_w = object.data.root_pos_w
#     robot: Articulation = env.scene[ee_cfg.name]
#     body_index = robot.data.body_names.index("fr3_hand")
#     ee_pos = robot.data.body_pos_w[:, body_index]
#     object_ee_distance = torch.norm(cube_pos_w - ee_pos, dim=1)

#     return 1 - torch.tanh(object_ee_distance / std)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # print(f"object_init_x: {ee_w[0, 0]:.4f}, actual: {cube_pos_w[0, 0]:.4f}")
    # print(f"object_init_y: {ee_w[0, 1]:.4f}, actual: {cube_pos_w[0, 1]:.4f}")
    # print(f"object_init_z: {ee_w[0, 2]:.4f}, actual: {cube_pos_w[0, 2]:.4f}")
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
    # print(f"object_ee_distance: {object_ee_distance}")
    reward = 1 - torch.tanh(object_ee_distance.abs() / std)
    # reward = reward * (1.0 - arm_joint_limits_reached(env))
    return reward


# def object_goal_distance(
#     env: ManagerBasedRLEnv,
#     std: float,
#     minimal_height: float,
#     command_name: str,
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
# )  to  torch.Tensor:
#     """Reward the agent for tracking the goal pose using tanh-kernel."""
#     # extract the used quantities (to enable type-hinting)
#     robot: RigidObject = env.scene[robot_cfg.name]
#     object: RigidObject = env.scene[object_cfg.name]
#     command = env.command_manager.get_command(command_name)
#     des_pos_b = command[:, :3]
#     des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
#     distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
#     return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


# def lift_and_drop_reward(
#     env: ManagerBasedRLEnv,
#     target_height: float,
#     threshold: float = 0.01,
#     # asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# )  to  torch.Tensor:
#     object: RigidObject = env.scene[object_cfg.name]
#     asset: Articulation = env.scene[robot_cfg.name]
#     obj_z = object.data.root_pos_w[:, 2]
#     gripper_pos = asset.data.joint_pos[:, robot_cfg.joint_ids]
#     gripper_open_command = (torch.abs(gripper_pos) <= threshold)
#     is_high_enough = obj_z > target_height
#     reward = torch.where(is_high_enough & gripper_open_command.squeeze(), 1.0, 0.0)
#     # reward = reward * (1.0 - arm_joint_limits_reached(env, asset_cfg=robot_cfg))
#     return reward


# def object_is_lifted(
#     env: ManagerBasedRLEnv, 
#     minimal_height: float, 
#     maximum_height: float, 
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object")
# )  to  torch.Tensor:
#     """Reward the agent for lifting the object above the minimal height.
#     The reward increases linearly with height within the valid range [minimal_height, maximum_height].
#     """
#     object: RigidObject = env.scene[object_cfg.name]
#     object_z = object.data.root_pos_w[:, 2] - 0.0483 # subtract object initial height
    
#     # Compute normalized height within valid range [0, 1]
#     height_range = maximum_height - minimal_height
#     normalized_height = (object_z - minimal_height) / height_range
    
#     # Clamp to [0, 1] range, then multiply by max reward 1.0
#     clamped_reward = torch.clamp(normalized_height, 0.0, 1.0)
    
#     return clamped_reward


def object_stable_at_height(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    maximum_height: float,
    velocity_threshold: float,
    command_name: str,
    reward_value: float = 10.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
 
    object: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    # Check if object is in the target height range
    object_x = object.data.root_pos_w[:, 0]
    object_y = object.data.root_pos_w[:, 1]
    object_z = object.data.root_pos_w[:, 2]
    
    # Use saved initial positions (relative to robot base) to compute target positions
    # instead of getting from command_manager (to avoid positions being updated)
    des_pos_w_x = robot.data.root_pos_w[:, 0] + env.object_init_x
    des_pos_w_y = robot.data.root_pos_w[:, 1] + env.object_init_y

    # object_init_pos = torch.stack([des_pos_w_x, des_pos_w_y, env.object_init_z], dim=1)
    # print("object_init_pos:", object_init_pos)
    # object_actual_pos = torch.stack([object_x, object_y, object_z], dim=1)
    # print("object actual pos:", object_actual_pos)

    height_condition = (object_z >= minimal_height) & (object_z <= maximum_height) & (torch.abs(object_x - des_pos_w_x) < 0.05) & (torch.abs(object_y - des_pos_w_y) < 0.05)
    # Check if average joint velocity is below threshold
    joint_vel = robot.data.joint_vel[:, robot_cfg.joint_ids]
    avg_joint_vel = torch.mean(torch.abs(joint_vel), dim=1)
    velocity_condition = avg_joint_vel < velocity_threshold

    velocity_reward = 1.0 - torch.tanh(avg_joint_vel / velocity_threshold)
    # Give reward only when height condition is met, scaled by velocity
    reward = torch.where(height_condition, reward_value * velocity_reward, 0.0)
    
    return reward


def ee_orientation_penalty(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
   
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    
    # Get end-effector orientation quaternion: (num_envs, 4)
    ee_quat = ee_frame.data.target_quat_w[..., 0, :]
    # print("ee_quat", ee_quat)
    
    # Rotate end-effector local Z axis (0, 0, 1) to world frame
    # to get the direction of end-effector Z axis in world frame
    local_z = torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1)
    ee_z_axis_w = quat_apply_inverse(ee_quat, local_z)
    # print("ee_z_axis_w", ee_z_axis_w)
    
    # Downward direction vector in world frame (0, 0, -1)
    down_vector_w = torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(env.num_envs, 1)
    
    # Compute dot product between EE Z-axis and downward direction
    # When EE points straight down, ee_z_axis_w ~ (0, 0, -1), dot product ~ 1
    # Dot product range [-1, 1]: 1 = fully aligned (pointing down), -1 = fully opposite (pointing up)
    alignment = torch.sum(ee_z_axis_w * down_vector_w, dim=1)
    
    # Compute error: alignment = 1 (fully down)  to  error = 0
    # alignment = 0 (horizontal)  to  error = 1
    # alignment = -1 (fully up)  to  error = 2
    error = 1.0 - alignment
    
    # Use exponential kernel to compute reward; smaller error means higher reward
    reward = torch.exp(-error / std)
    # print("ee_orientation_penalty reward", reward)
    # reward = reward * (1.0 - arm_joint_limits_reached(env, asset_cfg=robot_cfg))
    
    return reward


def arm_joint_limits_reached(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Check if arm joints (excluding gripper) have reached their limits.

    Returns a boolean tensor for each environment:
    - True: any non-gripper joint has reached its soft limit
    - False: all non-gripper joints are within limits

    Args:
        env: Environment instance.
        asset_cfg: Robot configuration, default is "robot".

    Returns:
        Boolean tensor of shape (num_envs,), indicating whether any joint reached its limit.
    """
    # Get robot asset
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get all joint positions and limits
    # asset_cfg.joint_ids may be a slice object, use it directly for indexing
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, :]
    
    # Exclude last two joints (gripper joints)
    # Only check arm joints
    if joint_pos.shape[1] > 2:
        arm_joint_pos = joint_pos[:, :-2]
        arm_joint_limits = joint_limits[:, :-2, :]
    else:
        arm_joint_pos = joint_pos
        arm_joint_limits = joint_limits
    
    # Compute out-of-limit violations
    # Lower limit violation: joint_pos < lower_limit
    lower_violations = -(arm_joint_pos - arm_joint_limits[:, :, 0]).clip(max=0.0)
    
    # Upper limit violation: joint_pos > upper_limit
    upper_violations = (arm_joint_pos - arm_joint_limits[:, :, 1]).clip(min=0.0)
    
    # Total violation amount
    out_of_limits = lower_violations + upper_violations
    
    # Check if any joint exceeds limits in each environment
    # If any joint's violation > 0, consider the limit reached
    any_joint_violated = torch.any(out_of_limits > 0.0, dim=1)
    
    return any_joint_violated.float()  # Convert to float for use as reward/termination

def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    
    # Use saved initial positions (relative to robot base) to compute target positions
    # instead of getting from command_manager (to avoid positions being updated)
    des_pos_w = torch.stack([
        robot.data.root_pos_w[:, 0] + env.object_init_x,
        robot.data.root_pos_w[:, 1] + env.object_init_y,
        torch.full((env.num_envs,), 0.3, device=env.device)  # Target height
    ], dim=1)
    
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)

    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))

