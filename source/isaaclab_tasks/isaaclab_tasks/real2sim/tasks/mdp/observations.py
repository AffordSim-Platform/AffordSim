from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, quat_mul, quat_conjugate
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.assets import Articulation

_FRANKA_FR3_ARM_JOINTS = [
    "fr3_joint1", "fr3_joint2", "fr3_joint3",
    "fr3_joint4", "fr3_joint5", "fr3_joint6", "fr3_joint7",
]

_UR5E_ARM_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

def arm_joint_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_names: list = None,
):
    robot: Articulation = env.scene[asset_cfg.name]
    if joint_names is None:
        joint_names = _FRANKA_FR3_ARM_JOINTS
    joint_indices = [
        i for i, name in enumerate(robot.data.joint_names) if name in joint_names
    ]
    joint_pos = robot.data.joint_pos[:, joint_indices]
    return joint_pos

def arm_joint_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_names: list = None,
):
    robot: Articulation = env.scene[asset_cfg.name]
    if joint_names is None:
        joint_names = _FRANKA_FR3_ARM_JOINTS
    joint_indices = [
        i for i, name in enumerate(robot.data.joint_names) if name in joint_names
    ]
    joint_vel = robot.data.joint_vel[:, joint_indices]
    return joint_vel

def arm_ee_pose(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "fr3_hand",
):
    robot: Articulation = env.scene[asset_cfg.name]
    body_index = robot.data.body_names.index(body_name)
    # End-effector position and orientation in world frame
    ee_pos_w = robot.data.body_pos_w[:, body_index]
    ee_quat_w = robot.data.body_quat_w[:, body_index]

    # Robot base position (world frame)
    root_pos_w = robot.data.root_pos_w  # (num_envs, 3)

    # Relative position (world frame)
    ee_pos_rel = ee_pos_w - root_pos_w

    ee_pose = torch.cat([ee_pos_rel, ee_quat_w], dim=-1)
    # print("ee_pos_rel", ee_pos_rel)
    # print("*" * 100)
    # print(f"End-effector position: {ee_pos}")
    return ee_pose

def arm_ee_vel(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "fr3_hand",
):
    """Get end-effector linear and angular velocity."""
    robot: Articulation = env.scene[asset_cfg.name]
    body_index = robot.data.body_names.index(body_name)
    
    # Linear velocity (3,) and angular velocity (3,)
    ee_lin_vel = robot.data.body_lin_vel_w[:, body_index]  # Linear velocity in world frame
    ee_ang_vel = robot.data.body_ang_vel_w[:, body_index]  # Angular velocity in world frame

    # Concatenate into (6,) vector
    ee_vel = torch.cat([ee_lin_vel, ee_ang_vel], dim=-1)
    return ee_vel


def gripper_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_names_pattern: list = None,
):
    robot: Articulation = env.scene[asset_cfg.name]
    if joint_names_pattern is None:
        joint_names_pattern = ["fr3_finger_joint.*"]
    joint_indices, _ = robot.find_joints(joint_names_pattern)
    joint_pos = robot.data.joint_pos[:, joint_indices]
    return joint_pos
