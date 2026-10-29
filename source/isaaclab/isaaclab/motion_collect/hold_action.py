"""Build joint-hold actions without changing the previous gripper command."""

import torch


def make_joint_hold_action(env, robot, num_envs, *, device="cpu"):
    """Hold current joint positions and preserve each environment's raw gripper action.

    The final action is a binary gripper command, not a finger joint position.
    Keeping the action manager's last *raw* command preserves the same state
    across robots whose physical open/closed joint positions differ.
    """
    manager = env.env.action_manager
    action_dim = manager.total_action_dim
    names = list(manager.active_terms)
    dimensions = list(manager.action_term_dim)
    if (len(names) != len(dimensions) or names.count("gripper_action") != 1
            or names[-1] != "gripper_action" or dimensions[-1] != 1
            or sum(dimensions) != action_dim):
        raise RuntimeError("Expected a final one-dimensional gripper_action term")
    previous = manager.action
    joints = robot.data.joint_pos
    if (previous.ndim != 2 or previous.shape[0] < num_envs or previous.shape[1] != action_dim
            or joints.ndim != 2 or joints.shape[0] < num_envs or joints.shape[1] < action_dim):
        raise RuntimeError("Cannot preserve gripper action: inconsistent action or joint shape")
    previous_gripper = previous[:num_envs, -1]
    if not torch.isfinite(previous_gripper).all():
        raise RuntimeError("Cannot preserve a non-finite gripper action")
    action = joints[:num_envs, :action_dim].detach().to(device=device).clone()
    action[:, -1] = previous_gripper.detach().to(device=action.device, dtype=action.dtype)
    return action
