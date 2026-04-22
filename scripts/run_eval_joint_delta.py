"""
Replay joint delta actions from a saved episode to verify correctness.

Usage:
    python scripts/run_eval_joint_delta.py \
        --npy_path /path/to/episode_0002.npy \
        --task_config /path/to/config.yaml \
        --headless
"""

import argparse
import gymnasium as gym
import torch
import numpy as np
from pathlib import Path


def main(
    npy_path: str,
    task_config: str,
    task: str = "FrankaFr3-OurGripper-Joint-VisPick-v0",
    num_envs: int = 1,
    device: str = "cuda:0",
    headless: bool = False,
):
    # ========== 启动 Isaac Sim ==========
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = headless
    args_cli.device = device
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import isaaclab_tasks

    # ========== 加载 episode 数据 ==========
    npy_path = Path(npy_path)
    print(f"[INFO] Loading episode from: {npy_path}")
    episode_data = np.load(str(npy_path), allow_pickle=True).item()

    joint_delta_actions = episode_data['obs_policy_joint_delta_actions']  # [T, 8] = 7 joint delta + 1 gripper
    recorded_qpos = episode_data['obs_policy_q_pos']                     # [T, 7]
    recorded_joint_actions = episode_data['obs_policy_joint_actions']     # [T, 8] = 7 abs joint + 1 gripper

    num_steps = len(joint_delta_actions)
    print(f"[INFO] Episode has {num_steps} steps")
    print(f"  joint_delta_actions:    {joint_delta_actions.shape}")
    print(f"  recorded_qpos:         {recorded_qpos.shape}")
    print(f"  recorded_joint_actions: {recorded_joint_actions.shape}")

    # ========== 创建环境 ==========
    task_config = Path(task_config)
    print(f"\n[INFO] Loading task config from: {task_config}")

    if "pick" in str(task_config):
        from isaaclab_tasks.real2sim.tasks.pick_r2s_env_cfg import (
            create_pick_env_cfg_from_yaml,
            get_pick_env_cfg_class_by_name,
        )
        create_env_cfg_from_yaml = create_pick_env_cfg_from_yaml
        get_env_cfg_class_by_name = get_pick_env_cfg_class_by_name
    else:
        raise ValueError(f"Cannot determine task type from config path: {task_config}")

    env_cfg_class = get_env_cfg_class_by_name(task)
    env_cfg = create_env_cfg_from_yaml(
        yaml_path=str(task_config),
        env_cfg_class=env_cfg_class,
        num_envs=num_envs,
        device=device,
        seed=42,
    )

    env = gym.make(task, cfg=env_cfg)
    obs, _ = env.reset()
    obs, _ = env.reset()

    # ========== Replay: 用 delta 重建绝对目标，通过绝对位置控制执行 ==========
    print(f"\n{'=' * 60}")
    print(f"REPLAY: joint delta -> absolute target, {num_steps} steps")
    print(f"{'=' * 60}")

    replay_qpos_list = []

    for i in range(num_steps):
        # 用录制的 qpos[t] + delta[t] 作为绝对目标，避免误差累积
        abs_target = recorded_qpos[i] + joint_delta_actions[i, :7]
        gripper = joint_delta_actions[i, 7:]
        abs_action = np.concatenate([abs_target, gripper])

        action = torch.tensor(abs_action, dtype=torch.float32).unsqueeze(0).to(device)
        obs, reward, term, trunc, info = env.step(action)

        actual_qpos = env.env.scene["robot"].data.joint_pos[0, :7].cpu().numpy()
        replay_qpos_list.append(actual_qpos)

        if i % 20 == 0 or i == num_steps - 1:
            recorded = recorded_qpos[i] if i < len(recorded_qpos) else recorded_qpos[-1]
            qpos_err = np.abs(actual_qpos - recorded).max()
            print(f"  Step {i:4d}/{num_steps}: qpos max err = {qpos_err:.6f}")

        if term or trunc:
            print(f"  [WARN] Environment terminated/truncated at step {i}")
            break

    # ========== 结果汇总 ==========
    replay_qpos = np.array(replay_qpos_list)
    min_len = min(len(replay_qpos), len(recorded_qpos))
    online_errors = np.abs(replay_qpos[:min_len] - recorded_qpos[:min_len]).max(axis=1)

    print(f"\n{'=' * 60}")
    print("REPLAY SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Steps replayed: {len(replay_qpos)}")
    print(f"  Max qpos error vs recorded: {online_errors.max():.6f}")
    print(f"  Mean qpos error vs recorded: {online_errors.mean():.6f}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay joint delta actions from saved episode")
    parser.add_argument("--npy_path", type=str, required=True, help="Path to episode .npy file")
    parser.add_argument("--task_config", type=str, required=True, help="Path to YAML task config")
    parser.add_argument("--task", type=str, default="FrankaFr3-OurGripper-Joint-VisPick-v0")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--headless", action="store_true")

    args = parser.parse_args()

    main(
        npy_path=args.npy_path,
        task_config=args.task_config,
        task=args.task,
        num_envs=args.num_envs,
        device=args.device,
        headless=args.headless,
    )
