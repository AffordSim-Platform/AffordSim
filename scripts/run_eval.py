"""
run the evaluation script:
python run_eval.py --episodes 10 --headless --task_config path/to/config.yaml
"""

import argparse
import gymnasium as gym
import torch
import numpy as np
import cv2
import os
from datetime import datetime
from pathlib import Path
from tqdm import tqdm


def main(
        episodes: int = 10,
        headless: bool = False,
        task_config: str = None,
        task: str = "FrankaFr3-OurGripper-IKRel-PPOBC-VisPour-v0",
        num_envs: int = 1,
        device: str = "cuda:0",
        ):
    
    # ========== 启动 Isaac Sim ==========
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description="Evaluation script for pouring task.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = headless
    args_cli.device = device
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    
    # ========== 导入任务相关模块 ==========
    import isaaclab_tasks
    
    # ========== 根据 task_config 路径确定任务类型 ==========
    if task_config is None:
        # 默认使用倒豆子任务配置
        task_config = Path("/home/hk/limingyang/code/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/pour/config/sceneconfig/pour_beans.yaml")
    else:
        task_config = Path(task_config)
    
    print(f"[INFO] Loading task config from: {task_config}")
    
    # 根据配置文件路径判断任务类型
    if "pick" in str(task_config):
        from isaaclab_tasks.real2sim.tasks.pick.pick_r2s_env_cfg import (
            create_pick_env_cfg_from_yaml,
            get_pick_env_cfg_class_by_name,
        )
        create_env_cfg_from_yaml = create_pick_env_cfg_from_yaml
        get_env_cfg_class_by_name = get_pick_env_cfg_class_by_name
        
    elif "pour" in str(task_config):
        from isaaclab_tasks.real2sim.tasks.pour.pour_r2s_env_cfg import (
            create_pour_env_cfg_from_yaml,
            get_pour_env_cfg_class_by_name,
        )
        create_env_cfg_from_yaml = create_pour_env_cfg_from_yaml
        get_env_cfg_class_by_name = get_pour_env_cfg_class_by_name
        
    else:
        raise ValueError(f"Cannot determine task type from config path: {task_config}")
    
    
    # ========== 从 YAML 创建环境配置 ==========
    print(f"[INFO] Creating environment: {task}")
    
    # 根据任务名称获取环境配置类
    env_cfg_class = get_env_cfg_class_by_name(task)
    
    # 从YAML创建环境配置
    env_cfg = create_env_cfg_from_yaml(
        yaml_path=str(task_config),
        env_cfg_class=env_cfg_class,
        num_envs=num_envs,
        device=device,
        seed=42,
    )
    
    # 打印环境配置信息
    print(f"[INFO] Environment configuration loaded:")
    print(f"  - Task: {task}")
    print(f"  - Num envs: {env_cfg.scene.num_envs}")
    print(f"  - Device: {env_cfg.sim.device}")
    print(f"  - Episode length: {env_cfg.episode_length_s}s")
    
    # ========== 创建环境 ==========
    env = gym.make(task, cfg=env_cfg)
    obs, _ = env.reset()
    obs, _ = env.reset()  # 重置两次（你的原始代码有这个，可能是为了确保初始化）
    
    # ========== 定义测试动作序列 ==========
    # 你可以根据需要选择不同的动作序列
    
    # 选项1: Joint Position Actions
    PourBeansTrajectory = torch.tensor([
    [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04],      # 0. Home位置
    [0.3, -0.5, 0.0, -2.2, 0.0, 1.9, 0.785, 0.04],            # 1. 移动到杯子上方
    [0.5, -0.3, 0.0, -2.0, 0.0, 2.0, 0.785, 0.04],            # 2. 接近杯子
    [0.55, -0.1, 0.0, -1.8, 0.0, 2.1, 0.785, 0.04],           # 3. 对准杯子把手（y=-0.10）
    [0.55, -0.1, 0.0, -1.8, 0.0, 2.1, 0.785, 0.0],            # 4. 夹紧杯子（gripper关闭）
    [0.55, -0.1, 0.0, -2.0, 0.0, 2.0, 0.785, 0.0],            # 5. 抬起杯子
    [0.55, 0.0, 0.0, -2.1, 0.0, 1.95, 0.785, 0.0],            # 6. 向碗方向移动（中间点）
    [0.6, 0.15, 0.0, -2.0, 0.0, 1.9, 0.785, 0.0],             # 7. 到达碗上方（x=0.6, y=0.15）
    [0.6, 0.15, 0.0, -1.85, 0.0, 2.0, 0.785, 0.0],            # 8. 调整高度到合适倾倒位置
    [0.6, 0.15, 0.3, -1.85, 0.0, 2.0, 0.785, 0.0],            # 9. 开始倾倒（joint3旋转）
    [0.6, 0.15, 0.6, -1.85, 0.0, 2.0, 0.785, 0.0],            # 10. 继续倾倒
    [0.6, 0.15, 0.9, -1.85, 0.0, 2.0, 0.785, 0.0],            # 11. 倾倒到位（约52度）
    [0.6, 0.15, 0.9, -1.85, 0.0, 2.0, 0.785, 0.0],            # 12. 保持倾倒姿态
    [0.6, 0.15, 0.3, -1.85, 0.0, 2.0, 0.785, 0.0],            # 13. 恢复直立
    [0.6, 0.15, 0.0, -1.85, 0.0, 2.0, 0.785, 0.04],           # 14. 放下杯子（gripper打开）
    [0.6, 0.15, 0.0, -2.0, 0.0, 1.9, 0.785, 0.04],            # 15. 抬起手臂
    [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04]       # 16. 回Home
])
    
    # 选项2: Differential IK Actions
    DifferentialIKPourTrajectory = torch.tensor([
    # [x, y, z, qx, qy, qz, qw, gripper]
    # 杯子: [0.45, 0.0, 0.0213], 碗: [0.5, 0.20, 0.0106]
    # 修改：先绕x轴旋转-90度（朝下），再绕z轴旋转90度
    # [0.45, 0.0, 0.15, 0 , 1, 0,0, 0],
    # [0.45, 0.0, 0.15, 0 , 1, 0,0, 0],
    # [0.45, 0.0, 0.15, 0 , 1, 0,0, 0],
    # [0.45, 0.0, 0.05, 0 , 1, 0,0, 0],
    # [0.45, 0.0, 0.05, 0 , 1, 0,0, -1],
    # [0.45, 0.0, 0.05, 0 , 1, 0,0, -1],
    # [0.45, 0.0, 0.05, 0 , 1, 0,0, -1],
    # [0.45, 0.0, 0.05, 0 , 1, 0,0, -1],
    # [0.50, 0.20, 0.20, -0.271, 0.653, 0.653, 0.271, -1],   # 10. 开始倾倒（45度）
    # [0.50, 0.20, 0.20, -0.092, 0.701, 0.701, 0.092, -1],   # 11. 继续倾倒（70度）
    # [0.50, 0.20, 0.20, 0.0, 0.707, 0.707, 0.0, -1],        # 12. 倾倒到位（90度）
    # [0.50, 0.20, 0.20, 0.0, 0.707, 0.707, 0.0, -1],        # 13. 保持倾倒
    # [0.50, 0.20, 0.20, -0.5, 0.5, 0.5, 0.5, -1],           # 14. 恢复直立
    # [0.50, 0.20, 0.08, -0.5, 0.5, 0.5, 0.5, 0],           # 15. 放下杯子
    # [0.50, 0.20, 0.25, -0.5, 0.5, 0.5, 0.5, 0],           # 16. 抬起手臂
    # [0.30, 0.00, 0.30, -0.5, 0.5, 0.5, 0.5, 0] 
    [0.30, 0.00, 0.20, -0.5, 0.5, 0.5, 0.5, 0],           # 0. Home上方
    [0.45, -0.15, 0.15, -0.5, 0.5, 0.5, 0.5, 0],          # 1. 移动到杯子侧面（y=-0.15）
    [0.45, -0.10, 0.08, -0.5, 0.5, 0.5, 0.5, 0],          # 2. 降低高度，靠近
    [0.45, -0.05, 0.05, -0.5, 0.5, 0.5, 0.5, 0],          # 3. 继续靠近
    [0.45, 0.05, 0.05, -0.5, 0.5, 0.5, 0.5, 0],           # 4. 对准杯子
    [0.45, 0.05, 0.05, -0.5, 0.5, 0.5, 0.5, -1],           # 5. 夹紧杯身
    [0.45, 0.00, 0.20, -0.5, 0.5, 0.5, 0.5, -1],           # 6. 抬起杯子
    [0.475, 0.10, 0.25, -0.5, 0.5, 0.5, 0.5, -1],          # 7. 向碗移动（中间点）
    [0.50, 0.20, 0.25, -0.5, 0.5, 0.5, 0.5, -1],           # 8. 到达碗上方
    [0.50, 0.20, 0.20, -0.5, 0.5, 0.5, 0.5, -1],           # 9. 降低到倾倒高度
    [0.50, 0.20, 0.20, -0.271, 0.653, 0.653, 0.271, -1],   # 10. 开始倾倒（45度）
    [0.50, 0.20, 0.20, -0.092, 0.701, 0.701, 0.092, -1],   # 11. 继续倾倒（70度）
    [0.50, 0.20, 0.20, 0.0, 0.707, 0.707, 0.0, -1],        # 12. 倾倒到位（90度）
    [0.50, 0.20, 0.20, 0.0, 0.707, 0.707, 0.0, -1],        # 13. 保持倾倒
    [0.50, 0.20, 0.20, -0.5, 0.5, 0.5, 0.5, -1],           # 14. 恢复直立
    [0.50, 0.20, 0.08, -0.5, 0.5, 0.5, 0.5, 0],           # 15. 放下杯子
    [0.50, 0.20, 0.25, -0.5, 0.5, 0.5, 0.5, 0],           # 16. 抬起手臂
    [0.30, 0.00, 0.30, -0.5, 0.5, 0.5, 0.5, 0]  
])
    
    # 选项3: 从录制的demonstration加载
    demo_path = Path("/data/limingyang_data/r2s2r_data/pick up cube/20251225_223314/episode_0000.npy")
    demo_path = None
    
    if demo_path is not None and demo_path.exists():
        print(f"[INFO] Loading demonstration from: {demo_path}")
        episode_data = np.load(str(demo_path), allow_pickle=True).item()
        actions_test = torch.tensor(episode_data['actions'], dtype=torch.float32)
        print(f"[INFO] Loaded {len(actions_test)} actions from demonstration")
    else:
        if demo_path is not None:
            print(f"[WARNING] Demo file not found: {demo_path}")
        print("[INFO] Using default IK actions")
        actions_test = DifferentialIKPourTrajectory
    
    # ========== 设置保存目录 ==========
    video_dir = Path("runs") / datetime.now().strftime("%Y-%m-%d") / datetime.now().strftime("%H-%M-%S")
    video_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Saving results to: {video_dir}")
    
    # 加载真实世界背景图像（如果需要）
    background_path = Path("/home/hk/limingyang/code/IsaacLab/runs/2025-12-16/00-07-39/background_realworld.jpg")
    if background_path.exists():
        background_img = cv2.imread(str(background_path))
        background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)
        background_img = cv2.resize(background_img, (128, 128))
        print(f"[INFO] Loaded background image from: {background_path}")
    else:
        print(f"[WARNING] Background image not found at {background_path}")
        background_img = None
    
    # ========== 运行评估 ==========
    max_steps = env.env.max_episode_length
    print(f"[INFO] Starting evaluation: {episodes} episodes, max {max_steps} steps each")
    
    for ep in tqdm(range(episodes), desc="Episodes"):
        obs, _ = env.reset()
        
        episode_rewards = []
        episode_success = False
        
        for i in range(max_steps):
            # 获取当前动作
            action = actions_test[i % len(actions_test)].unsqueeze(0)
            
            # 如果动作序列播放完了，重置环境
            if i % len(actions_test) == 0 and i != 0:
                obs, _ = env.reset()
            
            # 打印调试信息（可选）
            if i % 10 == 0:
                ee_pos = env.env.scene["robot"].data.body_pos_w[0, 9]
                ee_quat = env.env.scene["robot"].data.body_quat_w[0, 9]
                formatted_pos = ", ".join(f"{float(x):.4f}" for x in ee_pos)
                formatted_quat = ", ".join(f"{float(x):.4f}" for x in ee_quat)
                print(f"\nEp {ep+1}, Step {i}")
                print(f"  EE pos: [{formatted_pos}]")
                print(f"  EE quat: [{formatted_quat}]")
                print(f"  Action: {action[0].cpu().numpy()}")
            
            # 执行动作
            obs, reward, term, trunc, info = env.step(action)
            episode_rewards.append(reward)
            
            # 保存图像（如果需要）
            if background_img is not None and i % 10 == 0:
                save_images_with_background(
                    obs, background_img, video_dir, ep, i
                )
            
            # 检查终止条件
            if term or trunc:
                # episode_success = term  # 假设term表示成功
                # print(f"\nEpisode {ep+1} finished at step {i}")
                # print(f"  Success: {episode_success}")
                # print(f"  Total reward: {sum(episode_rewards):.2f}")
                break
        
        # 保存episode统计信息
        episode_stats = {
            "episode": ep,
            "steps": i + 1,
            "total_reward": sum(episode_rewards),
            "success": episode_success,
        }
        
        # 保存到文件
        stats_path = video_dir / f"episode_{ep:04d}_stats.txt"
        with open(stats_path, 'w') as f:
            for key, value in episode_stats.items():
                f.write(f"{key}: {value}\n")
    
    print(f"\n[INFO] Evaluation complete!")
    print(f"[INFO] Results saved to: {video_dir}")
    
    # 关闭环境
    env.close()
    simulation_app.close()


def save_images_with_background(obs, background_img, save_dir, episode, step):
    """
    保存带背景替换的图像
    
    Args:
        obs: 观测数据
        background_img: 背景图像
        save_dir: 保存目录
        episode: episode编号
        step: step编号
    """
    try:
        # 获取RGB和语义分割数据
        rgb = obs['policy']['external_cam_side']
        seg = obs['policy']['semantic_segmentation_side']
        
        # 处理RGB数据: [1, 3, H, W] -> [H, W, 3]
        rgb_img = rgb.view(1, 3, 128, 128)[0].permute(1, 2, 0).cpu().numpy()
        rgb_img = (rgb_img * 255).astype(np.uint8) if rgb_img.max() <= 1.0 else rgb_img
        
        # 处理语义分割: [1, 4, H, W] -> [H, W, 4] (RGBA)
        seg_rgba = seg.view(1, 4, 128, 128)[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        seg_rgb = seg_rgba[:, :, :3]
        
        # 定义cube和robot的颜色
        cube_color = np.array([255, 36, 66], dtype=np.uint8)      # 红色
        robot_color = np.array([204, 110, 248], dtype=np.uint8)   # 紫色
        
        # 创建mask
        cube_mask = np.all(seg_rgb == cube_color, axis=2)
        robot_mask = np.all(seg_rgb == robot_color, axis=2)
        foreground_mask = cube_mask | robot_mask
        
        # 合成图像: 前景(cube+robot) + 背景
        composite_img = background_img.copy()
        composite_img[foreground_mask] = rgb_img[foreground_mask]
        
        # 保存
        save_path = save_dir / f"ep{episode:03d}_step{step:04d}.png"
        cv2.imwrite(str(save_path), cv2.cvtColor(composite_img, cv2.COLOR_RGB2BGR))
        
    except Exception as e:
        print(f"[WARNING] Failed to save image: {e}")


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Evaluation script")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to evaluate")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument(
        "--task_config", 
        type=str, 
        default=None,
        help="Path to YAML task config file"
    )
    parser.add_argument(
        "--task", 
        type=str, 
        default="FrankaFr3-OurGripper-IKRel-PPOBC-VisPour-v0",
        help="Task name (e.g., FrankaFr3-OurGripper-IKRel-PPOBC-VisPour-v0)"
    )
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    
    args = parser.parse_args()
    
    main(
        episodes=args.episodes,
        headless=args.headless,
        task_config=args.task_config,
        task=args.task,
        num_envs=args.num_envs,
        device=args.device,
    )