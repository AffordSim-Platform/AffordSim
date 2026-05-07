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
    
 
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description="Evaluation script for pouring task.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = headless
    args_cli.device = device
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    
    import isaaclab_tasks
    
 
    if task_config is None:
 
        task_config = Path("/home/hk/limingyang/code/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/pour/config/sceneconfig/pour_beans.yaml")
    else:
        task_config = Path(task_config)
    
    print(f"[INFO] Loading task config from: {task_config}")
    

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
    
    
    print(f"[INFO] Creating environment: {task}")
    
  
    env_cfg_class = get_env_cfg_class_by_name(task)
    

    env_cfg = create_env_cfg_from_yaml(
        yaml_path=str(task_config),
        env_cfg_class=env_cfg_class,
        num_envs=num_envs,
        device=device,
        seed=42,
    )

    print(f"[INFO] Environment configuration loaded:")
    print(f"  - Task: {task}")
    print(f"  - Num envs: {env_cfg.scene.num_envs}")
    print(f"  - Device: {env_cfg.sim.device}")
    print(f"  - Episode length: {env_cfg.episode_length_s}s")
    
   
    env = gym.make(task, cfg=env_cfg)
    obs, _ = env.reset()
    obs, _ = env.reset()
 
    
    # 1: Joint Position Actions
    PourBeansTrajectory = torch.tensor([
    [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04],    
    [0.3, -0.5, 0.0, -2.2, 0.0, 1.9, 0.785, 0.04],     
    [0.5, -0.3, 0.0, -2.0, 0.0, 2.0, 0.785, 0.04],          
    [0.55, -0.1, 0.0, -1.8, 0.0, 2.1, 0.785, 0.04],          
    [0.55, -0.1, 0.0, -1.8, 0.0, 2.1, 0.785, 0.0],          
    [0.55, -0.1, 0.0, -2.0, 0.0, 2.0, 0.785, 0.0],        
    [0.55, 0.0, 0.0, -2.1, 0.0, 1.95, 0.785, 0.0],           
    [0.6, 0.15, 0.0, -2.0, 0.0, 1.9, 0.785, 0.0],          
    [0.6, 0.15, 0.0, -1.85, 0.0, 2.0, 0.785, 0.0],         
    [0.6, 0.15, 0.3, -1.85, 0.0, 2.0, 0.785, 0.0],         
    [0.6, 0.15, 0.6, -1.85, 0.0, 2.0, 0.785, 0.0],            
    [0.6, 0.15, 0.9, -1.85, 0.0, 2.0, 0.785, 0.0],           
    [0.6, 0.15, 0.9, -1.85, 0.0, 2.0, 0.785, 0.0],          
    [0.6, 0.15, 0.3, -1.85, 0.0, 2.0, 0.785, 0.0],           
    [0.6, 0.15, 0.0, -1.85, 0.0, 2.0, 0.785, 0.04],         
    [0.6, 0.15, 0.0, -2.0, 0.0, 1.9, 0.785, 0.04],        
    [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04]      
])
    
    # 2: Differential IK Actions
    DifferentialIKPourTrajectory = torch.tensor([
    # [x, y, z, qx, qy, qz, qw, gripper]
    # : [0.45, 0.0, 0.0213], : [0.5, 0.20, 0.0106]
  
    # [0.45, 0.0, 0.15, 0 , 1, 0,0, 0],
    # [0.45, 0.0, 0.15, 0 , 1, 0,0, 0],
    # [0.45, 0.0, 0.15, 0 , 1, 0,0, 0],
    # [0.45, 0.0, 0.05, 0 , 1, 0,0, 0],
    # [0.45, 0.0, 0.05, 0 , 1, 0,0, -1],
    # [0.45, 0.0, 0.05, 0 , 1, 0,0, -1],
    # [0.45, 0.0, 0.05, 0 , 1, 0,0, -1],
    # [0.45, 0.0, 0.05, 0 , 1, 0,0, -1],
    # [0.50, 0.20, 0.20, -0.271, 0.653, 0.653, 0.271, -1],  
    # [0.50, 0.20, 0.20, -0.092, 0.701, 0.701, 0.092, -1], 
    # [0.50, 0.20, 0.20, 0.0, 0.707, 0.707, 0.0, -1],    
    # [0.50, 0.20, 0.20, 0.0, 0.707, 0.707, 0.0, -1],      
    # [0.50, 0.20, 0.20, -0.5, 0.5, 0.5, 0.5, -1],      
    # [0.50, 0.20, 0.08, -0.5, 0.5, 0.5, 0.5, 0],        
    # [0.50, 0.20, 0.25, -0.5, 0.5, 0.5, 0.5, 0],         
    # [0.30, 0.00, 0.30, -0.5, 0.5, 0.5, 0.5, 0] 
    [0.30, 0.00, 0.20, -0.5, 0.5, 0.5, 0.5, 0],         
    [0.45, -0.15, 0.15, -0.5, 0.5, 0.5, 0.5, 0],        
    [0.45, -0.10, 0.08, -0.5, 0.5, 0.5, 0.5, 0],         
    [0.45, -0.05, 0.05, -0.5, 0.5, 0.5, 0.5, 0],        
    [0.45, 0.05, 0.05, -0.5, 0.5, 0.5, 0.5, 0],        
    [0.45, 0.05, 0.05, -0.5, 0.5, 0.5, 0.5, -1],         
    [0.45, 0.00, 0.20, -0.5, 0.5, 0.5, 0.5, -1],          
    [0.475, 0.10, 0.25, -0.5, 0.5, 0.5, 0.5, -1],         
    [0.50, 0.20, 0.25, -0.5, 0.5, 0.5, 0.5, -1],           
    [0.50, 0.20, 0.20, -0.5, 0.5, 0.5, 0.5, -1],          
    [0.50, 0.20, 0.20, -0.271, 0.653, 0.653, 0.271, -1],  
    [0.50, 0.20, 0.20, -0.092, 0.701, 0.701, 0.092, -1],  
    [0.50, 0.20, 0.20, 0.0, 0.707, 0.707, 0.0, -1],        
    [0.50, 0.20, 0.20, 0.0, 0.707, 0.707, 0.0, -1],       
    [0.50, 0.20, 0.20, -0.5, 0.5, 0.5, 0.5, -1],          
    [0.50, 0.20, 0.08, -0.5, 0.5, 0.5, 0.5, 0],         
    [0.50, 0.20, 0.25, -0.5, 0.5, 0.5, 0.5, 0],           
    [0.30, 0.00, 0.30, -0.5, 0.5, 0.5, 0.5, 0]  
])
    
  
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
    
  
    video_dir = Path("runs") / datetime.now().strftime("%Y-%m-%d") / datetime.now().strftime("%H-%M-%S")
    video_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Saving results to: {video_dir}")
 
    background_path = Path("/home/hk/limingyang/code/IsaacLab/runs/2025-12-16/00-07-39/background_realworld.jpg")
    if background_path.exists():
        background_img = cv2.imread(str(background_path))
        background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)
        background_img = cv2.resize(background_img, (128, 128))
        print(f"[INFO] Loaded background image from: {background_path}")
    else:
        print(f"[WARNING] Background image not found at {background_path}")
        background_img = None
    
    max_steps = env.env.max_episode_length
    print(f"[INFO] Starting evaluation: {episodes} episodes, max {max_steps} steps each")
    
    for ep in tqdm(range(episodes), desc="Episodes"):
        obs, _ = env.reset()
        
        episode_rewards = []
        episode_success = False
        
        for i in range(max_steps):
           
            action = actions_test[i % len(actions_test)].unsqueeze(0)
            
            if i % len(actions_test) == 0 and i != 0:
                obs, _ = env.reset()
            
            if i % 10 == 0:
                ee_pos = env.env.scene["robot"].data.body_pos_w[0, 9]
                ee_quat = env.env.scene["robot"].data.body_quat_w[0, 9]
                formatted_pos = ", ".join(f"{float(x):.4f}" for x in ee_pos)
                formatted_quat = ", ".join(f"{float(x):.4f}" for x in ee_quat)
                print(f"\nEp {ep+1}, Step {i}")
                print(f"  EE pos: [{formatted_pos}]")
                print(f"  EE quat: [{formatted_quat}]")
                print(f"  Action: {action[0].cpu().numpy()}")
            
          
            obs, reward, term, trunc, info = env.step(action)
            episode_rewards.append(reward)
            
         
            if background_img is not None and i % 10 == 0:
                save_images_with_background(
                    obs, background_img, video_dir, ep, i
                )
            
         
            if term or trunc:
                # episode_success = term
                # print(f"\nEpisode {ep+1} finished at step {i}")
                # print(f"  Success: {episode_success}")
                # print(f"  Total reward: {sum(episode_rewards):.2f}")
                break
        
       
        episode_stats = {
            "episode": ep,
            "steps": i + 1,
            "total_reward": sum(episode_rewards),
            "success": episode_success,
        }
    
        stats_path = video_dir / f"episode_{ep:04d}_stats.txt"
        with open(stats_path, 'w') as f:
            for key, value in episode_stats.items():
                f.write(f"{key}: {value}\n")
    
    print(f"\n[INFO] Evaluation complete!")
    print(f"[INFO] Results saved to: {video_dir}")
    
  
    env.close()
    simulation_app.close()


def save_images_with_background(obs, background_img, save_dir, episode, step):
 
    try:
      
        rgb = obs['policy']['external_cam_side']
        seg = obs['policy']['semantic_segmentation_side']
        
   
        rgb_img = rgb.view(1, 3, 128, 128)[0].permute(1, 2, 0).cpu().numpy()
        rgb_img = (rgb_img * 255).astype(np.uint8) if rgb_img.max() <= 1.0 else rgb_img
   
        seg_rgba = seg.view(1, 4, 128, 128)[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        seg_rgb = seg_rgba[:, :, :3]
        
    
        cube_color = np.array([255, 36, 66], dtype=np.uint8)      
        robot_color = np.array([204, 110, 248], dtype=np.uint8) 
        
        
        cube_mask = np.all(seg_rgb == cube_color, axis=2)
        robot_mask = np.all(seg_rgb == robot_color, axis=2)
        foreground_mask = cube_mask | robot_mask
      
        composite_img = background_img.copy()
        composite_img[foreground_mask] = rgb_img[foreground_mask]
        
      
        save_path = save_dir / f"ep{episode:03d}_step{step:04d}.png"
        cv2.imwrite(str(save_path), cv2.cvtColor(composite_img, cv2.COLOR_RGB2BGR))
        
    except Exception as e:
        print(f"[WARNING] Failed to save image: {e}")


if __name__ == "__main__":
  
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