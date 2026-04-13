import argparse
import traceback
from datetime import datetime
from pathlib import Path
from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser(description="CuRobo motion planning in Real2Sim environment.")
    parser.add_argument("--task_type", type=str, default="task_gen", help="Type of the task.")
    parser.add_argument("--config", type=str, default="generated_task_config.yaml", help="Name of the config file.")
    # parser.add_argument("--task_type", type=str, default="pick", help="Type of the task.")
    # parser.add_argument("--config", type=str, default="pick_up_cube.yaml", help="Name of the config file.")

    parser.add_argument("--num_envs", type=int, default=None, help="Override number of environments.")
    parser.add_argument("--seed", type=int, default=None, help="Override seed.")
    parser.add_argument("--episodes", type=int, default=None, help="Override number of episodes.")
    parser.add_argument("--data_dir", type=str, default=None, help="Override data directory.")
    
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    from isaaclab.motion_collect.datacollector import DataCollector
    # from isaaclab.motion_collect.copy.datacollector import DataCollector
    from isaaclab_tasks.real2sim.utils.task_config_loader import load_task_config
    
    try:
      
        from isaaclab_tasks.real2sim import REAL2SIM_ROOT_DIR
        config_path = f"{REAL2SIM_ROOT_DIR}/tasks/{args_cli.task_type}/config/sceneconfig/{args_cli.config}"
        print(f"[INFO] Using config file: {config_path}")

        task_config = load_task_config(config_path)
        runtime = task_config.runtime
        scene = task_config.scene

     
        num_envs = args_cli.num_envs if args_cli.num_envs is not None else scene.num_envs
        seed = args_cli.seed if args_cli.seed is not None else getattr(runtime, 'seed', 42)
        episodes = args_cli.episodes if args_cli.episodes is not None else runtime.episodes
        data_dir = args_cli.data_dir if args_cli.data_dir is not None else runtime.data_dir

        task_name = runtime.task_name
        save_npy = runtime.save_npy
        save_camera_images = runtime.save_camera_images
        save_composite_images = runtime.save_composite_images
        save_mask_images = runtime.save_mask_images
        save_original_images = runtime.save_original_images
        enable_motion_planning = getattr(runtime, 'enable_motion_planning', True)
        robot_cfg_name = task_config.robot.cfg_name

        print(f"[INFO] Configuration from YAML:")
        print(f"  - Task: {task_name}")
        print(f"  - Num envs: {num_envs}")
        print(f"  - Episodes: {episodes}")
        print(f"  - Save npy: {save_npy}")
        print(f"  - Save camera images: {save_camera_images}")
        print(f"  - Save composite images: {save_composite_images}")
        print(f"  - Save mask images: {save_mask_images}")
        print(f"  - Save original images: {save_original_images}")
        print(f"  - Data dir: {data_dir}")
        print(f"  - Robot config: {robot_cfg_name}")

        collector = DataCollector(
            task_name=task_name,
            task_config_path=config_path,
            robot_cfg_name=robot_cfg_name,
            num_envs=num_envs,
            device=args_cli.device,
            seed=seed,
            headless=args_cli.headless,
            enable_motion_planning=enable_motion_planning,
            episodes=episodes,
            save_npy=save_npy,
            save_camera_images=save_camera_images,
            save_composite_images=save_composite_images,
            save_mask_images=save_mask_images,
            save_original_images=save_original_images,
            data_dir=data_dir,
        )
        
        print(f"[INFO] Starting data collection. Config: {args_cli.config}")
        stats = collector.collect()
        
    except Exception as e:
        print(f"[ERROR] An error occurred during data collection:")
        traceback.print_exc()
    finally:
        if 'collector' in locals():
            collector.close()
        simulation_app.close()
        print("✓ DataCollector closed")


if __name__ == "__main__":
    main()