import argparse
import subprocess
import sys
import os
import traceback
from pathlib import Path
from isaaclab.app import AppLauncher
import pty
import select

ENV_EDITOR_PATH = str(
    (Path(__file__).resolve().parent.parent
     / "source/isaaclab/isaaclab/motion_collect/scene_create/env_editor.py")
)

def _infer_task_category(task_name: str) -> str:
    """Infer task category folder name from task_name prefix."""
    name = task_name.lower()
    if name.startswith("pick"):
        return "pick"
    if name.startswith("pour"):
        return "pour"
    if name.startswith("hang"):
        return "hang"
    if name.startswith("stack"):
        return "stack"
    if name.startswith("push") or name.startswith("pull"):
        return "pull_push"
    if name.startswith("open") or name.startswith("close"):
        return "open_close"
    return "composite"


def _resolve_task_config_path(real2sim_root_dir: str, task_name: str) -> Path:
    """Resolve YAML path from task_name using only categorized layout."""
    sceneconfig_dir = Path(real2sim_root_dir) / "tasks" / "config"
    category = _infer_task_category(task_name)

    categorized_path = sceneconfig_dir / "task" / category / f"{task_name}.yaml"

    if categorized_path.exists():
        print(f"[INFO] Found config: {categorized_path}")
        return categorized_path

    raise FileNotFoundError(
        "Config file not found.\n"
        f"  task_name: {task_name}\n"
        f"  Inferred category: {category}\n"
        f"  Expected path: {categorized_path}\n"
        "Please verify the YAML is placed under sceneconfig/task/<category>/."
    )

def run_scene_editor():

    if not os.path.exists(ENV_EDITOR_PATH):
        print(f"[ERROR] Editor script not found: {ENV_EDITOR_PATH}")
        return False

    master_fd, slave_fd = pty.openpty()

    process = subprocess.Popen(
        [sys.executable, ENV_EDITOR_PATH],
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True
    )


    os.close(slave_fd)

    target_phrase = "Complete shutdown finished"
    detected = False

    try:

        while True:

            if process.poll() is not None:
                break


            rlist, _, _ = select.select([master_fd, sys.stdin], [], [], 0.1)

            # A. If the subprocess produced output
            if master_fd in rlist:
                try:
                    data = os.read(master_fd, 1024)
                    if not data: # EOF
                        break
                except OSError:
                    break

                text_chunk = data.decode(errors='replace')
                sys.stdout.write(text_chunk)
                sys.stdout.flush()

                # === Core detection logic ===
                if target_phrase in text_chunk:
                    print(f"\n\n[TRIGGER] Shutdown signal detected: '{target_phrase}'")
                    print("[INFO] Force-killing editor process to continue...")
                    process.kill()
                    detected = True
                    break


            if sys.stdin in rlist:

                user_input = os.read(sys.stdin.fileno(), 1024)

                os.write(master_fd, user_input)

    except KeyboardInterrupt:
        print("\n[WARN] Interrupted by user...")
        process.kill()
        return False
    finally:
        os.close(master_fd)
        if process.poll() is None:

             pass
        try:
             process.wait(timeout=1)
        except:
             pass

    if detected or process.returncode == 0:
        print(f"\n[INFO] Editor stage completed.")
        return True
    else:
        print(f"\n[ERROR] Editor exited abnormally (Exit Code: {process.returncode})")
        return False
def main():
    # 1. Parse command-line arguments
    parser = argparse.ArgumentParser(description="Interactive data collection pipeline.")

    # Pipeline arguments
    parser.add_argument("--task_name", type=str, default="pick",
                        help=(
                            "Task name. Resolves to sceneconfig/task/<category>/<task_name>.yaml by prefix: "
                            "pick_**->pick, pour_**->pour, hang_**->hang, stack_**->stack, "
                            "push_**/pull_**->pull_push, open_**/close_**->open_close, others->composite. "
                            "Only the new directory structure is supported; legacy sceneconfig/<task_name>.yaml is no longer used."
                        ))

    parser.add_argument("--robot", type=str, default=None, choices=["franka_fr3", "ur5e", "kinova_robotiq85"],
                        help="Robot type, overrides the robot config in YAML. Options: franka_fr3, ur5e, or kinova_robotiq85")

    parser.add_argument("--skip_editor", action="store_true", help="Skip the editor stage and run collection directly using existing YAML")

    # Override arguments
    parser.add_argument("--num_envs", type=int, default=None, help="Override number of environments")
    parser.add_argument("--episodes", type=int, default=None, help="Override number of collection episodes")
    parser.add_argument("--data_dir", type=str, default=None, help="Override data save directory")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")

    # Isaac Lab standard arguments (headless, device, etc.)
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()

    # --- Stage 1: Run scene editor (subprocess) ---
    if not args_cli.skip_editor:
        success = run_scene_editor()
        if not success:
            print("[ERROR] Editor stage failed or was cancelled. Exiting.")
            sys.exit(1)
    else:
        print("[INFO] Skipping editor stage (--skip_editor), starting collection directly.")

    # --- Stage 2: Run data collection (main process) ---
    print(f"\n{'='*80}")
    print(f"STAGE 2: Automated data collection")
    print(f"{'='*80}")

    # Initialize the simulation app for the current process (resource-intensive; must run after editor closes)
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Lazy imports to avoid loading Omniverse modules before app initialization
    # from isaaclab.motion_collect.interactive_datacollector_improve import DataCollector
    from isaaclab.motion_collect.datacollector import DataCollector
    from isaaclab_tasks.real2sim.utils.task_config_loader import load_task_config
    from isaaclab_tasks.real2sim import REAL2SIM_ROOT_DIR

    # Robot type configuration mapping
    ROBOT_CONFIGS = {
        "franka_fr3": {
            "env_task_name": "FrankaFr3-OurGripper-Joint-VisPick-v0",
            "cfg_name": "franka_fr3_our.yml",
            "type": "franka_fr3_default",
        },
        "ur5e": {
            "env_task_name": "UR5e-Robotiq85-IKRel-EE-VisPick-v0",
            "cfg_name": "ur5e_robotiq_2f_85.yml",
            "type": "ur5e",
        },
        "kinova_robotiq85": {
            "env_task_name": "Kinova-Robotiq85-IKRel-EE-VisPick-v0",
            "cfg_name": "kinova_robotiq85.yml",
            "type": "kinova_robotiq85",
        },
    }

    collector = None
    try:
        # Resolve config path from task_name prefix and category directory
        config_path_obj = _resolve_task_config_path(REAL2SIM_ROOT_DIR, args_cli.task_name)
        config_path = str(config_path_obj)
        print(f"Loading task config: {config_path}")

        # Load configuration
        task_config = load_task_config(config_path)
        runtime = task_config.runtime
        scene = task_config.scene

        # --robot overrides robot and env_task_name from YAML
        env_task_name = runtime.task_name
        robot_cfg_name = task_config.robot.cfg_name
        robot_type = args_cli.robot  # None if not specified

        if robot_type is not None:
            rc = ROBOT_CONFIGS[robot_type]
            env_task_name = rc["env_task_name"]
            robot_cfg_name = rc["cfg_name"]
            print(f"[INFO] --robot={robot_type} override: env={env_task_name}, cfg={robot_cfg_name}")

        # Override priority: CLI > YAML
        num_envs = args_cli.num_envs if args_cli.num_envs is not None else scene.num_envs
        episodes = args_cli.episodes if args_cli.episodes is not None else runtime.episodes
        data_dir = args_cli.data_dir if args_cli.data_dir is not None else runtime.data_dir
        if robot_type is not None and data_dir:
            data_dir = f"{data_dir}_{robot_type}"
        seed = args_cli.seed if args_cli.seed is not None else getattr(runtime, 'seed', 42)

        # Instantiate collector
        collector = DataCollector(
            env_task_name=env_task_name,
            task_config_path=config_path,
            robot_cfg_name=robot_cfg_name,
            robot_type=robot_type,
            num_envs=num_envs,
            device=args_cli.device,
            seed=seed,
            headless=args_cli.headless,
            enable_motion_planning=getattr(runtime, 'enable_motion_planning', True),
            episodes=episodes,
            save_npy=runtime.save_npy,
            save_camera_images=runtime.save_camera_images,
            save_composite_images=runtime.save_composite_images,
            save_mask_images=runtime.save_mask_images,
            save_original_images=runtime.save_original_images,
            data_dir=data_dir,
            task_name=args_cli.task_name,
        )

        # Start collection loop
        stats = collector.collect()

        print(f"\n{'='*80}")
        print("Pipeline completed")
        print(f"{'='*80}")
        print(f"Success rate: {stats.get('success_rate', 0):.1%}")

    except KeyboardInterrupt:
        print("\n[INFO] Data collection interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] Error during data collection:")
        traceback.print_exc()
    finally:
        # Resource cleanup
        if collector:
            collector.close()
        simulation_app.close()
        print("\nDone. Program exited.")

if __name__ == "__main__":
    main()
