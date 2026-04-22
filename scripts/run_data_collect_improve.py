import argparse
import subprocess
import sys
import os
import traceback
from pathlib import Path
from isaaclab.app import AppLauncher
import pty
import select
# === 配置: 场景编辑器脚本的路径 ===
# 相对于仓库根目录，基于本文件位置解析为绝对路径
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
        print(f"[INFO] 命中新任务目录: {categorized_path}")
        return categorized_path

    raise FileNotFoundError(
        "配置文件不存在。\n"
        f"  task_name: {task_name}\n"
        f"  预测分类: {category}\n"
        f"  新路径: {categorized_path}\n"
        "请检查 YAML 是否放在 sceneconfig/task/<category>/ 下。"
    )

def run_scene_editor():
    """
    使用伪终端 (pty) 启动编辑器。
    既允许用户交互 (输入/输出)，又能让主程序在后台监听关键词。
    """
    print(f"\n{'='*80}")
    print(f"STAGE 1: 启动交互式场景编辑器 (PTY模式)")
    print(f"{'='*80}")
    print(f"正在启动子进程: {ENV_EDITOR_PATH}")

    if not os.path.exists(ENV_EDITOR_PATH):
        print(f"[ERROR] 找不到编辑器脚本: {ENV_EDITOR_PATH}")
        return False

    # 1. 创建伪终端主从对
    # master_fd: 主程序用来读写数据的句柄
    # slave_fd:  子程序以为的标准输入/输出终端
    master_fd, slave_fd = pty.openpty()

    # 2. 启动子进程
    # stdin, stdout, stderr 全部指向 slave_fd，模拟真实终端环境
    process = subprocess.Popen(
        [sys.executable, ENV_EDITOR_PATH],
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True # 必须关闭多余的文件描述符
    )

    # 父进程不再需要 slave_fd，关闭它
    os.close(slave_fd)

    target_phrase = "Complete shutdown finished"
    detected = False

    try:
        # 3. 进入 I/O 循环
        while True:
            # 检查子进程是否已结束
            if process.poll() is not None:
                break

            # 使用 select 监听 master_fd 是否有数据可读，或者是用户的标准输入是否有数据
            # rlist: 我们想读的列表 [master_fd (子程序的输出), sys.stdin (用户的键盘输入)]
            rlist, _, _ = select.select([master_fd, sys.stdin], [], [], 0.1)

            # A. 如果子程序输出了内容
            if master_fd in rlist:
                try:
                    data = os.read(master_fd, 1024)
                    if not data: # EOF
                        break
                except OSError:
                    break

                # 解码并打印给用户看 (实现“透传”)
                # errors='replace' 防止某些特殊字符导致崩溃
                text_chunk = data.decode(errors='replace')
                sys.stdout.write(text_chunk)
                sys.stdout.flush()

                # === 核心检测逻辑 ===
                if target_phrase in text_chunk:
                    print(f"\n\n[TRIGGER] 检测到结束信号: '{target_phrase}'")
                    print("[INFO] 正在强制关闭编辑器进程以继续...")
                    process.kill()
                    detected = True
                    break

            # B. 如果用户在键盘输入了内容
            if sys.stdin in rlist:
                # 读取用户输入
                user_input = os.read(sys.stdin.fileno(), 1024)
                # 将用户输入转发给子程序的 master_fd
                os.write(master_fd, user_input)

    except KeyboardInterrupt:
        print("\n[WARN] 用户中断...")
        process.kill()
        return False
    finally:
        os.close(master_fd)
        if process.poll() is None: # 确保进程已死
             # process.kill() 
             # 这里不一定要再次 kill，因为上面 break 前已经 kill 了，或者进程自己退出了
             pass
        try:
             process.wait(timeout=1)
        except:
             pass

    if detected or process.returncode == 0:
        print(f"\n[INFO] 编辑器阶段结束。")
        return True
    else:
        print(f"\n[ERROR] 编辑器非正常退出 (Exit Code: {process.returncode})")
        return False
def main():
    # 1. 解析命令行参数
    parser = argparse.ArgumentParser(description="Interactive data collection pipeline.")
    
    # Pipeline 参数
    parser.add_argument("--task_name", type=str, default="pick",
                        help=(
                            "任务名称。优先按前缀在 sceneconfig/task/<category>/ 中找同名yaml："
                            "pick_**->pick, pour_**->pour, hang_**->hang, stack_**->stack, "
                            "push_**/pull_**->pull_push, open_**/close_**->open_close, 其余->composite。"
                            "仅使用新目录结构，不再兼容旧版 sceneconfig/<task_name>.yaml。"
                        ))

    parser.add_argument("--robot", type=str, default=None, choices=["franka_fr3", "ur5e", "kinova_robotiq85"],
                        help="机器人类型，覆盖YAML中的robot配置。franka_fr3、ur5e 或 kinova_robotiq85")

    parser.add_argument("--skip_editor", action="store_true", help="跳过第一步编辑器，直接使用现有 yaml 跑采集")
    
    # 覆盖参数
    parser.add_argument("--num_envs", type=int, default=None, help="覆盖环境数量")
    parser.add_argument("--episodes", type=int, default=None, help="覆盖采集回合数")
    parser.add_argument("--data_dir", type=str, default=None, help="覆盖数据保存路径")
    parser.add_argument("--seed", type=int, default=None, help="覆盖随机种子")

    # Isaac Lab 标准参数 (headless, device 等)
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()

    # --- 阶段 1: 运行场景编辑器 (子进程) ---
    if not args_cli.skip_editor:
        success = run_scene_editor()
        if not success:
            print("[ERROR] 编辑器阶段失败或被取消，程序终止。")
            sys.exit(1)
    else:
        print("[INFO] 跳过编辑器阶段 (--skip_editor)，直接开始采集。")

    # --- 阶段 2: 运行数据采集 (主进程) ---
    print(f"\n{'='*80}")
    print(f"STAGE 2: 自动化数据采集")
    print(f"{'='*80}")

    # 初始化当前进程的仿真 App (这是极耗资源的操作，必须在编辑器关闭后进行)
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # 延迟导入业务逻辑，防止在 App 初始化前加载 Omniverse 模块
    # from isaaclab.motion_collect.interactive_datacollector_improve import DataCollector
    from isaaclab.motion_collect.datacollector import DataCollector
    from isaaclab_tasks.real2sim.utils.task_config_loader import load_task_config
    from isaaclab_tasks.real2sim import REAL2SIM_ROOT_DIR

    # 机器人类型配置映射
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
        # 根据 task_name 前缀选择分类目录，并兼容旧路径
        config_path_obj = _resolve_task_config_path(REAL2SIM_ROOT_DIR, args_cli.task_name)
        config_path = str(config_path_obj)
        print(f"读取任务配置: {config_path}")

        # 加载配置
        task_config = load_task_config(config_path)
        runtime = task_config.runtime
        scene = task_config.scene

        # --robot 覆盖 YAML 中的 robot 和 env_task_name
        env_task_name = runtime.task_name
        robot_cfg_name = task_config.robot.cfg_name
        robot_type = args_cli.robot  # None if not specified

        if robot_type is not None:
            rc = ROBOT_CONFIGS[robot_type]
            env_task_name = rc["env_task_name"]
            robot_cfg_name = rc["cfg_name"]
            print(f"[INFO] --robot={robot_type} 覆盖: env={env_task_name}, cfg={robot_cfg_name}")

        # 参数覆盖优先级: CLI > YAML
        num_envs = args_cli.num_envs if args_cli.num_envs is not None else scene.num_envs
        episodes = args_cli.episodes if args_cli.episodes is not None else runtime.episodes
        data_dir = args_cli.data_dir if args_cli.data_dir is not None else runtime.data_dir
        if robot_type is not None and data_dir:
            data_dir = f"{data_dir}_{robot_type}"
        seed = args_cli.seed if args_cli.seed is not None else getattr(runtime, 'seed', 42)

        # 实例化采集器
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
            save_depth_images=getattr(runtime, "save_depth_images", False),
            data_dir=data_dir,
            task_name=args_cli.task_name,
        )
        
        # 开始采集循环
        stats = collector.collect()
        
        print(f"\n{'='*80}")
        print("全流程结束")
        print(f"{'='*80}")
        print(f"成功率: {stats.get('success_rate', 0):.1%}")
        
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断了数据采集。")
    except Exception as e:
        print(f"\n[ERROR] 数据采集过程中发生错误:")
        traceback.print_exc()
    finally:
        # 资源清理
        if collector:
            collector.close()
        simulation_app.close()
        print("\n✓ 程序已退出")

if __name__ == "__main__":
    main()