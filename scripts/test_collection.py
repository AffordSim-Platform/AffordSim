"""Run one local-checkout collection episode using a generated DA3 task YAML."""

import argparse
import json
import os
from pathlib import Path
import sys
import traceback

from test_backgrounds import ROOT, local_modules
from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Task YAML, e.g. from test_backgrounds.py")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/collection_validation"))
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"repository": str(ROOT), "modules": local_modules(), "config": str(Path(args.config).resolve()), "completed": False}
    app = AppLauncher(args).app
    try:
        from isaaclab.motion_collect.datacollector import DataCollector
        from isaaclab_tasks.real2sim.utils.task_config_loader import load_task_config

        config = load_task_config(args.config)
        collector = DataCollector(
            env_task_name=config.runtime.task_name,
            task_config_path=str(Path(args.config).resolve()),
            robot_cfg_name=config.robot.cfg_name,
            robot_type="franka_fr3",
            headless=args.headless,
            enable_cameras=args.enable_cameras,
            enable_motion_planning=True,
            device=args.device,
            num_envs=1,
            episodes=1,
            seed=42,
            data_dir=output / "episodes",
            task_name=config.task.name,
        )
        report["stats"] = collector.collect()
        report["completed"] = bool(report["stats"])
        report["modules_after_collection"] = local_modules()
        print("COLLECTION_RESULT", json.dumps(report, ensure_ascii=False, default=str), flush=True)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        sys.stdout.flush()
        sys.stderr.flush()
        if "error" in report:
            os._exit(1)
        app.close(skip_cleanup=True)


if __name__ == "__main__":
    main()
