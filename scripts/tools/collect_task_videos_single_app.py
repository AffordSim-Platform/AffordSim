#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import cv2

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = ROOT / "source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task"
VIDEO_ROOT = ROOT.parent / "video"


@dataclass
class TaskRunResult:
    task_name: str
    status: str
    attempts: int
    data_dir: str = ""
    videos: list[str] | None = None
    error: str = ""


def discover_task_paths() -> dict[str, Path]:
    task_paths: dict[str, Path] = {}
    for path in sorted(TASK_ROOT.rglob("*.yaml")):
        if path.name == "example.yaml" or "先不用" in str(path):
            continue
        task_paths[path.stem] = path
    return task_paths


def _write_mp4(frame_dir: Path, out_path: Path, fps: int = 12) -> bool:
    frames = sorted(frame_dir.glob("step_*.png"))
    if not frames:
        return False

    first = cv2.imread(str(frames[0]))
    if first is None:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    for frame_path in frames:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        writer.write(frame)

    writer.release()
    return out_path.exists() and out_path.stat().st_size > 0


def make_videos(recorder_dir: Path, out_dir: Path) -> list[str]:
    image_roots = []
    for name in ("original_images", "camera_images"):
        candidate = recorder_dir / name
        if candidate.exists():
            image_roots.append(candidate)

    video_paths: list[str] = []
    for image_root in image_roots:
        for camera in ("external_cam_side", "external_cam_front", "wrist_cam"):
            camera_root = image_root / camera
            if not camera_root.exists():
                continue
            env_dirs = sorted(camera_root.glob("env*_ep*"))
            if not env_dirs:
                continue
            frame_dir = env_dirs[-1]
            out_path = out_dir / f"{camera}.mp4"
            if _write_mp4(frame_dir, out_path):
                video_paths.append(str(out_path))
    return video_paths


def is_non_retriable_error(error_text: str) -> bool:
    patterns = (
        "Failed to find a rigid body",
        "Failed to find a single rigid body",
        "Not all regular expressions are matched",
        "Config file not found",
        "no primitives configured",
    )
    return any(pattern in error_text for pattern in patterns)


def run_task(
    task_name: str,
    config_path: Path,
    out_root: Path,
    device: str,
    headless: bool,
    max_attempts: int,
) -> TaskRunResult:
    from isaaclab.motion_collect.datacollector import DataCollector
    from isaaclab_tasks.real2sim.utils.task_config_loader import load_task_config

    task_root = out_root / task_name
    task_root.mkdir(parents=True, exist_ok=True)

    last_error = ""
    last_data_dir = ""
    for attempt in range(1, max_attempts + 1):
        task_config = load_task_config(str(config_path))
        runtime = task_config.runtime
        attempt_dir = task_root / "data" / f"attempt_{attempt:02d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        attempt_dir.mkdir(parents=True, exist_ok=True)

        collector = None
        try:
            collector = DataCollector(
                env_task_name=runtime.task_name,
                task_config_path=str(config_path),
                robot_cfg_name=task_config.robot.cfg_name,
                robot_type=task_config.robot.type,
                num_envs=1,
                device=device,
                seed=1000 + attempt,
                headless=headless,
                enable_motion_planning=runtime.enable_motion_planning,
                episodes=1,
                save_npy=runtime.save_npy,
                save_camera_images=runtime.save_camera_images,
                save_composite_images=runtime.save_composite_images,
                save_mask_images=runtime.save_mask_images,
                save_original_images=runtime.save_original_images,
                save_depth_images=getattr(runtime, "save_depth_images", False),
                data_dir=attempt_dir,
                task_name=task_name,
            )
            stats = collector.collect()
            recorder_dir = collector.recorder.save_dir if collector.recorder is not None else None
            last_data_dir = str(recorder_dir or attempt_dir)
            if stats.get("successful_episodes", 0) > 0:
                video_paths = []
                if recorder_dir is not None:
                    video_paths = make_videos(Path(recorder_dir), task_root / "videos")
                return TaskRunResult(
                    task_name=task_name,
                    status="success",
                    attempts=attempt,
                    data_dir=last_data_dir,
                    videos=video_paths,
                )
            last_error = json.dumps(stats)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if is_non_retriable_error(last_error):
                break
        finally:
            if collector is not None:
                collector.close()

    return TaskRunResult(
        task_name=task_name,
        status="failed",
        attempts=max_attempts,
        data_dir=last_data_dir,
        videos=[],
        error=last_error,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect one successful trajectory per task and export videos.")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument("--output_name", type=str, default=None)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)

    task_paths = discover_task_paths()
    tasks = args.tasks or sorted(task_paths)

    out_root = VIDEO_ROOT / (args.output_name or f"all_task_single_app_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out_root.mkdir(parents=True, exist_ok=True)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    manifest_path = out_root / "manifest.tsv"
    try:
        with manifest_path.open("w", encoding="utf-8") as manifest:
            manifest.write("task\tstatus\tattempts\tdata_dir\tvideos\terror\n")
            for idx, task_name in enumerate(tasks, 1):
                config_path = task_paths.get(task_name)
                if config_path is None:
                    manifest.write(f"{task_name}\tmissing\t0\t\t\tconfig_not_found\n")
                    manifest.flush()
                    continue

                print(f"[{idx:02d}/{len(tasks):02d}] {task_name}", flush=True)
                result = run_task(
                    task_name=task_name,
                    config_path=config_path,
                    out_root=out_root,
                    device=args.device,
                    headless=args.headless,
                    max_attempts=args.max_attempts,
                )
                videos = ",".join(result.videos or [])
                manifest.write(
                    f"{result.task_name}\t{result.status}\t{result.attempts}\t"
                    f"{result.data_dir}\t{videos}\t{result.error}\n"
                )
                manifest.flush()
                print(f"  -> {result.status} after {result.attempts} attempt(s)", flush=True)
                if result.videos:
                    print(f"  -> videos: {', '.join(result.videos)}", flush=True)
    finally:
        simulation_app.close()

    print(out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
