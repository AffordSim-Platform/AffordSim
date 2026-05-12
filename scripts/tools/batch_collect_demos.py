#!/usr/bin/env python3
"""
Batch data collection script for multiple tasks with automatic GIF export.

This script automates the collection of 50 demonstration trajectories for each task,
with randomization enabled via task YAML configurations. GIFs are automatically
exported for each task upon collection completion.

Usage:
    python batch_collect_demos.py --tasks task1 task2 task3 ... --num_episodes 50
    python batch_collect_demos.py --all --num_episodes 50  # Collect for all 23 tasks
"""

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

from PIL import Image


# 23 successful tasks ready for batch collection
ALL_TASKS = [
    "add_dough_in_drawer",
    "close_drawer",
    "close_microwave",
    "drop_ball_in_box",
    "drop_ball_in_cup",
    "drop_pen_in_drawer",
    "hang_basket_in_storage",
    "hang_cup_on_rack",
    "hang_towel_in_rack",
    "keep_towel_in_rack",
    "move_pan_to_fire",
    "open_dishwasher",
    "open_drawer",
    "open_fridge",
    "open_oven",
    "pick_basket",
    "pick_corn_place_beaker",
    "pick_cup_place_shelf",
    "pick_headset_place_on_hook",
    "pick_plank_place_on_shelf",
    "place_banana_in_fruitbowl",
    "plug_phone_in_charger",
    "push_chair",
]


def find_image_root(data_dir: Path) -> Path | None:
    """Find the camera images directory within collected data."""
    candidates = sorted(data_dir.rglob("original_images"))
    if not candidates:
        candidates = sorted(data_dir.rglob("camera_images"))
    return candidates[-1] if candidates else None


def iter_camera_dirs(image_root: Path) -> Iterable[tuple[str, Path]]:
    """Iterate over available camera directories."""
    for camera in ("external_cam_side", "external_cam_front", "wrist_cam"):
        camera_root = image_root / camera
        if not camera_root.exists():
            continue
        env_dirs = sorted(camera_root.glob("env*_ep*"))
        if not env_dirs:
            continue
        yield camera, env_dirs[-1]


def export_gif(frame_dir: Path, out_path: Path, fps: int = 12) -> tuple[bool, int]:
    """Export a directory of frames as an animated GIF."""
    frame_paths = sorted(frame_dir.glob("step_*.png"))
    if not frame_paths:
        return False, 0

    frames: list[Image.Image] = []
    for frame_path in frame_paths:
        try:
            with Image.open(frame_path) as img:
                frames.append(img.convert("RGB"))
        except Exception as e:
            print(f"      Warning: Failed to load frame {frame_path}: {e}")
            continue

    if not frames:
        return False, 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, int(round(1000 / fps)))
    
    try:
        frames[0].save(
            out_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
        )
        for frame in frames:
            frame.close()
        return True, len(frame_paths)
    except Exception as e:
        print(f"      Error saving GIF to {out_path}: {e}")
        return False, 0


def export_task_gifs(task: str, output_subdir: Path) -> bool:
    """
    Export GIFs for all available cameras in a collected task.
    
    Args:
        task: Task name
        output_subdir: The output subdirectory for this task run
        
    Returns:
        bool: True if at least one GIF was successfully exported
    """
    image_root = find_image_root(output_subdir)
    if image_root is None:
        print(f"   ⚠️  No camera images found in {output_subdir}")
        return False

    gif_dir = output_subdir / "gifs"
    
    exported_count = 0
    total_frames = 0
    
    for camera, frame_dir in iter_camera_dirs(image_root):
        gif_path = gif_dir / f"{camera}.gif"
        ok, frame_count = export_gif(frame_dir, gif_path, fps=12)
        if ok:
            print(f"   ✅ {camera}: {frame_count} frames → {gif_path.name}")
            exported_count += 1
            total_frames += frame_count
        else:
            print(f"   ⚠️  {camera}: No frames found")
    
    return exported_count > 0


def run_collection(task: str, num_episodes: int, data_dir: Path, headless: bool = True) -> bool:
    """
    Run data collection for a single task.

    Args:
        task: Task name (e.g., "close_drawer")
        num_episodes: Number of episodes to collect (50)
        data_dir: Directory to save data
        headless: Whether to run in headless mode

    Returns:
        bool: True if collection succeeded, False otherwise
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_subdir = data_dir / f"{task}_50_run{timestamp}"
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    dataset_file = output_subdir / "dataset.hdf5"
    
    # Build the record_demos.py command
    cmd = [
        "python",
        "scripts/tools/record_demos.py",
        f"--task={task}",
        f"--num_demos={num_episodes}",
        f"--dataset_file={dataset_file}",
        "--step_hz=30",
        "--teleop_device=keyboard",
    ]
    
    if headless:
        # Note: record_demos might not support headless mode in the same way
        # Adjust if needed based on actual implementation
        pass
    
    print(f"\n{'='*80}")
    print(f"Collecting: {task}")
    print(f"Episodes: {num_episodes}, Output: {output_subdir}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n ✅ Data collection completed for {task}")
        
        # Export GIFs automatically after collection
        print(f" 📹 Exporting GIFs...")
        if export_task_gifs(task, output_subdir):
            print(f" ✅ GIFs exported for {task}")
        else:
            print(f" ⚠️  No GIFs generated (no camera images found)")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n ❌ Failed to collect data for {task}")
        print(f" Error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n ⚠️  Collection for {task} interrupted by user")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch collect demonstrations for multiple tasks with automatic GIF export"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Specific tasks to collect. If not provided, requires --all flag"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Collect for all 23 successful tasks"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=50,
        help="Number of episodes to collect per task (default: 50)"
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/r2s2r_data"),
        help="Base directory for collected data (default: data/r2s2r_data)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no GUI)"
    )
    parser.add_argument(
        "--start_from",
        type=int,
        default=0,
        help="Start from task index (useful for resuming interrupted collections)"
    )
    
    args = parser.parse_args()
    
    # Determine which tasks to collect
    if args.all:
        tasks_to_collect = ALL_TASKS
    elif args.tasks:
        tasks_to_collect = args.tasks
    else:
        parser.error("Please specify --tasks or use --all flag")
    
    # Ensure data directory exists
    args.data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"BATCH DATA COLLECTION WITH GIF EXPORT")
    print(f"{'='*80}")
    print(f"Tasks: {len(tasks_to_collect)} total")
    print(f"Episodes per task: {args.num_episodes}")
    print(f"Output directory: {args.data_dir.absolute()}")
    print(f"GIF export: ✅ Enabled (automatic after each task)")
    print(f"Randomization: ✅ Enabled (via task YAML configs)")
    print(f"{'='*80}\n")
    
    # Track results
    successful = []
    failed = []
    
    # Run collection for each task
    for idx, task in enumerate(tasks_to_collect):
        if idx < args.start_from:
            print(f"[{idx+1}/{len(tasks_to_collect)}] Skipping: {task}")
            continue
        
        print(f"\n[{idx+1}/{len(tasks_to_collect)}] Task: {task}")
        
        if run_collection(task, args.num_episodes, args.data_dir, args.headless):
            successful.append(task)
        else:
            failed.append(task)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"BATCH COLLECTION SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Successful: {len(successful)}/{len(tasks_to_collect)}")
    if successful:
        for task in successful:
            print(f"   • {task}")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)}/{len(tasks_to_collect)}")
        for task in failed:
            print(f"   • {task}")
        print(f"\nTo resume, run:")
        print(f"  python batch_collect_demos.py --all --start_from {len(successful)}")
    
    print(f"\n📁 Output directory: {args.data_dir.absolute()}")
    print(f"   Structure per task: <task>_50_run<timestamp>/")
    print(f"     • dataset.hdf5 (collected trajectories)")
    print(f"     • gifs/ (external_cam_side.gif, external_cam_front.gif, wrist_cam.gif)")
    print(f"{'='*80}\n")
    
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
