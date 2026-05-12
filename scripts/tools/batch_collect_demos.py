#!/usr/bin/env python3
"""
Batch data collection script for multiple tasks.

This script automates the collection of 50 demonstration trajectories for each task,
with randomization enabled via task YAML configurations.

Usage:
    python batch_collect_demos.py --tasks task1 task2 task3 ... --num_episodes 50
    python batch_collect_demos.py --all --num_episodes 50  # Collect for all 23 tasks
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

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
    print(f"Starting collection for: {task}")
    print(f"Episodes: {num_episodes}")
    print(f"Output: {output_subdir}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✅ Successfully collected data for {task}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed to collect data for {task}")
        print(f"Error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  Collection for {task} interrupted by user")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch collect demonstrations for multiple tasks with randomization enabled"
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
    print(f"BATCH DATA COLLECTION")
    print(f"Tasks: {len(tasks_to_collect)} total")
    print(f"Episodes per task: {args.num_episodes}")
    print(f"Data directory: {args.data_dir.absolute()}")
    print(f"Randomization: Enabled (via task YAML configs)")
    print(f"{'='*80}\n")
    
    # Track results
    successful = []
    failed = []
    
    # Run collection for each task
    for idx, task in enumerate(tasks_to_collect):
        if idx < args.start_from:
            print(f"Skipping task {idx+1}/{len(tasks_to_collect)}: {task}")
            continue
        
        print(f"\n[{idx+1}/{len(tasks_to_collect)}] Processing: {task}")
        
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
            print(f"   - {task}")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)}/{len(tasks_to_collect)}")
        for task in failed:
            print(f"   - {task}")
        print(f"\nTo resume, run:")
        print(f"  python batch_collect_demos.py --all --start_from {len(successful)}")
    
    print(f"\nData saved to: {args.data_dir.absolute()}")
    print(f"{'='*80}\n")
    
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
