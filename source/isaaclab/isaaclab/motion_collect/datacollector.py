"""
Interactive DataCollector - Interactive data collection main module

Vision-pipeline-based data collector, maintaining the same interface structure as datacollector.py.
Path: isaaclab/motion_collect/interactive_datacollector_improve.py

Improvements:
1. Multi-environment obstacle update support
2. Simplified label parsing logic based on YAML format
"""

import sys
import os
import gymnasium as gym
import torch
import numpy as np
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
from ruamel.yaml import YAML
# === Isaac Lab modules ===
from isaaclab_tasks.real2sim.utils.EpisodeBuffer import EpisodeDataRecorder
from isaaclab.motion_collect.motion_planner import MotionPlanner

# === Custom modules (same directory) ===
from .tools.vision_pipeline import VisionPipeline
from .utils import CollectorUtilsMixin
from .primitives.runtime import build_primitive_context, build_primitive_handlers
from .primitives.utils import (
    normalize_primitive_mode, prepare_primitive_episode,
    get_pick_robot_params, get_pick_pos_robot_params,
    load_grasp_config, compute_aligned_grasp_quat,
)

# Get current file path
current_data_path = Path(__file__).parent
TEMP_VISION_DIR = current_data_path.parent.parent.parent.parent / "anygrasp_file"


class DataCollector(
    CollectorUtilsMixin,
):
    """
    Data collector for Real2Sim environment with CuRobo motion planning.
    Uses vision perception pipeline for grasp detection.

    Maintains the same interface as datacollector.py, but internally uses the vision pipeline
    instead of ground truth poses.
    """
    
    def __init__(
        self,
        env_task_name: str = "FrankaFr3-OurGripper-Joint-VisPick-v0",
        task_config_path: Optional[str] = None,
        robot_cfg_name: str = "franka_fr3_our.yml",
        robot_type: Optional[str] = None,
        headless: bool = True,
        enable_cameras: bool = True,
        enable_motion_planning: bool = True,
        device: str = "cuda:0",
        curobo_path: Optional[Path] = None,
        save_npy: bool = True,
        save_camera_images: bool = True,
        save_composite_images: bool = False,
        save_mask_images: bool = False,
        save_original_images: bool = False,
        data_dir: Optional[Path] = None,
        episodes: int = 10,
        video_dir: Optional[Path] = None,
        num_envs: int = 1,
        seed: int = 42,
        task_name: str = "pick",
    ):
        """
        Initialize DataCollector with unified environment creation logic

        Args:
            env_task_name: Registered gym task name (e.g., "FrankaFr3-OurGripper-Joint-VisPick-v0")
            task_config_path: Path to YAML task config file
            robot_cfg_name: Robot configuration file name for CuRobo
            headless: Run in headless mode
            enable_cameras: Enable camera observations
            enable_motion_planning: Enable CuRobo motion planning
            device: Device to use (e.g., "cuda:0")
            curobo_path: Path to CuRobo installation
            save_npy: Whether to save structured trajectory data
            save_camera_images: Whether to save final camera images
            save_composite_images: Whether to save background-replaced composite images
            save_mask_images: Whether to save foreground mask images
            save_original_images: Whether to save original images before replacement
            data_dir: Directory to save data
            episodes: Number of episodes to collect
            video_dir: Directory to save videos
            num_envs: Number of parallel environments
            seed: Random seed
            task_name: Task name for dispatch (pick, pull, stack, wave, pick_redcup)
        """
        self.task_name = task_name
        self.env_task_name = env_task_name
        self.task_config_path = task_config_path
        self.robot_cfg_name = robot_cfg_name
        self.robot_type = robot_type
        self.headless = headless
        self.enable_cameras = enable_cameras
        self.enable_motion_planning = enable_motion_planning
        self.device = device
        self.curobo_path = curobo_path
        self.episodes = episodes
        self.video_dir = video_dir
        self.num_envs = num_envs
        self.seed = seed
        
        self.save_npy = save_npy
        self.save_camera_images = save_camera_images
        self.save_composite_images = save_composite_images
        self.save_mask_images = save_mask_images
        self.save_original_images = save_original_images
        
        # State placeholders
        self.env = None
        self.world = None
        self.robot = None
        self.scene = None
        self.robot_prim_path = None
        self.target_prim_path = None
        self.motion_planner = None
        self.recorder = None
        self.env_cfg = None
        self.task_config = None
        self.data_dir = (Path(data_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")) if data_dir else None
        
        # YAML parser
        self.yaml = YAML()
        self.yaml.preserve_quotes = True
        
        # Vision pipeline components
        self.vision_pipeline = VisionPipeline(temp_dir=TEMP_VISION_DIR)
        
        # Setup temp directory
        TEMP_VISION_DIR.mkdir(parents=True, exist_ok=True)
        
        print("[INFO] DataCollector initialized.")

    # =========================================================================
    # Initialization & Configuration
    # =========================================================================

    def _init_environment(self):
        """Initialize environment using the same logic as train.py"""
        print(f"\n{'='*60}")
        print(f"[INFO] Initializing environment: {self.env_task_name}")
        print(f"{'='*60}")
        
        if "PPOBC" in self.env_task_name or self.task_config_path is not None:
            if self.task_config_path is None:
                raise ValueError(f"Task {self.env_task_name} requires task_config_path")
            
            print(f"[INFO] Loading environment from YAML config: {self.task_config_path}")
            
            # from isaaclab_tasks.real2sim.tasks.pick.pick_r2s_env_cfg import (
            #     create_pick_env_cfg_from_yaml,
            #     get_pick_env_cfg_class_by_name,
            # )

            if "Pick" in str(self.env_task_name):
                from isaaclab_tasks.real2sim.tasks.pick_r2s_env_cfg import (
                    create_pick_env_cfg_from_yaml,
                    get_pick_env_cfg_class_by_name,
                )
                create_env_cfg_from_yaml = create_pick_env_cfg_from_yaml
                get_env_cfg_class_by_name = get_pick_env_cfg_class_by_name
                
            else:
                raise ValueError(f"Unsupported task config path: {self.env_task_name}")

            env_cfg_class = get_env_cfg_class_by_name(self.env_task_name)
            env_cfg = create_env_cfg_from_yaml(
                yaml_path=self.task_config_path,
                env_cfg_class=env_cfg_class,
                num_envs=self.num_envs,
                device=self.device,
                seed=self.seed,
            )
            
            self.env_cfg = env_cfg
            self.task_config = env_cfg.task_config
        else:
            print(f"[INFO] Creating standard environment: {self.env_task_name}")
            env_cfg = None
            self.env_cfg = None
            self.task_config = None
        
        if env_cfg is not None:
            self.env = gym.make(
                self.env_task_name,
                cfg=env_cfg,
                render_mode="rgb_array" if not self.headless else None
            )
        else:
            self.env = gym.make(
                self.env_task_name,
                render_mode="rgb_array" if not self.headless else None
            )
        
        # Reset twice (as in original code)
        self.env.reset()
        self.env.reset()
        
        # Store references
        self.world = self.env.env.sim
        self.robot = self.env.env.scene["robot"]
        self.scene = self.env.env.scene
        self.robot_prim_path = str(self.robot.cfg.prim_path).replace(".*", "0")
        
        print(f"✓ Environment initialized successfully")
        print(f"  - Robot prim path: {self.robot_prim_path}")
        print(f"  - Number of environments: {self.num_envs}")
        print(f"{'='*60}\n")

    # =========================================================================
    # Main Collection Logic
    # =========================================================================

    def _get_pick_robot_params(self) -> Dict[str, Any]:
        return get_pick_robot_params()

    def _get_pick_pos_robot_params(self) -> Dict[str, Any]:
        return get_pick_pos_robot_params()

    def _load_grasp_config(self) -> Tuple[np.ndarray, np.ndarray]:
        return load_grasp_config(self.task_config_path)

    def _compute_aligned_grasp_quat(self, obj_quat_wxyz, base_grasp_quat_wxyz):
        return compute_aligned_grasp_quat(obj_quat_wxyz, base_grasp_quat_wxyz)

    def _get_primitive_steps(self) -> List[Dict[str, Any]]:
        """Read primitive sequence from task config; returns empty list if not configured."""
        if self.task_config is None:
            return []

        if not hasattr(self.task_config, "has_primitives") or not self.task_config.has_primitives():
            return []

        if hasattr(self.task_config, "get_primitives"):
            return self.task_config.get_primitives(as_dict=True)

        return []

    def _build_primitive_context(
        self,
        episode_num: int,
        scene_desc: List[dict],
        task_desc: str,
    ) -> Dict[str, Any]:
        """Build shared context for primitive execution."""
        return build_primitive_context(self, episode_num, scene_desc, task_desc)

    def _get_primitive_handlers(self) -> Dict[str, Any]:
        """Register primitive handler functions."""
        return build_primitive_handlers(self)

    def _prepare_primitive_episode(self, context: Dict[str, Any]) -> bool:
        """Execute shared episode preparation for primitive mode."""
        return prepare_primitive_episode(self, context)

    def _normalize_primitive_mode(self, value: Any) -> Any:
        return normalize_primitive_mode(value)

    def _extract_latest_success_mask(
        self, context: Dict[str, Any], primitive_type: str | None = None
    ) -> List[bool] | None:
        """Extract per-env success mask for the current primitive first, then fallback."""
        primitive_mask_key = {
            "pick": "post_pick_success_mask",
            "place": "post_place_success_mask",
            "pour": "post_pour_success_mask",
            "hang": "post_hang_success_mask",
            "push": "post_push_success_mask",
            "open": "post_open_success_mask",
            "close": "post_close_success_mask",
            "return_home": "post_return_home_success_mask",
        }

        mask_keys: List[str] = []
        primary_key = primitive_mask_key.get(primitive_type or "")
        if primary_key is not None:
            mask_keys.append(primary_key)

        # Fallback order: current alive state -> task-level mask -> other primitive masks.
        for key in (
            "env_alive_mask",
            "task_success_mask",
            "post_hang_success_mask",
            "post_place_success_mask",
            "post_pour_success_mask",
            "post_push_success_mask",
            "post_open_success_mask",
            "post_close_success_mask",
            "post_return_home_success_mask",
            "post_pick_success_mask",
        ):
            if key not in mask_keys:
                mask_keys.append(key)

        for key in mask_keys:
            mask = context.get(key)
            if mask is None:
                continue
            try:
                mask_list = [bool(v) for v in list(mask)]
            except Exception:
                continue
            if len(mask_list) == self.num_envs:
                return mask_list
        return None

    def _run_primitives(self, primitive_steps: List[Dict[str, Any]], context: Dict[str, Any]) -> bool:
        """Execute primitive steps in order."""
        primitive_handlers = self._get_primitive_handlers()

        if not self._prepare_primitive_episode(context):
            return False

        print(f"[INFO] Running {len(primitive_steps)} primitives for episode {context['episode_num']}")
        if "env_alive_mask" not in context:
            context["env_alive_mask"] = [True] * self.num_envs
        if "env_failure_reasons" not in context or len(list(context.get("env_failure_reasons", []))) != self.num_envs:
            context["env_failure_reasons"] = ["none"] * self.num_envs

        for step_idx, primitive in enumerate(primitive_steps, start=1):
            if not any(context.get("env_alive_mask", [True] * self.num_envs)):
                print("[ERROR] No active environments remaining. Aborting primitive sequence.")
                return False

            prev_alive = list(context.get("env_alive_mask", [True] * self.num_envs))
            primitive_type = primitive.get("type", "")
            primitive_mode = self._normalize_primitive_mode(primitive.get("mode"))
            primitive_params = dict(primitive.get("params", {}) or {})
            handler = primitive_handlers.get(primitive_type)
            context["step_failure_reason_mask"] = None

            if handler is None:
                print(f"[ERROR] Unsupported primitive type: {primitive_type}")
                context["env_failure_reasons"] = [
                    "planning_failed" if prev_alive[i] and context["env_failure_reasons"][i] == "none" else context["env_failure_reasons"][i]
                    for i in range(self.num_envs)
                ]
                context["env_alive_mask"] = [False] * self.num_envs
                context["task_success_mask"] = [False] * self.num_envs
                return False

            print(
                f"[INFO] Primitive {step_idx}/{len(primitive_steps)}: "
                f"type={primitive_type}, mode={primitive_mode}, params={primitive_params}"
            )

            try:
                success = handler(primitive_mode, primitive_params, context)
            except NotImplementedError as exc:
                print(f"[ERROR] {exc}")
                context["env_failure_reasons"] = [
                    "planning_failed" if prev_alive[i] and context["env_failure_reasons"][i] == "none" else context["env_failure_reasons"][i]
                    for i in range(self.num_envs)
                ]
                context["env_alive_mask"] = [False] * self.num_envs
                context["task_success_mask"] = [False] * self.num_envs
                return False
            except Exception as exc:
                print(f"[ERROR] Primitive '{primitive_type}' failed: {exc}")
                context["env_failure_reasons"] = [
                    "planning_failed" if prev_alive[i] and context["env_failure_reasons"][i] == "none" else context["env_failure_reasons"][i]
                    for i in range(self.num_envs)
                ]
                context["env_alive_mask"] = [False] * self.num_envs
                context["task_success_mask"] = [False] * self.num_envs
                return False

            step_mask = self._extract_latest_success_mask(context, primitive_type=primitive_type)
            if step_mask is not None:
                step_reason_mask_raw = context.get("step_failure_reason_mask")
                if step_reason_mask_raw is not None:
                    try:
                        step_reason_mask = list(step_reason_mask_raw)
                    except Exception:
                        step_reason_mask = [None] * self.num_envs
                    if len(step_reason_mask) != self.num_envs:
                        step_reason_mask = [None] * self.num_envs
                else:
                    step_reason_mask = [None] * self.num_envs

                env_failure_reasons = list(context.get("env_failure_reasons", ["none"] * self.num_envs))
                if len(env_failure_reasons) != self.num_envs:
                    env_failure_reasons = ["none"] * self.num_envs

                for env_idx in range(self.num_envs):
                    if not prev_alive[env_idx] or step_mask[env_idx]:
                        continue
                    reason = step_reason_mask[env_idx]
                    if reason not in {"planning_failed", "success_eval_failed", "target_missing"}:
                        reason = "planning_failed"
                    if env_failure_reasons[env_idx] == "none":
                        env_failure_reasons[env_idx] = reason
                context["env_failure_reasons"] = env_failure_reasons
                context["env_alive_mask"] = [a and b for a, b in zip(prev_alive, step_mask)]
                # Keep task-success mask aligned with the latest primitive result.
                context["task_success_mask"] = list(context["env_alive_mask"])

            if not success:
                print(f"[ERROR] Primitive '{primitive_type}' returned failure")
                if step_mask is None:
                    env_failure_reasons = list(context.get("env_failure_reasons", ["none"] * self.num_envs))
                    if len(env_failure_reasons) != self.num_envs:
                        env_failure_reasons = ["none"] * self.num_envs
                    for env_idx in range(self.num_envs):
                        if prev_alive[env_idx] and env_failure_reasons[env_idx] == "none":
                            env_failure_reasons[env_idx] = "planning_failed"
                    context["env_failure_reasons"] = env_failure_reasons
                    context["env_alive_mask"] = [False] * self.num_envs
                context["task_success_mask"] = list(context.get("env_alive_mask", [False] * self.num_envs))
                return False

            if step_mask is not None:
                alive_count = sum(context["env_alive_mask"])
                print(f"[INFO] Active envs after primitive {step_idx}: {alive_count}/{self.num_envs}")
                if alive_count == 0:
                    print("[ERROR] All environments failed before finishing primitive sequence.")
                    return False

        if context.get("task_success_mask") is None:
            context["task_success_mask"] = list(context.get("env_alive_mask", [False] * self.num_envs))
        env_failure_reasons = list(context.get("env_failure_reasons", ["none"] * self.num_envs))
        if len(env_failure_reasons) != self.num_envs:
            env_failure_reasons = ["none"] * self.num_envs
        task_success_mask = list(context.get("task_success_mask", [False] * self.num_envs))
        for env_idx in range(self.num_envs):
            if task_success_mask[env_idx]:
                env_failure_reasons[env_idx] = "none"
            elif env_failure_reasons[env_idx] == "none":
                env_failure_reasons[env_idx] = "planning_failed"
        context["env_failure_reasons"] = env_failure_reasons
        return True

    def collect(self, save_video: bool = False) -> Dict:
        """
        Main workflow: Load YAML -> Configure Task -> Collect Loop
        """
        print("\n" + "="*80)
        print("STEP 1: LOADING SCENE CONFIGURATION")
        print("="*80)
        
        scene_desc, task_desc = self.rebuild_scene_desc_from_yaml()
        if not scene_desc:
            print("[ERROR] Failed to load objects from YAML.")
            return {}

        print(f"Task: {task_desc}")

        # Initialize Systems
        self._init_environment()
        
        if self.enable_motion_planning:
            use_table = (self.robot_type not in {"ur5e", "kinova_robotiq85"})
            self.motion_planner = MotionPlanner(
                robot_cfg_name=self.robot_cfg_name,
                device=self.device,
                curobo_path=self.curobo_path,
                num_envs=self.num_envs,
                use_table_obstacle=use_table,
            )
            self._init_motion_planning()
            self._init_ik_solver()

        primitive_steps = self._get_primitive_steps()
        if not primitive_steps:
            print(f"[ERROR] Task '{self.task_name}' has no primitives configured. Legacy execution has been removed.")
            return {}
        print(f"[INFO] Primitive-driven execution enabled with {len(primitive_steps)} configured steps.")
        
        if self.data_dir:
            data_dir = Path(self.data_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
            image_noise_std = self.task_config.image_noise_std if self.task_config else 0.5
            image_speckle_noise_std = self.task_config.image_speckle_noise_std if self.task_config else 0.0
            save_npy = self.save_npy
            save_camera_images = self.save_camera_images
            replace_background = False
            save_composite_images = self.save_composite_images
            save_mask_images = self.save_mask_images
            save_original_images = self.save_original_images
            cam_side_bg_path = None
            cam_wrist_bg_path = None
            sim_camera_resolution = (640, 480)
            background_image_size = (480, 480)
            if self.task_config is not None and hasattr(self.task_config, "runtime"):
                runtime_cfg = self.task_config.runtime
                save_npy = bool(runtime_cfg.save_npy)
                save_camera_images = bool(runtime_cfg.save_camera_images)
                replace_background = bool(runtime_cfg.use_background_replace)
                save_composite_images = bool(runtime_cfg.save_composite_images)
                save_mask_images = bool(runtime_cfg.save_mask_images)
                save_original_images = bool(runtime_cfg.save_original_images)
                cam_side_bg_path = runtime_cfg.cam_side_background or None
                cam_wrist_bg_path = runtime_cfg.cam_wrist_background or None
                sim_camera_resolution = tuple(runtime_cfg.sim_camera_resolution)
                background_image_size = tuple(runtime_cfg.background_image_size)

            should_save_any_output = any(
                [
                    save_npy,
                    save_camera_images,
                    save_composite_images,
                    save_mask_images,
                    save_original_images,
                ]
            )

            if should_save_any_output:
                self.recorder = EpisodeDataRecorder(
                    save_dir=data_dir,
                    save_npy=save_npy,
                    save_camera_images=save_camera_images,
                    save_composite_images=save_composite_images,
                    save_mask_images=save_mask_images,
                    save_original_images=save_original_images,
                    num_envs=self.num_envs,
                    replace_background=replace_background,
                    sim_camera_resolution=sim_camera_resolution,
                    background_image_size=background_image_size,
                    cam_side_background_path=cam_side_bg_path,
                    cam_wrist_background_path=cam_wrist_bg_path,
                    image_noise_std=image_noise_std,
                    image_speckle_noise_std=image_speckle_noise_std,
                )

        # Run Episodes
        print(f"\n[INFO] Starting collection for {self.episodes} episodes...")
        total_success = 0
        total_env_attempts = 0
        total_env_success = 0
        total_env_planning_failed = 0
        total_env_success_eval_failed = 0
        total_env_other_failed = 0
        
        try:
            with torch.no_grad():
                for ep in range(self.episodes):
                    print(f"\n{'='*60}")
                    print(f"Episode {ep+1}/{self.episodes}")
                    print(f"{'='*60}")
                    
                    self._reset_and_stabilize()

                    primitive_context = self._build_primitive_context(
                        episode_num=ep + 1,
                        scene_desc=scene_desc,
                        task_desc=task_desc,
                    )
                    success = self._run_primitives(primitive_steps, primitive_context)
                    task_success_mask = list(
                        primitive_context.get(
                            "task_success_mask",
                            primitive_context.get("env_alive_mask", [False] * self.num_envs),
                        )
                    )
                    if len(task_success_mask) != self.num_envs:
                        task_success_mask = [bool(success)] * self.num_envs
                    successful_env_count = int(sum(bool(v) for v in task_success_mask))
                    env_failure_reasons = list(primitive_context.get("env_failure_reasons", ["unknown"] * self.num_envs))
                    if len(env_failure_reasons) != self.num_envs:
                        env_failure_reasons = ["unknown"] * self.num_envs
                    for env_idx in range(self.num_envs):
                        total_env_attempts += 1
                        if bool(task_success_mask[env_idx]):
                            total_env_success += 1
                            continue
                        reason = str(env_failure_reasons[env_idx])
                        if reason == "success_eval_failed":
                            total_env_success_eval_failed += 1
                        elif reason == "planning_failed":
                            total_env_planning_failed += 1
                        else:
                            total_env_other_failed += 1
                        
                    if successful_env_count == self.num_envs:
                        status_icon, status_label = "✓", "FULL"
                    elif successful_env_count > 0:
                        status_icon, status_label = "◐", "PARTIAL"
                    else:
                        status_icon, status_label = "✗", "FAIL"

                    reason_counter = Counter(
                        str(env_failure_reasons[i])
                        for i in range(self.num_envs)
                        if not bool(task_success_mask[i])
                    )
                    reason_str = ", ".join(f"{k}={v}" for k, v in sorted(reason_counter.items())) or "none"

                    if success and successful_env_count > 0:
                        if self.recorder is not None:
                            self.recorder.save_episode(success=True, successful_envs=task_success_mask)
                        total_success += 1
                        print(f"{status_icon} Episode {ep+1} {status_label}: {successful_env_count}/{self.num_envs} envs succeeded")
                        if successful_env_count < self.num_envs:
                            print(f"    failure breakdown: {reason_str}")
                    else:
                        # Clear episode buffers to prevent memory leak on failed episodes
                        if self.recorder is not None:
                            self.recorder.episode_buffers = [None] * self.num_envs
                        print(f"{status_icon} Episode {ep+1} {status_label}: {successful_env_count}/{self.num_envs} envs succeeded")
                        print(f"    failure breakdown: {reason_str}")

                    # Cleanup temp files and force garbage collection
                    # self.vision_pipeline.cleanup_temp_files()  # Keep anygrasp_file for debug visualization
                    import gc; gc.collect()
                    torch.cuda.empty_cache()
                    
        except KeyboardInterrupt:
            print("\n[INFO] Collection interrupted by user.")
        finally:
            pass

        stats = {
            "total_episodes": self.episodes,
            "successful_episodes": total_success,
            "success_rate": total_success / self.episodes if self.episodes > 0 else 0,
            "total_env_attempts": total_env_attempts,
            "successful_envs": total_env_success,
            "env_success_rate": (total_env_success / total_env_attempts) if total_env_attempts > 0 else 0,
            "planning_failed_envs": total_env_planning_failed,
            "planning_failure_rate": (total_env_planning_failed / total_env_attempts) if total_env_attempts > 0 else 0,
            "success_eval_failed_envs": total_env_success_eval_failed,
            "success_eval_failure_rate": (
                total_env_success_eval_failed / total_env_attempts
            ) if total_env_attempts > 0 else 0,
            "other_failed_envs": total_env_other_failed,
            "other_failure_rate": (total_env_other_failed / total_env_attempts) if total_env_attempts > 0 else 0,
            "task_name": self.env_task_name,
        }
        
        print(f"\n{'='*60}")
        print(f"Collection Summary:")
        print(f"  Total episodes: {stats['total_episodes']}")
        print(f"  Successful: {stats['successful_episodes']}")
        print(f"  Episode success rate: {stats['success_rate']:.1%}")
        print(
            f"  Env success: {stats['successful_envs']}/{stats['total_env_attempts']} "
            f"({stats['env_success_rate']:.1%})"
        )
        print(
            f"  Planning failed envs: {stats['planning_failed_envs']}/{stats['total_env_attempts']} "
            f"({stats['planning_failure_rate']:.1%})"
        )
        print(
            f"  Success-eval failed envs: {stats['success_eval_failed_envs']}/{stats['total_env_attempts']} "
            f"({stats['success_eval_failure_rate']:.1%})"
        )
        if stats["other_failed_envs"] > 0:
            print(
                f"  Other failed envs: {stats['other_failed_envs']}/{stats['total_env_attempts']} "
                f"({stats['other_failure_rate']:.1%})"
            )
        print(f"{'='*60}\n")
        
        return stats

    def close(self):
        """Close environment and simulation"""
        if self.motion_planner:
            del self.motion_planner
        if hasattr(self, 'env') and self.env:
            self.env.close()
        print("✓ DataCollector closed")
