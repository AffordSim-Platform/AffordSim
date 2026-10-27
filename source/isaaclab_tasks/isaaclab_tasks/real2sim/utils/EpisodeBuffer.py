import h5py
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import cv2
from PIL import Image
from scipy.spatial.transform import Rotation as R


def replace_background_with_segmentation(
    rgb_image: np.ndarray,
    seg_rgba: np.ndarray,
    background_image: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Replace background in RGB image using semantic segmentation

    Args:
        rgb_image: RGB image from simulation, shape [H, W, C] or [C, H, W]
        seg_rgba: Semantic segmentation RGBA, shape [H, W, 4]
        background_image: Background image to use, shape [H, W, 3], RGB format

    Returns:
        Tuple of:
            - Composite image with background replaced, float in [0, 1]
            - Foreground mask, uint8 [H, W], 255=foreground, 0=background
    """
    # Convert RGB to [H, W, C] format
    if rgb_image.ndim == 3 and rgb_image.shape[0] in [1, 3, 4]:
        rgb_img = rgb_image.transpose(1, 2, 0)  # [C, H, W]  to  [H, W, C]
    else:
        rgb_img = rgb_image

    # Convert to uint8 if needed
    if rgb_img.dtype in [np.float32, np.float64] and rgb_img.max() <= 1.0:
        rgb_img = (rgb_img * 255).astype(np.uint8)
    else:
        rgb_img = rgb_img.astype(np.uint8)

    # Extract RGB from RGBA
    seg_rgb = seg_rgba[:, :, :3]

    # Create foreground mask (non-black pixels)
    foreground_mask = (seg_rgb.sum(axis=2) > 0)

    # Ensure background matches size
    H, W = rgb_img.shape[:2]
    if background_image.shape[:2] != (H, W):
        bg_resized = cv2.resize(background_image, (W, H))
    else:
        bg_resized = background_image.copy()

    # Create composite: background + foreground
    composite_img = bg_resized.copy()
    composite_img[foreground_mask] = rgb_img[foreground_mask]

    # Convert foreground mask to uint8 image (255=foreground, 0=background)
    mask_image = (foreground_mask.astype(np.uint8) * 255)

    return composite_img / 255.0, mask_image  # Return as float in [0, 1]


class EpisodeDataRecorder:

    def __init__(
        self,
        save_dir: Optional[Path] = None,
        save_npy: bool = True,
        save_camera_images: bool = True,
        save_composite_images: bool = False,
        save_mask_images: bool = False,
        save_original_images: bool = False,
        save_depth_images: bool = False,
        image_size: Tuple[int, int] = (500, 500),
        num_envs: int = 1,
        replace_background: bool = False,
        sim_camera_resolution: Tuple[int, int] = (640, 480),
        background_image_size: Tuple[int, int] = (480, 480),
        cam_side_background_path: Optional[str] = None,
        cam_wrist_background_path: Optional[str] = None,
        cam_side_background_image: Optional[np.ndarray] = None,
        wrist_cam_background_image: Optional[np.ndarray] = None,
        image_noise_std: float = 0.5,
        image_speckle_noise_std: float = 0.0,
        original_image_format: str = "png",
    ):
        if save_dir is None:
            save_dir = Path("data") / "episodes" / datetime.now().strftime("%Y%m%d_%H%M%S")

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.save_npy = save_npy
        self.save_camera_images = save_camera_images
        self.save_composite_images = save_composite_images
        self.save_mask_images = save_mask_images
        self.save_original_images = save_original_images
        self.original_image_format = original_image_format
        self.save_depth_images = save_depth_images
        self.image_size = image_size
        self.num_envs = num_envs
        self.replace_background = replace_background
        self.sim_camera_resolution = sim_camera_resolution  # (width, height) of sim camera
        self.background_image_size = background_image_size  # (width, height) of background / target output
        self.image_noise_std = image_noise_std
        self.image_speckle_noise_std = image_speckle_noise_std

        # Support both direct image injection (legacy) and path-based loading (0.8 style)
        self.cam_side_background_image = cam_side_background_image
        self.cam_wrist_background_image = wrist_cam_background_image

        if self.replace_background:
            if (
                self.cam_side_background_image is None
                and self.cam_wrist_background_image is None
                and not cam_side_background_path
                and not cam_wrist_background_path
            ):
                raise ValueError(
                    "replace_background is enabled but no background image is provided "
                    "(cam_side_background/cam_wrist_background path or image array)"
                )

            if self.cam_side_background_image is None and cam_side_background_path:
                bg_img = cv2.imread(str(cam_side_background_path))
                if bg_img is None:
                    raise FileNotFoundError(f"Side background image not found: {cam_side_background_path}")
                self.cam_side_background_image = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
                print(f"  Loaded side background image from: {cam_side_background_path}")
            elif self.cam_side_background_image is None:
                print(f"  [WARN] cam_side_background not configured, side camera background replacement disabled")
            else:
                print("  Loaded side background image from provided array")

            if self.cam_wrist_background_image is None and cam_wrist_background_path:
                bg_img = cv2.imread(str(cam_wrist_background_path))
                if bg_img is None:
                    raise FileNotFoundError(f"Wrist background image not found: {cam_wrist_background_path}")
                self.cam_wrist_background_image = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
                print(f"  Loaded wrist background image from: {cam_wrist_background_path}")
            elif self.cam_wrist_background_image is None:
                print(f"  [WARN] cam_wrist_background not configured, wrist camera background replacement disabled")
            else:
                print("  Loaded wrist background image from provided array")

            if self.cam_wrist_background_image is not None:
                h, w = self.cam_wrist_background_image.shape[:2]
                crop_height = int(h * 2 / 3)  # Keep top 2/3
                bg_img_cropped = self.cam_wrist_background_image[:crop_height, :]
                self.cam_wrist_background_image = cv2.resize(bg_img_cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        # Current episode buffer - one per environment
        self.episode_buffers = [None] * num_envs
        self.current_episode_num = None
        # Track previous EE quaternion per env for sign-continuity fix
        self._prev_ee_quat: Dict[int, np.ndarray] = {}
        self.aggregate_json_path = self.save_dir / "episodes_metadata.json"
        self._aggregate_json_data: Dict[str, Any] = {"episodes": []}

        print(f" EpisodeDataRecorder initialized")
        print(f"  Save directory: {self.save_dir}")
        print(f"  Save npy: {save_npy}")
        print(f"  Save camera images: {save_camera_images}")
        print(f"  Save composite images: {save_composite_images}")
        print(f"  Save mask images: {save_mask_images}")
        print(f"  Save original images: {save_original_images}")
        print(f"  Save depth images: {save_depth_images}")
        print(f"  Replace background: {replace_background}")
        print(f"  Image noise std: {image_noise_std}")
        print(f"  Image speckle noise std: {image_speckle_noise_std}")

    def start_episode(
        self,
        episode_num: int,
        metadata: Optional[Dict[str, Any]] = None
    ):  
        self.current_episode_num = episode_num
        self.current_step_count = 0  # Track step number within episode
        self._prev_ee_quat = {}

        # Initialize buffer for each environment
        for env_idx in range(self.num_envs):
            self.episode_buffers[env_idx] = {
                "observations": [],
                "actions": [],
                "terminations": [],
                "truncations": [],
                "metadata": metadata or {},
                "env_idx": env_idx,
            }
    
    def add_step(
        self,
        obs: Dict[str, Any],
        action: torch.Tensor,
        termination,
        truncation,
        info: Optional[Dict] = None,
        step_index: Optional[int] = None,
        plan_lengths: Optional[List[int]] = None,
        replace_background: Optional[bool] = None,
    ):
        """Add a step of data for all environments"""
        if self.episode_buffers[0] is None:
            raise RuntimeError("Must call start_episode() before add_step()")

        # Convert action to numpy
        action_data = action.cpu().numpy() if isinstance(action, torch.Tensor) else np.array(action)

        # Check if action batch size matches num_envs
        if action_data.shape[0] != self.num_envs:
            actual_num_envs = min(action_data.shape[0], self.num_envs)
        else:
            actual_num_envs = self.num_envs

        # Keep backward compatibility with per-call switch; default to recorder-level config.
        use_background_replacement = self.replace_background if replace_background is None else replace_background

        if use_background_replacement:
            if self.save_original_images or self.save_depth_images:
                sim_w, sim_h = self.sim_camera_resolution
                self._save_original_images(obs, actual_num_envs, image_height=sim_h, image_width=sim_w)
            if self.cam_side_background_image is not None:
                obs = self._apply_background_replacement(obs, "external_cam_side", "semantic_segmentation_side", self.cam_side_background_image)
            if self.cam_wrist_background_image is not None:
                obs = self._apply_background_replacement(obs, "wrist_cam", "semantic_segmentation_wrist", self.cam_wrist_background_image)
        else:
            # Save original RGB/depth images without background replacement
            if self.save_original_images or self.save_depth_images:
                sim_w, sim_h = self.sim_camera_resolution
                self._save_original_images(obs, actual_num_envs, image_height=sim_h, image_width=sim_w)
        
        # Increment step counter
        self.current_step_count += 1

        # RGB/composite/mask files are already written above. Image-only runs
        # need no in-memory trajectory (which otherwise also retains depth and
        # segmentation observations for every step).
        if not self.save_npy and not self.save_camera_images:
            return
        
        # Split batch data into per-environment data
        for env_idx in range(actual_num_envs):
            # Skip if this environment's trajectory is complete
            if step_index is not None and plan_lengths is not None:
                if env_idx >= len(plan_lengths) or step_index >= plan_lengths[env_idx]:
                    continue  # Don't record steps beyond this environment's trajectory
            # Extract data for this environment
            obs_env = self._extract_env_obs(obs, env_idx)
            action_env = action_data[env_idx]
            
            # Handle termination/truncation (can be bool or tensor)
            if isinstance(termination, torch.Tensor):
                term_env = termination[env_idx].item()
            elif isinstance(termination, (list, np.ndarray)):
                term_env = termination[env_idx]
            else:
                term_env = termination
                
            if isinstance(truncation, torch.Tensor):
                trunc_env = truncation[env_idx].item()
            elif isinstance(truncation, (list, np.ndarray)):
                trunc_env = truncation[env_idx]
            else:
                trunc_env = truncation
            
            # Convert observation to numpy
            obs_data = self._convert_obs_to_numpy(obs_env)

            # Fix quaternion sign discontinuity in arm_ee_pose.
            # obs_data may be nested (e.g. obs_data["policy"]["arm_ee_pose"]),
            # so search one level deep.
            ee_pose = None
            if "arm_ee_pose" in obs_data:
                ee_pose = obs_data["arm_ee_pose"]
            elif isinstance(obs_data.get("policy"), dict) and "arm_ee_pose" in obs_data["policy"]:
                ee_pose = obs_data["policy"]["arm_ee_pose"]
            if ee_pose is not None and isinstance(ee_pose, np.ndarray) and ee_pose.shape[-1] >= 7:
                quat = ee_pose[3:7].copy()
                if env_idx in self._prev_ee_quat:
                    if np.dot(quat, self._prev_ee_quat[env_idx]) < 0:
                        quat = -quat
                        ee_pose[3:7] = quat
                self._prev_ee_quat[env_idx] = quat.copy()

            # Add to buffer
            self.episode_buffers[env_idx]["observations"].append(obs_data)
            self.episode_buffers[env_idx]["actions"].append(action_env)
            self.episode_buffers[env_idx]["terminations"].append(term_env)
            self.episode_buffers[env_idx]["truncations"].append(trunc_env)
    
    def _extract_env_obs(self, obs: Dict[str, Any], env_idx: int) -> Dict[str, Any]:
        """Extract observation for a specific environment from batch"""
        obs_env = {}
        for key, value in obs.items():
            if isinstance(value, torch.Tensor):
                # Check if tensor has batch dimension
                if value.shape[0] == self.num_envs:
                    # Extract env_idx from batch
                    obs_env[key] = value[env_idx]
                else:
                    # Not batched or different batch size, keep as is
                    obs_env[key] = value
            elif isinstance(value, dict):
                # Recursively extract from nested dict
                obs_env[key] = self._extract_env_obs(value, env_idx)
            elif isinstance(value, (list, np.ndarray)):
                # Check if array has batch dimension
                if isinstance(value, np.ndarray) and len(value) == self.num_envs:
                    obs_env[key] = value[env_idx]
                elif isinstance(value, list) and len(value) == self.num_envs:
                    obs_env[key] = value[env_idx]
                else:
                    # Not batched or different batch size, keep as is
                    obs_env[key] = value
            else:
                obs_env[key] = value
        return obs_env

    def _convert_obs_to_numpy(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Convert observation dict to numpy arrays"""
        obs_data = {}
        
        for key, value in obs.items():
            if isinstance(value, torch.Tensor):
                # Convert to numpy
                numpy_value = value.detach().cpu().numpy().copy()
                
                is_image = self._is_image(numpy_value) or self._is_image_key(key)
                if is_image:
                    numpy_value = self._resize_image(numpy_value, key)
                    if (numpy_value.dtype in (np.float32, np.float64)
                            and 'semantic' not in key):
                        if self.image_noise_std > 0:
                            numpy_value = self._add_sensor_noise(numpy_value)
                        if self.image_speckle_noise_std > 0:
                            numpy_value = self._add_speckle_noise(numpy_value)
                
                obs_data[key] = numpy_value
                
            elif isinstance(value, dict):
                # Recursively convert nested dicts
                obs_data[key] = self._convert_obs_to_numpy(value)
                
            elif isinstance(value, (list, tuple)):
                obs_data[key] = np.array(value, copy=True)
                
            else:
                obs_data[key] = value
        
        return obs_data

    def _is_image_key(self, key: str) -> bool:
        """Check if key name indicates image data"""
        image_keywords = ['cam', 'rgb', 'depth', 'image', 'distance']
        key_lower = key.lower()
        return any(kw in key_lower for kw in image_keywords)

    def _add_sensor_noise(self, image: np.ndarray) -> np.ndarray:
        """Add Gaussian sensor noise to a float image in [0, 1] range"""
        noise = np.random.normal(0, self.image_noise_std, image.shape).astype(image.dtype)
        return np.clip(image + noise, 0.0, 1.0)

    def _add_speckle_noise(self, image: np.ndarray) -> np.ndarray:
        """Add speckle (multiplicative) noise to a float image in [0, 1] range.

        output = image + image * noise, where noise ~ N(0, std)
        """
        noise = np.random.normal(0, self.image_speckle_noise_std, image.shape).astype(image.dtype)
        return np.clip(image + image * noise, 0.0, 1.0)
    
    def _is_image(self, array: np.ndarray) -> bool:
        """Check if array is an image (has spatial dimensions)"""
        #  (H, W, C)  (B, H, W, C)  (C, H, W)
        if len(array.shape) >= 3:
            if len(array.shape) == 3:  # (H, W, C) or (C, H, W)
                return True
            elif len(array.shape) == 4:  # (B, H, W, C) or (B, C, H, W)
                return True
        return False
    
    def _resize_image(self, image: np.ndarray, key: str = "") -> np.ndarray:
        if not self.save_camera_images:
            return image
        
        original_shape = image.shape
        
        if len(image.shape) == 4:  # (1, H, W, C) or (1, C, H, W)
            image = image[0]
        
        if len(image.shape) == 3:
            #  (H, W, C)  (C, H, W)
            if image.shape[0] in [1, 3, 4]:  # (C, H, W) - channels first
                # Convert to (H, W, C)
                image = np.transpose(image, (1, 2, 0))
                channels_first = True
            else:  # (H, W, C) - channels last
                channels_first = False
            
            if image.shape[:2] != self.image_size[::-1]:  # 
                try:
                    # cv2.resize  (width, height)
                    resized = cv2.resize(
                        image, 
                        self.image_size,  # (width, height)
                        interpolation=cv2.INTER_LINEAR
                    )
                    
                    # Convert back to original format if needed
                    if channels_first:
                        resized = np.transpose(resized, (2, 0, 1))
                    
                    if len(original_shape) == 4:
                        resized = resized[np.newaxis, ...]  # Add batch dim back
                    
                    # if key:
                        # print(f"    Resized {key}: {original_shape}  to  {resized.shape}")
                    
                    return resized
                    
                except Exception as e:
                    print(f"      Failed to resize {key}: {e}, keeping original")
                    return image
        
        return image
    
    def save_episode(
        self,
        success: bool,
        successful_envs: Optional[List[bool]] = None,
    ) -> List[Path]:
        """Save episode data for all environments
        
        Args:
            success: Overall success flag
            successful_envs: List of boolean flags indicating which environments succeeded.
                           If None, save all environments.
        """
        saved_paths = []

        if not self.save_npy and not self.save_camera_images:
            self.episode_buffers = [None] * self.num_envs
            self.current_episode_num = None
            return saved_paths
        
        # If no successful_envs list provided, assume all environments succeeded
        if successful_envs is None:
            successful_envs = [True] * self.num_envs
        
        for env_idx in range(self.num_envs):
            # Skip environments that didn't succeed
            if not successful_envs[env_idx]:
                print(f"   Skipping env {env_idx} - planning failed")
                continue
            episode_buffer = self.episode_buffers[env_idx]
            if episode_buffer is None:
                continue
            num_steps_raw = len(episode_buffer["observations"])
        
            obs_dict_full = self._flatten_observations_for_env(episode_buffer)
            
            ee_pose_key = 'obs_policy_arm_ee_pose'  # Adjust this key based on your actual observation structure
            ee_delta = None
            # print(f"obs_dict_full keys: {list(obs_dict_full.keys())}")
            ee_poses = obs_dict_full[ee_pose_key]  # Shape: [T, 7] - [x, y, z, qw, qx, qy, qz]
            ee_delta_joint_action = self._compute_ee_pose_delta(ee_poses)  # Shape: [T-1, 6] - [dx, dy, dz, droll, dpitch, dyaw]
            raw_joint_actions = np.array(episode_buffer['actions'])  # Shape: [T, 8] - 7 joints + 1 gripper
            ee_delta_gripper_action = raw_joint_actions[:-1, -1:]
            ee_delta_action = np.concatenate([ee_delta_joint_action, ee_delta_gripper_action], axis=-1)  # Add gripper delta if needed
            # actions will be aligned with observations after data alignment below
            actions = ee_delta_action
            last_actions = np.concatenate([np.zeros((self.num_envs, ee_delta_action.shape[1])), ee_delta_action], axis=0)
            
            raw_joint_actions = raw_joint_actions[:-1]
            episode_buffer['actions'] = actions
            # episode_buffer['observations']['obs_policy_last_action'] = last_actions

            # Now perform data alignment
            # episode_buffer['actions'] = episode_buffer['actions'][1:]
            episode_buffer['observations'] = episode_buffer['observations'][:-1]
            episode_buffer['terminations'] = episode_buffer['terminations'][:-1]
            episode_buffer['truncations'] = episode_buffer['truncations'][:-1]
            # episode_buffer['observations']['obs_policy_last_action'] = last_actions

            actions_arr = np.array(episode_buffer['actions'])
            ee_action_norm = np.abs(actions_arr[:, :6]).sum(axis=1)  # sum of absolute EE deltas
            active_mask = ee_action_norm > 1e-8  # True for steps where EE actually moved
            num_idle = int((~active_mask).sum())

            if num_idle > 0:
                # Filter actions
                episode_buffer['actions'] = actions_arr[active_mask]
                # Filter observations (already aligned 1:1 with actions)
                episode_buffer['observations'] = [obs for obs, keep in zip(episode_buffer['observations'], active_mask) if keep]
                episode_buffer['terminations'] = [t for t, keep in zip(episode_buffer['terminations'], active_mask) if keep]
                episode_buffer['truncations'] = [t for t, keep in zip(episode_buffer['truncations'], active_mask) if keep]
                raw_joint_actions = raw_joint_actions[active_mask]
                last_actions = np.concatenate([np.zeros((self.num_envs, actions_arr.shape[1])), episode_buffer['actions']], axis=0)
                print(f"  Env {env_idx} - Removed {num_idle} idle steps (EE delta == 0)")
            
            num_steps = len(episode_buffer['actions'])
            print(f"  Env {env_idx} - Aligned data: {num_steps_raw} raw steps  to  {num_steps} aligned steps")
            
            save_dict = {}
            episode_dir = self.save_dir / f"episode_{((self.current_episode_num-1)*self.num_envs+env_idx):04d}.npy"
            
            actions = np.array(episode_buffer['actions'])
            save_dict['obs_policy_actions'] = actions
            print(f"  Env {env_idx} - actions: shape={actions.shape}, dtype={actions.dtype}")
            
            # Flatten observations from aligned episode_buffer
            obs_dict = self._flatten_observations_for_env(episode_buffer)
            obs_dict['obs_policy_lastactions'] = last_actions[:-1]
            obs_dict['obs_policy_joint_actions'] = raw_joint_actions  # Shape: [T, 8] - 7 joints + 1 gripper
            
            # Save camera images to separate folders (these are composite images with background replaced)
            if self.save_camera_images:
                self._save_camera_images(obs_dict, env_idx)
            
            for key, value in obs_dict.items():
                save_dict[str(key)] = value
                print(f"  Env {env_idx} - {str(key)}: shape={value.shape}, dtype={value.dtype}")
            
            if ee_delta is not None:
                save_dict['ee_pose_delta'] = ee_delta
                print(f"  Env {env_idx} - ee_pose_delta: shape={ee_delta.shape}, dtype={ee_delta.dtype}")
            
            if self.save_npy:
                np.save(episode_dir, save_dict, allow_pickle=True)
                print(f"   Saved episode {self.current_episode_num} env {env_idx} to {episode_dir.name}")
            print(f"    Steps: {num_steps}")

            self._append_episode_state_json_entry(
                success=success,
                num_steps=num_steps,
                npy_path=episode_dir,
                env_idx=env_idx,
                metadata=episode_buffer.get("metadata", {}),
            )
            
            if self.save_npy:
                saved_paths.append(episode_dir)
        
        # Clear buffers
        self.episode_buffers = [None] * self.num_envs
        self.current_episode_num = None
        
        return saved_paths

    @staticmethod
    def _to_json_compatible(value: Any) -> Any:
        """Convert nested values (numpy/torch/etc.) into JSON-serializable types."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(k): EpisodeDataRecorder._to_json_compatible(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [EpisodeDataRecorder._to_json_compatible(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
        if hasattr(value, "tolist"):
            try:
                return value.tolist()
            except Exception:
                pass
        return str(value)

    def _append_episode_state_json_entry(
        self,
        success: bool,
        num_steps: int,
        npy_path: Path,
        env_idx: int,
        metadata: Dict[str, Any],
    ) -> None:
        """Append current episode/env metadata into a single aggregate JSON file."""
        metadata = metadata or {}
        initial_env_states = metadata.get("initial_env_states", [])
        env_initial_state = None
        if isinstance(initial_env_states, list) and 0 <= env_idx < len(initial_env_states):
            env_initial_state = initial_env_states[env_idx]

        episode_entry = {
            # Keep only the 4-digit episode id from file name (episode_XXXX.npy  to  XXXX).
            "episode_id": self._extract_episode_id(npy_path),
            # Keep only object poses; positions are in meters relative to env origin,
            # consistent with YAML init_pos/init_rot convention.
            "objects": self._extract_object_poses_in_m(env_initial_state),
        }
        self._aggregate_json_data["episodes"].append(episode_entry)

        with open(self.aggregate_json_path, "w", encoding="utf-8") as f:
            json.dump(self._aggregate_json_data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _extract_episode_id(npy_path: Path) -> str:
        """Extract 4-digit episode id from file name episode_XXXX.npy."""
        stem = npy_path.stem
        if stem.startswith("episode_"):
            suffix = stem.split("episode_", 1)[1]
            if suffix.isdigit():
                return suffix.zfill(4)
        return stem

    @staticmethod
    def _extract_object_poses_in_m(env_initial_state: Any) -> Dict[str, Dict[str, Any]]:
        """Extract object init_pos/init_rot in meters from captured env state."""
        if not isinstance(env_initial_state, dict):
            return {}

        env_origin = env_initial_state.get("env_origin")
        origin_arr = None
        if isinstance(env_origin, (list, tuple, np.ndarray)) and len(env_origin) >= 3:
            origin_arr = np.array(env_origin[:3], dtype=float)

        objects = env_initial_state.get("objects", {})
        if not isinstance(objects, dict):
            return {}

        result: Dict[str, Dict[str, Any]] = {}
        for name, state in objects.items():
            if not isinstance(state, dict):
                continue

            pos_world = state.get("position_world")
            quat_wxyz = state.get("orientation_wxyz_world")

            position_m = None
            if isinstance(pos_world, (list, tuple, np.ndarray)) and len(pos_world) >= 3:
                pos_arr = np.array(pos_world[:3], dtype=float)
                # Convert world position to env-relative meters to match YAML init_pos.
                if origin_arr is not None:
                    pos_arr = pos_arr - origin_arr
                position_m = pos_arr.tolist()

            orientation = None
            if isinstance(quat_wxyz, (list, tuple, np.ndarray)) and len(quat_wxyz) >= 4:
                orientation = np.array(quat_wxyz[:4], dtype=float).tolist()

            result[str(name)] = {
                "init_pos": position_m,
                "init_rot": orientation,
            }

        return result
    
    def _compute_ee_pose_delta(self, ee_poses: np.ndarray) -> np.ndarray:
        """Compute relative end-effector pose changes (delta)
        
        Args:
            ee_poses: Array of end-effector poses [T, 7] where each pose is [x, y, z, qw, qx, qy, qz]
        
        Returns:
            Array of pose deltas [T-1, 6] where each delta is [dx, dy, dz, droll, dpitch, dyaw] in radians
        """
        if len(ee_poses) < 2:
            return np.zeros((0, 6))
        
        deltas = []
        for i in range(len(ee_poses) - 1):
            # Current and next poses
            pos_curr = ee_poses[i, :3]
            quat_curr = ee_poses[i, 3:]  # [qw, qx, qy, qz]
            
            pos_next = ee_poses[i + 1, :3]
            quat_next = ee_poses[i + 1, 3:]
            
            # Position delta (straightforward)
            delta_pos = pos_next - pos_curr
            
            # Rotation delta (relative rotation)
            # Convert quaternions to rotation objects
            # scipy expects [x, y, z, w] format
            rot_curr = R.from_quat([quat_curr[1], quat_curr[2], quat_curr[3], quat_curr[0]])
            rot_next = R.from_quat([quat_next[1], quat_next[2], quat_next[3], quat_next[0]])
            
            # Compute relative rotation: rot_delta = rot_next * rot_curr^(-1)
            rot_delta = rot_next * rot_curr.inv()
            
            # Convert to euler angles (roll, pitch, yaw) in radians
            delta_euler = rot_delta.as_euler('xyz', degrees=False)
            # delta_rotvec = rot_delta.as_rotvec()
            
            # Combine into 6D delta [dx, dy, dz, droll, dpitch, dyaw]
            # Combine into 6D delta [dx, dy, dz, ax, ay, az] roll, pitch, yaw
            delta = np.concatenate([delta_pos, delta_euler])
            deltas.append(delta)
        
        return np.array(deltas)
    
    def _flatten_observations_for_env(self, episode_buffer: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Flatten observations for a single environment"""
        flattened = {}
        
        if not episode_buffer['observations']:
            return flattened
        
        first_obs = episode_buffer['observations'][0]
    
        def flatten_recursive(obs_dict, prefix="obs", key_path=None):
            if key_path is None:
                key_path = []
            
            for key, value in obs_dict.items():
                # String name for saving
                full_key = f"{prefix}_{key}"
                # Actual path for navigation
                current_key_path = key_path + [key]
                
                if isinstance(value, dict):
                    # Recursively process nested dict
                    flatten_recursive(value, full_key, current_key_path)
                
                elif isinstance(value, np.ndarray):
                    # Do not store semantic segmentation images into episode npy.
                    # Keep them only for online processing (e.g., background replacement).
                    if full_key in (
                        "obs_policy_semantic_segmentation_side",
                        "obs_policy_semantic_segmentation_wrist",
                    ):
                        continue
                    # Check if we should save this observation
                    is_image = len(value.shape) >= 3
                    if is_image and not self.save_camera_images:
                        continue
                    data_list = []
                    for obs_step in episode_buffer['observations']:
                        # Navigate using actual key path
                        data = obs_step
                        for k in current_key_path:
                            data = data[k]
                        data_list.append(data)
                    
                    # Stack into [T, ...]
                    data_array = np.stack(data_list, axis=0)
                    flattened[full_key] = data_array
                        
        flatten_recursive(first_obs)
        return flattened

    def _save_json_summary(
        self,
        episode_num: int,
        success: bool,
        num_steps: int,
        npz_path: Path,
        env_idx: int,
        metadata: Dict[str, Any],
    ):
        """Save a JSON summary file for easy inspection"""
        summary = {
            'episode_num': episode_num,
            'env_idx': env_idx,
            'success': success,
            'num_steps': num_steps,
            'timestamp': datetime.now().isoformat(),
            'data_file': 'data.npy',
            'metadata': metadata,
        }
        
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                summary[key] = value
            elif isinstance(value, dict):
                summary[key] = value
        
        json_path = npz_path / 'summary.json'
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    @staticmethod
    def _resize_bg_no_distortion(bg_img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        """Resize background image to (target_h, target_w) without distortion.

        Scales to fully cover the target area (crop-to-fit), then center-crops.
        """
        h, w = bg_img.shape[:2]
        scale = max(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(bg_img, (new_w, new_h), interpolation=interp)
        top = (new_h - target_h) // 2
        left = (new_w - target_w) // 2
        return resized[top:top + target_h, left:left + target_w]

    def _apply_background_replacement(
        self,
        obs: Dict[str, Any],
        rgb_key: str,
        semantic_key: str,
        background_image: np.ndarray,
    ) -> Dict[str, Any]:
        """Apply background replacement using semantic segmentation
        
        Args:
            obs: Observation dict
            rgb_key: Key for RGB data in obs['policy']
            semantic_key: Key for semantic segmentation in obs['policy']
            background_image: Background image to use
        
        Returns:
            Modified observation dict with background replaced
        """
        if 'policy' not in obs or rgb_key not in obs['policy'] or semantic_key not in obs['policy']:
            return obs
        
        rgb = obs['policy'][rgb_key]
        seg = obs['policy'][semantic_key]
        num_envs = rgb.shape[0]
        
        # Sim camera resolution from YAML (for reshape)
        sim_w, sim_h = self.sim_camera_resolution
        # Background / target output size from YAML
        bg_w, bg_h = self.background_image_size

        # Process each environment
        modified_rgb_list = []
        for env_idx in range(num_envs):
            # Convert to numpy
            rgb_np = rgb[env_idx].cpu().numpy() if isinstance(rgb[env_idx], torch.Tensor) else rgb[env_idx]
            seg_np = seg[env_idx].cpu().numpy() if isinstance(seg[env_idx], torch.Tensor) else seg[env_idx]

            # Process RGB: [1, 3, H, W]  to  [3, H, W]
            if rgb_np.ndim == 1:
                rgb_img = rgb_np.reshape(3, sim_h, sim_w)
            else:
                rgb_img = rgb_np.squeeze() if rgb_np.ndim == 4 else rgb_np

            # Process seg: [1, 4, H, W]  to  [H, W, 4]
            if seg_np.ndim == 1:
                seg_rgba = seg_np.reshape(4, sim_h, sim_w).transpose(1, 2, 0).astype(np.uint8)
            else:
                seg_rgba = seg_np.squeeze() if seg_np.ndim == 4 else seg_np
                if seg_rgba.ndim == 3 and seg_rgba.shape[0] == 4:
                    seg_rgba = seg_rgba.transpose(1, 2, 0).astype(np.uint8)

            # Resize sim images to background size (no distortion: scale + center crop)
            if (sim_h, sim_w) != (bg_h, bg_w):
                # rgb_img: [C, H, W]  to  [H, W, C] for resize, then back
                rgb_hwc = rgb_img.transpose(1, 2, 0) if rgb_img.ndim == 3 and rgb_img.shape[0] in [1, 3, 4] else rgb_img
                rgb_hwc = self._resize_bg_no_distortion(rgb_hwc, bg_w, bg_h)
                rgb_img = rgb_hwc.transpose(2, 0, 1)  # back to [C, H, W]
                seg_rgba = self._resize_bg_no_distortion(seg_rgba, bg_w, bg_h)

            # Apply background replacement (now both are at background size)
            composite_img, mask_img = replace_background_with_segmentation(rgb_img, seg_rgba, background_image)

            # Save composite image and mask
            if self.save_composite_images:
                self._save_composite_image(composite_img, rgb_key, env_idx)
            if self.save_mask_images:
                self._save_mask_image(mask_img, rgb_key, env_idx)
            
            # Convert back to original format: [H, W, 3]  to  [3, H, W]  to  flatten if needed
            composite_img_chw = composite_img.transpose(2, 0, 1)  # [H, W, 3]  to  [3, H, W]
            
            if rgb[env_idx].ndim == 1:
                # If original was flattened, flatten the composite too
                composite_img_final = composite_img_chw.flatten()
            else:
                # Keep as [3, H, W]
                composite_img_final = composite_img_chw
            
            composite_tensor = torch.from_numpy(composite_img_final).to(rgb.device)
            modified_rgb_list.append(composite_tensor)
        
        # Update observation
        obs['policy'][rgb_key] = torch.stack(modified_rgb_list, dim=0)
        
        return obs
    
    def _save_composite_image(
        self,
        composite_img: np.ndarray,
        camera_name: str,
        env_idx: int,
    ) -> None:
        """Save composite image to disk
        
        Args:
            composite_img: Composite image [H, W, 3] or [3, H, W]
            camera_name: Camera name for organizing saved images
            env_idx: Environment index
        """
        # Create directory
        composite_dir = self.save_dir / "composite_images" / camera_name / f"env{env_idx}_ep{self.current_episode_num}"
        composite_dir.mkdir(parents=True, exist_ok=True)
        # print(f"[INFO] Saving composite image to: {composite_dir}")
        
        # Convert to [H, W, 3] if needed
        if composite_img.ndim == 3 and composite_img.shape[0] in [1, 3, 4]:
            img_to_save = composite_img.transpose(1, 2, 0)
        else:
            img_to_save = composite_img
        
        # Convert to uint8 if needed
        if img_to_save.dtype in [np.float32, np.float64]:
            if img_to_save.max() <= 1.0:
                img_to_save = (img_to_save * 255).astype(np.uint8)
            else:
                img_to_save = img_to_save.astype(np.uint8)
        
        # Save image
        filename = composite_dir / f"step_{self.current_step_count:04d}.png"
        cv2.imwrite(str(filename), cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR))

    def _save_mask_image(
        self,
        mask_img: np.ndarray,
        camera_name: str,
        env_idx: int,
    ) -> None:
        """Save foreground mask image to disk

        Args:
            mask_img: Foreground mask [H, W], uint8, 255=foreground, 0=background
            camera_name: Camera name for organizing saved images
            env_idx: Environment index
        """
        mask_dir = self.save_dir / "mask_images" / camera_name / f"env{env_idx}_ep{self.current_episode_num}"
        mask_dir.mkdir(parents=True, exist_ok=True)

        filename = mask_dir / f"step_{self.current_step_count:04d}.png"
        cv2.imwrite(str(filename), mask_img)

    def _save_original_images(
        self,
        obs: Dict[str, Any],
        num_envs: int,
        image_height: int = 480, # 480
        image_width: int = 640, # 480
    ) -> None:
        """Save original images"""
        if 'policy' not in obs:
            return
        
        camera_keys = [
            key for key in obs['policy'].keys() 
            if not key.startswith('semantic_') and self._is_image_tensor(obs['policy'][key])
        ]
        
        for img_key in camera_keys:
            img_data = obs['policy'][img_key]
            is_depth = 'depth' in img_key.lower() or 'distance' in img_key.lower()
            if is_depth and not self.save_depth_images:
                continue
            if not is_depth and not self.save_original_images:
                continue
            
            for env_idx in range(num_envs):
                # Convert to numpy
                if isinstance(img_data[env_idx], torch.Tensor):
                    img_np = img_data[env_idx].cpu().numpy()
                else:
                    img_np = img_data[env_idx]
                
                if is_depth:
                    depth_dir = self.save_dir / "depth_images" / img_key / f"env{env_idx}_ep{self.current_episode_num}"
                    depth_dir.mkdir(parents=True, exist_ok=True)
                    filename = depth_dir / f"step_{self.current_step_count:04d}"
                    depth_img = img_np.reshape(image_height, image_width)  # 
                    depth_mm = (depth_img * 1000).astype(np.uint16)
                    cv2.imwrite(str(filename) + '.png', depth_mm)
                else:
                    original_dir = self.save_dir / "original_images" / img_key / f"env{env_idx}_ep{self.current_episode_num}"
                    original_dir.mkdir(parents=True, exist_ok=True)
                    filename = original_dir / f"step_{self.current_step_count:04d}"
                    # RGB png
                    img_to_save = img_np.reshape(3, image_height, image_width).transpose(1, 2, 0)
                    if img_to_save.max() <= 1.0:
                        img_to_save = (img_to_save * 255).astype(np.uint8)
                    else:
                        img_to_save = img_to_save.astype(np.uint8)
                    if self.original_image_format == "webp_lossless":
                        Image.fromarray(img_to_save).save(
                            str(filename) + '.webp', format='WEBP', lossless=True, method=0, exact=True
                        )
                    elif self.original_image_format == "png":
                        if not cv2.imwrite(str(filename) + '.png', cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)):
                            raise OSError(f"Could not save RGB frame: {filename}.png")
                    else:
                        raise ValueError(f"Unsupported original_image_format: {self.original_image_format}")
    
    def _is_image_tensor(self, tensor) -> bool:
        """Check if a tensor looks like image data"""
        if isinstance(tensor, torch.Tensor):
            shape = tensor.shape
        elif isinstance(tensor, np.ndarray):
            shape = tensor.shape
        else:
            return False
        
        # Image tensors typically have shape [B, C, H, W] or [B, H, W, C] or flattened
        if len(shape) >= 2:
            # Check for typical image dimensions
            if len(shape) == 4:  # [B, C, H, W] or [B, H, W, C]
                return shape[1] in [1, 3, 4] or shape[3] in [1, 3, 4]
            elif len(shape) == 3:  # [B, H*W*C] flattened or [C, H, W]
                return True
            elif len(shape) == 2:  # [B, H*W*C] flattened
                # Check if the last dimension could be a flattened image
                return shape[1] > 100  # Arbitrary threshold for image size
        return False

    def _save_camera_images(
        self,
        obs_dict: Dict[str, np.ndarray],
        env_idx: int,
    ) -> None:
        """Save camera images from observations to separate folders
        
        Args:
            obs_dict: Flattened observation dictionary containing image data
            env_idx: Environment index
        """
        # Camera keys to save
        camera_keys = {
            'obs_policy_external_cam_side': 'external_cam_side',
            'obs_policy_wrist_cam': 'wrist_cam',
            'obs_policy_external_cam_front': 'external_cam_front',
        }
        
        for obs_key, camera_name in camera_keys.items():
            if obs_key not in obs_dict:
                continue
            
            images = obs_dict[obs_key]  # Shape: [T, ...] where T is number of timesteps
            
            # Create directory for this camera
            camera_dir = self.save_dir / "camera_images" / camera_name / f"env{env_idx}_ep{self.current_episode_num}"
            camera_dir.mkdir(parents=True, exist_ok=True)
            
            # Save each timestep
            for step_idx, img_data in enumerate(images):
                try:
                    # Convert to numpy if needed
                    if isinstance(img_data, torch.Tensor):
                        img_np = img_data.cpu().numpy()
                    else:
                        img_np = img_data
                    
                    # Handle different image formats
                    if img_np.ndim == 1:
                        img_np = self._reshape_flat_rgb_image(img_np)
                    elif img_np.ndim == 3:
                        if img_np.shape[0] in [1, 3, 4]:
                            # Channel-first [C, H, W]  to  [H, W, C]
                            img_np = img_np.transpose(1, 2, 0)
                        # else: already [H, W, C]
                    
                    # Convert to uint8 if needed
                    if img_np.dtype in [np.float32, np.float64]:
                        if img_np.max() <= 1.0:
                            img_np = (img_np * 255).astype(np.uint8)
                        else:
                            img_np = img_np.astype(np.uint8)
                    else:
                        img_np = img_np.astype(np.uint8)
                    
                    # Save image
                    filename = camera_dir / f"step_{step_idx:04d}.png"
                    cv2.imwrite(str(filename), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
                    
                except Exception as e:
                    print(f" Failed to save {camera_name} image at step {step_idx}: {e}")
            
            print(f"   Saved {len(images)} {camera_name} images to {camera_dir.relative_to(self.save_dir)}")

    def _reshape_flat_rgb_image(self, img_np: np.ndarray) -> np.ndarray:
        """Reshape a flattened RGB image using configured resolutions.

        Priority:
        1. background_image_size when background replacement is enabled
        2. sim_camera_resolution
        3. square-image fallback for older datasets
        """
        flat_size = img_np.shape[0]
        candidate_sizes = []

        if self.replace_background:
            candidate_sizes.append(self.background_image_size)
        candidate_sizes.append(self.sim_camera_resolution)

        for width, height in candidate_sizes:
            if width * height * 3 == flat_size:
                return img_np.reshape(3, height, width).transpose(1, 2, 0)

        side = int(np.sqrt(flat_size / 3))
        if side * side * 3 == flat_size:
            return img_np.reshape(3, side, side).transpose(1, 2, 0)

        raise ValueError(
            f"Cannot infer RGB image shape from flat size {flat_size}. "
            f"Tried background_image_size={self.background_image_size} and "
            f"sim_camera_resolution={self.sim_camera_resolution}."
        )
