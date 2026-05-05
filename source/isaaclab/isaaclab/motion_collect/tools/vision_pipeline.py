"""
Vision perception pipeline module

Contains camera data capture, BBox generation, AnyGrasp invocation and grasp pose conversion.
Path: isaaclab/motion_collect/vision_pipeline.py

Improvements:
1. Simplified label parsing logic based on YAML format (labels like "banana", "bowl", "can")
2. Removed redundant candidate name construction logic
"""

import json
import subprocess
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple, List, Any
from PIL import Image
from scipy.spatial.transform import Rotation

# Get current file path
current_data_path = Path(__file__).parent
repo_root = current_data_path.parents[4]

class GraspPoseConverter:
    """
    Class for handling grasp pose coordinate transformations.
    Converts AnyGrasp output poses from camera frame to robot base frame.
    """

    def __init__(self, camera_pose: dict, robot_base_pose: Optional[dict] = None):
        """
        Initialize the converter.

        Args:
            camera_pose: Camera pose in world frame, format {'pos': [x,y,z], 'quat': [w,x,y,z]}.
            robot_base_pose: Robot base pose in world frame.
                            If None, assumes the base is at the world origin with no rotation.
        """
        self.cam_pos_world = np.array(camera_pose['pos'])
        self.cam_rot_world_matrix = self._quaternion_to_matrix_np(camera_pose['quat'])

        if robot_base_pose is None:
            robot_base_pose = {'pos': [0.0, 0.0, 0.0], 'quat': [1.0, 0.0, 0.0, 0.0]}
        
        robot_base_pos_world = np.array(robot_base_pose['pos'])
        robot_base_rot_world_matrix = self._quaternion_to_matrix_np(robot_base_pose['quat'])
        
        self.inv_robot_base_rot_matrix = robot_base_rot_world_matrix.T
        self.robot_base_pos_world = robot_base_pos_world

    def _quaternion_to_matrix_np(self, quat: list) -> np.ndarray:
        """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
        scipy_quat = [quat[1], quat[2], quat[3], quat[0]]
        return Rotation.from_quat(scipy_quat).as_matrix()

    def _matrix_to_quaternion_np(self, rot_matrix: np.ndarray) -> np.ndarray:
        """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
        rotation = Rotation.from_matrix(rot_matrix)
        scipy_quat = rotation.as_quat()  # returns [x, y, z, w]
        isaac_quat = np.array([scipy_quat[3], scipy_quat[0], scipy_quat[1], scipy_quat[2]])
        return isaac_quat

    def compute_grasp_pose(self, json_path: str, sim_device: str) -> torch.Tensor:
        """
        Read grasp pose from JSON file, perform all coordinate transforms, and return
        the final robot target pose.

        Args:
            json_path: Path to best_grasp_pose.json file.
            sim_device: Device used by the simulator (e.g., "cuda:0" or "cpu").

        Returns:
            Tensor with 7 elements [x, y, z, w, qx, qy, qz] representing the final grasp target pose.
        """
        with open(json_path, 'r') as f:
            grasp_data = json.load(f)['best_grasp']
        
        position_cam = np.array(grasp_data['translation'])
        rotation_cam = np.array(grasp_data['rotation_matrix'])
        gripper_depth = grasp_data['depth']

        # Coordinate system correction
        Rx_pi = np.diag([1.0, -1.0, -1.0]) 
        rotation_corrected = Rx_pi @ rotation_cam
        position_corrected = Rx_pi @ position_cam

        # Compute wrist target point
        forward = rotation_corrected[:, 0]
        position_moved = position_corrected + forward * (gripper_depth)

        # Camera frame -> World frame
        wrist_position_world = self.cam_rot_world_matrix @ position_moved + self.cam_pos_world
        wrist_rotation_world_matrix = self.cam_rot_world_matrix @ rotation_corrected

        # World frame -> Robot base frame
        wrist_position_robot = self.inv_robot_base_rot_matrix @ (wrist_position_world - self.robot_base_pos_world)
        wrist_rotation_robot_matrix = self.inv_robot_base_rot_matrix @ wrist_rotation_world_matrix

        Ry_90 = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0]
        ])
        wrist_rotation_robot_matrix = wrist_rotation_robot_matrix @ Ry_90

        wrist_rotation_robot_quat = self._matrix_to_quaternion_np(wrist_rotation_robot_matrix)
        grasp_pose = np.concatenate([wrist_position_robot, wrist_rotation_robot_quat])
        
        return torch.tensor(grasp_pose, dtype=torch.float32, device=sim_device)


class VisionPipeline:
    """
    Vision perception pipeline class.
    Responsible for camera data capture, BBox generation and AnyGrasp grasp detection.
    """
    
    # Default configuration
    ANYGRASP_SCRIPT_PICK = Path("/home/xhf/anygrasp_sdk/grasp_detection/demo_pick.py")
    DEFAULT_ANYGRASP_CWD = Path("/home/xhf/anygrasp_sdk/grasp_detection")
    DEFAULT_TEMP_DIR = repo_root / "anygrasp_file"
    DEFAULT_CAMERA_POSE = {'pos': [0.68, 0, 1], 'quat': [0.707, 0, 0, 0.707]}

    def __init__(
        self,
        temp_dir: Optional[Path] = None,
        anygrasp_script: Optional[str] = None,
        anygrasp_cwd: Optional[str] = None,
        camera_pose: Optional[dict] = None,
    ):
        """
        Initialize the vision perception pipeline.

        Args:
            temp_dir: Temporary file directory
            anygrasp_script: AnyGrasp script path
            anygrasp_cwd: AnyGrasp working directory
            camera_pose: Camera pose configuration
        """
        self.temp_dir = Path(temp_dir) if temp_dir else self.DEFAULT_TEMP_DIR
        self.anygrasp_script = anygrasp_script
        self.anygrasp_cwd = anygrasp_cwd or self.DEFAULT_ANYGRASP_CWD
        self.camera_pose = camera_pose or self.DEFAULT_CAMERA_POSE
        
        # Create temporary directory
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize pose converter
        self.pose_converter = GraspPoseConverter(self.camera_pose)
        
        print(f"[VisionPipeline] Initialized with temp_dir: {self.temp_dir}")
    
    def capture_camera_data(
        self, 
        scene: Any, 
        num_envs: int,
        rgb_camera_name: str = "top_rgb_camera",
        depth_camera_name: str = "top_depth_camera"
    ) -> bool:
        """
        Capture and save RGB-D data for all environments.
        """
        try:
            rgb_cam = scene[rgb_camera_name]
            depth_cam = scene[depth_camera_name]

            rgb_np = rgb_cam.data.output["rgb"].cpu().numpy()
            depth_np = depth_cam.data.output["depth"][..., 0].cpu().numpy()

            for env_idx in range(num_envs):
                rgb_path = self.temp_dir / f"rgb_{env_idx}.png"
                depth_path = self.temp_dir / f"depth_{env_idx}.npy"
                rgb_npy_path = self.temp_dir / f"rgb_{env_idx}.npy"

                Image.fromarray(rgb_np[env_idx].astype(np.uint8)).save(rgb_path)
                np.save(depth_path, (depth_np[env_idx] * 1000.0).astype(np.float32))
                np.save(rgb_npy_path, rgb_np[env_idx].astype(np.float32))

            print(f"  [VisionPipeline] Captured camera data for {num_envs} environments")
            return True
        except Exception as e:
            print(f"  [VisionPipeline] ❌ Camera capture failed: {e}")
            return False
    
    def _find_object_in_scene(self, target_label: str, scene: Any) -> Optional[str]:
        """
        Find an object in scene.rigid_objects by label.
        
        Based on YAML format where labels are simple names like "banana", "bowl", "can".
        The rigid_objects keys should match these labels directly.
        
        Args:
            target_label: Object label from YAML (e.g., "banana")
            scene: Scene object containing rigid_objects
            
        Returns:
            The key in rigid_objects if found, None otherwise
        """
        if not target_label:
            return None
            
        rigid_objects = getattr(scene, "rigid_objects", {})
        if not rigid_objects:
            return None
        
        if target_label in rigid_objects:
            return target_label
        
        return None
    
    def generate_bbox(
        self,
        target_label: str,
        scene: Any,
        num_envs: int,
        env_origins: np.ndarray,
        bbox_size: float = 0.1,
    ) -> bool:
        """
        Generate 3D BBox for the target object.

        Args:
            target_label: Target object label from YAML (e.g., "banana")
            scene: Scene object containing rigid_objects
            num_envs: Number of environments
            env_origins: Environment origin positions
            bbox_size: Size of bounding box
            
        Returns:
            True if successful, False otherwise
        """
        # Find the object in scene.rigid_objects
        matched_name = self._find_object_in_scene(target_label, scene)
        
        if not matched_name:
            rigid_keys = list(getattr(scene, "rigid_objects", {}).keys())
            print(f"  [VisionPipeline] ❌ Target '{target_label}' not found in rigid_objects.")
            print(f"  [VisionPipeline]    Available objects: {rigid_keys}")
            return False
        
        half = bbox_size / 2.0
        saved_count = 0

        for env_idx in range(num_envs):
            obj_prim = scene.rigid_objects[matched_name]
            pose = obj_prim.data.root_state_w[env_idx, :7].cpu().numpy()
            pos_w = pose[:3]
            
            pos_local = pos_w - env_origins[env_idx]
            cx, cy, cz = map(float, pos_local)

            # Transform coordinates for AnyGrasp (camera coordinate system)
            bbox = {
                "world_xmin": cy - half,
                "world_xmax": cy + half,
                "world_ymin": cx - half - 0.68,
                "world_ymax": cx + half - 0.68,
                "world_zmin": max(0.0, cz - half),
                "world_zmax": cz + half,
            }

            with open(self.temp_dir / f"bbox_{env_idx}.json", "w") as f:
                json.dump(bbox, f)
            saved_count += 1

        print(f"  [VisionPipeline] Generated BBox for {saved_count}/{num_envs} environments (target: {matched_name})")
        return saved_count > 0
    
    def run_anygrasp(self, num_envs: int) -> bool:
        """Run AnyGrasp grasp detection subprocess.

        Args:
            num_envs: Number of environments
        """
        script = self.anygrasp_script if self.anygrasp_script else self.ANYGRASP_SCRIPT_PICK
        cmd = [
            "conda", "run", "-n", "sdk", "python", str(script),
            "--top_down_grasp",
            "--num_envs", str(num_envs),
        ]
        
        try:
            print(f"  [VisionPipeline] Running AnyGrasp...")
            result = subprocess.run(cmd, cwd=self.anygrasp_cwd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  [VisionPipeline] ✓ AnyGrasp completed successfully")
                return True
            print(f"  [VisionPipeline] ❌ AnyGrasp Error:\n{result.stderr}")
            return False
        except Exception as e:
            print(f"  [VisionPipeline] ❌ Failed to launch AnyGrasp: {e}")
            return False
    
    def parse_grasp_results(
        self,
        num_envs: int,
        device: str,
        ee_quat_w: np.ndarray = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
        """Parse AnyGrasp grasp results and perform coordinate transformation."""
        all_pick_pos, all_pick_quat, valid_env_indices = [], [], []

        for i in range(num_envs):
            grasp_file = self.temp_dir / f"best_grasp_pose_{i}.json"

            p_pos = np.array([0.5, 0, 0.5])
            p_quat = np.array([0, 1, 0, 0])

            if grasp_file.exists():
                try:
                    target_pose = self.pose_converter.compute_grasp_pose(str(grasp_file), device)
                    p_pos = target_pose[:3].cpu().numpy()
                    p_quat = target_pose[3:].cpu().numpy()
                    from .pose_utils import normalize_ee_orientation_wxyz
                    env_ee_quat = ee_quat_w[i] if ee_quat_w is not None and ee_quat_w.ndim > 1 else ee_quat_w
                    p_quat = normalize_ee_orientation_wxyz(p_quat, env_ee_quat)
                    print(f"    [VisionPipeline] [Env {i}] Grasp Pose: Pos {p_pos}, Quat {p_quat}")
                    valid_env_indices.append(i)
                except Exception as e:
                    print(f"    [VisionPipeline] [Env {i}] Pose conversion error: {e}")
            
            all_pick_pos.append(p_pos)
            all_pick_quat.append(p_quat)

        print(f"  [VisionPipeline] Parsed {len(valid_env_indices)}/{num_envs} valid grasp poses")
        return all_pick_pos, all_pick_quat, valid_env_indices
    
    def cleanup_temp_files(self):
        """Clean up temporary files."""
        patterns = ["rgb_*.png", "rgb_*.npy", "depth_*.npy", "bbox_*.json", "best_grasp_pose*.json"]
        for pattern in patterns:
            for f in self.temp_dir.glob(pattern):
                try:
                    f.unlink()
                except Exception:
                    pass
