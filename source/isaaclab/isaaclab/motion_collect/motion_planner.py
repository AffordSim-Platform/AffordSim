"""Motion planning & execution module using CuRobo

Responsibilities:
- CuRobo motion planning (plan_to_pose)
- IK solver (solve_ik_batch)
- Trajectory execution (execute_trajectory_batch, execute_ik_motion)
- Gripper control (control_gripper_batch)
- EE pose query (get_current_ee_pose)
"""
import sys
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple, List, Dict, Any
from scipy.spatial.transform import Rotation as R, Slerp

from isaaclab.motion_collect.success.constants import IK_MAX_JOINT_JUMP


class MotionPlanner:
    """CuRobo motion planner wrapper with multi-environment support"""
    
    def __init__(
        self,
        robot_cfg_name: str,
        device: str = "cuda:0",
        curobo_path: Optional[Path] = None,
        num_envs: int = 1,
        use_table_obstacle: bool = True,
    ):
        """
        Initialize motion planner

        Args:
            robot_cfg_name: Robot configuration file name (e.g., "franka.yml")
            device: Device to run planning on
            curobo_path: Path to CuRobo installation
            num_envs: Number of parallel environments (for obstacle filtering)
            use_table_obstacle: Whether to add a table cuboid obstacle.
        """
        self.robot_cfg_name = robot_cfg_name
        self.device = device
        self.num_envs = num_envs
        self.use_table_obstacle = use_table_obstacle
        self._last_target_prim_path: Optional[str] = None
        self._env_context_configured = False
        
        if curobo_path is None:
            curobo_path = Path(__file__).parent.parent / "curobo-0.7.7" / "src"
        sys.path.insert(0, str(curobo_path))
        self._import_curobo()
        self._init_motion_gen()
        print(" MotionPlanner initialized")
    
    def _import_curobo(self):
        """Import CuRobo modules"""
        from curobo.geom.sdf.world import CollisionCheckerType
        from curobo.geom.types import WorldConfig
        from curobo.types.base import TensorDeviceType
        from curobo.types.math import Pose
        from curobo.types.state import JointState
        from curobo.util.logger import setup_curobo_logger
        from curobo.util.usd_helper import UsdHelper
        from curobo.util_file import get_robot_configs_path, join_path, load_yaml
        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
        
        # Store modules
        self.CollisionCheckerType = CollisionCheckerType
        self.WorldConfig = WorldConfig
        self.TensorDeviceType = TensorDeviceType
        self.Pose = Pose
        self.JointState = JointState
        self.setup_curobo_logger = setup_curobo_logger
        self.UsdHelper = UsdHelper
        self.get_robot_configs_path = get_robot_configs_path
        self.join_path = join_path
        self.load_yaml = load_yaml
        self.MotionGen = MotionGen
        self.MotionGenConfig = MotionGenConfig
        self.MotionGenPlanConfig = MotionGenPlanConfig
    
    def _init_motion_gen(self):
        """Initialize CuRobo motion generator"""
        self.setup_curobo_logger("error")
        
        self.tensor_args = self.TensorDeviceType()
        robot_cfg_path = self.get_robot_configs_path()
        self.robot_cfg = self.load_yaml(
            self.join_path(robot_cfg_path, self.robot_cfg_name)
        )["robot_cfg"]
        
        # Table cuboid obstacle (table-mounted robot base would be blocked by it; can be disabled per robot type)
        from curobo.geom.types import Cuboid as CuroboCuboid
        if self.use_table_obstacle:
            self._table_cuboid = CuroboCuboid(
                name="table",
                dims=[2.0, 2.0, 0.04],
                pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            )
            world_cfg = self.WorldConfig.from_dict({
                "cuboid": {
                    "table": {
                        "dims": [2.0, 2.0, 0.04],
                        "pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    }
                }
            })
            print("  [INFO] Table cuboid obstacle: ENABLED")
        else:
            world_cfg = self.WorldConfig.from_dict({})
            print("  [INFO] Table cuboid obstacle: DISABLED (robot-mounted-table mode)")

        motion_gen_config = self.MotionGenConfig.load_from_robot_config(
            self.robot_cfg,
            world_cfg,
            self.tensor_args,
            collision_checker_type=self.CollisionCheckerType.MESH,
            num_trajopt_seeds=12,
            num_graph_seeds=12,
            interpolation_dt=0.02,
            collision_cache={"obb": 300, "mesh": 100},
            optimize_dt=True,
            trajopt_tsteps=16,
            collision_activation_distance=0.15,  # red cup
            # collision_activation_distance=0.03, 
            # collision_activation_distance=0.23,
        )
        
        self.motion_gen = self.MotionGen(motion_gen_config)
        self.motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)
        
        print("   Motion generator initialized")
    
    # =========================================================================
    # Environment context (injected by datacollector after env init)
    # =========================================================================

    def configure_env_context(
        self,
        world_stage: Any,
        robot_prim_path: str,
        scene: Any,
    ):
        """Inject runtime environment references needed by obstacle management.

        NOTE: ``target_prim_path`` is NOT stored here because primitives update
        it dynamically on the DataCollector during execution.  It is passed
        fresh to every call that needs it.
        """
        self._world_stage = world_stage
        self._robot_prim_path = robot_prim_path
        self._scene = scene
        self._env_context_configured = True

    # =========================================================================
    # Obstacle Management
    # =========================================================================

    def get_ignored_objects_for_env(
        self, env_idx: int, target_prim_path: Optional[str] = None,
    ) -> List[str]:
        """Generate list of ignored objects for a specific environment."""
        base_prefix = f"/World/envs/env_{env_idx}"
        robot_prim_path_env = self._robot_prim_path.replace("env_0", f"env_{env_idx}")

        ignored = [
            robot_prim_path_env,
            f"{base_prefix}/scene/GroundPlane",
            f"{base_prefix}/table",
        ]

        if target_prim_path:
            target_path_env = target_prim_path.replace("env_0", f"env_{env_idx}")
            ignored.append(target_path_env)

        return ignored

    def update_obstacles_for_env(
        self, env_idx: int, target_prim_path: Optional[str] = None,
    ):
        """Update motion planner obstacles for a specific environment."""
        from . import scene_query as sq

        self._last_target_prim_path = target_prim_path
        ignored_objects = self.get_ignored_objects_for_env(env_idx, target_prim_path)
        robot_prim_path_env = self._robot_prim_path.replace("env_0", f"env_{env_idx}")
        object_poses = sq.get_object_poses_for_env(env_idx, self._scene)

        self.my_update_obstacles(
            world_stage=self._world_stage,
            robot_prim_path=robot_prim_path_env,
            ignored_objects=ignored_objects,
            object_poses=object_poses,
            env_idx=env_idx,
            total_num_envs=self.num_envs,
            debug=False,
        )

    def update_all_env_obstacles(self, target_prim_path: Optional[str] = None):
        """Compatibility shim: remember target object for legacy callers.

        This method intentionally does not update all env obstacle worlds
        anymore. CuRobo MotionGen stores only one active world, and
        plan_to_pose_batch refreshes the correct per-env world immediately
        before each plan.

        New batch-planning code should pass target_prim_path directly to
        plan_to_pose_batch(). New single-env planning code should call
        update_obstacles_for_env(env_idx, target_prim_path) before plan_to_pose().
        """
        self._last_target_prim_path = target_prim_path

    def init_motion_planning(self, target_prim_path: Optional[str] = None):
        """Compatibility hook for motion planning initialization.

        Obstacle worlds are no longer initialized here. Batch planning refreshes
        the correct per-env world immediately before each plan, and single-env
        planning should call update_obstacles_for_env() before plan_to_pose().
        """
        self._last_target_prim_path = target_prim_path
        print(" Motion planning initialized")

    def update_obstacles(self, world_stage, robot_prim_path: str, ignored_objects: List[str]):
        """
        Update obstacle world from USD stage (legacy single-environment method)
        
        Args:
            world_stage: USD stage containing the world
            robot_prim_path: Path to robot prim in USD
            ignored_objects: List of object paths to ignore
        """
        usd_help = self.UsdHelper()
        usd_help.load_stage(world_stage)
        
        obstacles = usd_help.get_obstacles_from_stage(
            only_paths=["/World"],
            reference_prim_path=robot_prim_path,
            ignore_substring=ignored_objects,
        ).get_collision_check_world()
        
        if obstacles is not None:
            if self.use_table_obstacle and hasattr(self, '_table_cuboid'):
                if obstacles.cuboid is None:
                    obstacles.cuboid = [self._table_cuboid]
                else:
                    obstacles.cuboid.append(self._table_cuboid)
            self.motion_gen.update_world(obstacles)
            print(f"   Obstacles updated: {len(obstacles.objects)} objects")

    @staticmethod
    def _obstacle_name_matches_label(obstacle_name: str, label: str) -> bool:
        """Return True only when the obstacle name contains label as a full path segment."""
        if not obstacle_name or not label:
            return False
        if obstacle_name == label:
            return True

        normalized_name = str(obstacle_name).replace("\\", "/")
        return label in [part for part in normalized_name.split("/") if part]
    
    def my_update_obstacles(
        self,
        world_stage,
        robot_prim_path: str,
        ignored_objects: List[str],
        object_poses: Optional[Dict[str, np.ndarray]] = None,
        env_idx: int = 0,
        total_num_envs: int = 1,
        debug: bool = False,
    ):
        """
        Update obstacle world from USD stage (improved multi-environment version).
        
        This method only scans the specific environment path to avoid cross-environment
        contamination, then overrides obstacle positions with actual simulation poses.
        
        Args:
            world_stage: USD stage containing the world
            robot_prim_path: Path to robot prim in USD
            ignored_objects: List of object paths to ignore
            object_poses: Dict mapping object names to positions relative to env origin
            env_idx: Environment index to update for
            total_num_envs: Total number of environments
            debug: Print debug information
        """
        usd_help = self.UsdHelper()
        usd_help.load_stage(world_stage)
        
        # Only scan the specific environment path
        env_path = f"/World/envs/env_{env_idx}"
        
        # Build ignore list: include all OTHER environments
        full_ignore_list = list(ignored_objects)
        for other_idx in range(total_num_envs):
            if other_idx != env_idx:
                full_ignore_list.append(f"/World/envs/env_{other_idx}")
        
        obstacles = usd_help.get_obstacles_from_stage(
            only_paths=[env_path],  # Only scan this environment
            reference_prim_path=robot_prim_path,
            ignore_substring=full_ignore_list,
        ).get_collision_check_world()
        
        if obstacles is None:
            print(f"    No obstacles found for env_{env_idx}")
            return
        
        # Override obstacle positions with actual simulation poses
        if object_poses:
            if debug:
                # print(f"  [DEBUG] Object poses to apply: {list(object_poses.keys())}")
                # for k, v in object_poses.items():
                #     print(f"    {k}: {np.round(v, 4).tolist()}")
                pass
            
            for obj in obstacles.objects:
                obj_name = getattr(obj, "name", "")
                
                # Find matching object pose
                matched_pos = None
                matched_key = None
                
                for key, pos in object_poses.items():
                    if self._obstacle_name_matches_label(obj_name, key):
                        matched_pos = pos
                        matched_key = key
                        break
                
                if matched_pos is None:
                    continue
                
                try:
                    matched_pos = np.asarray(matched_pos, dtype=float).reshape(-1)
                    if matched_pos.shape[0] < 3:
                        continue
                    
                    pose_attr = getattr(obj, "pose", None)
                    
                    # Override position with simulation position
                    if isinstance(pose_attr, list) and len(pose_attr) >= 3 and isinstance(
                        pose_attr[0], (int, float, np.floating)
                    ):
                        old_pos = [pose_attr[0], pose_attr[1], pose_attr[2]]
                        pose_attr[0] = float(matched_pos[0])
                        pose_attr[1] = float(matched_pos[1])
                        pose_attr[2] = float(matched_pos[2])
                        
                        if debug:
                            # print(f"     Updated '{matched_key}'  to  '{obj_name}':")
                            # print(f"      USD: {np.round(old_pos, 4).tolist()}  to  Sim: {np.round(matched_pos[:3], 4).tolist()}")
                            pass
                    else:
                        if debug:
                            # print(f"    Cannot update pose for {obj_name}: unsupported type {type(pose_attr)}")
                            pass
                except Exception as e:
                    if debug:
                        # print(f"    Failed to update pose for {obj_name}: {e}")
                        pass
        
        # Append table cuboid to obstacle list when enabled.
        if self.use_table_obstacle and hasattr(self, '_table_cuboid'):
            if obstacles.cuboid is None:
                obstacles.cuboid = [self._table_cuboid]
            else:
                obstacles.cuboid.append(self._table_cuboid)

        self.motion_gen.update_world(obstacles)

        if debug:
            n_objs = len(obstacles.objects) if obstacles.objects else 0
            n_cuboids = len(obstacles.cuboid) if obstacles.cuboid else 0
            # print(f"   Obstacles updated for env_{env_idx}: {n_objs} mesh + {n_cuboids} cuboid")
            for i, obj in enumerate(obstacles.objects):
                name = getattr(obj, "name", "N/A")
                # Extract position string for debugging
                pos_str = "unknown"
                try:
                    pose = getattr(obj, "pose", None)
                    if pose is not None:
                        if isinstance(pose, list) and len(pose) >= 3 and isinstance(pose[0], (int, float)):
                            pos_str = str(np.round(pose[:3], 4).tolist())
                        elif hasattr(pose, "position"):
                            pos = getattr(pose, "position")
                            if hasattr(pos, "cpu"):
                                pos = pos.cpu().numpy()
                            pos_str = str(np.round(np.array(pos).flatten()[:3], 4).tolist())
                        elif isinstance(pose, (torch.Tensor, np.ndarray)):
                            if hasattr(pose, "cpu"):
                                pose = pose.cpu().numpy()
                            pos_str = str(np.round(pose.flatten()[:3], 4).tolist())
                except Exception as e:
                    pos_str = f"error: {e}"
                # print(f"    [{i}] {name}: pos={pos_str}")
    
    def plan_to_pose(
        self,
        current_joint_pos: np.ndarray,
        current_joint_vel: np.ndarray,
        joint_names: List[str],
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        enable_graph: bool = False,
        max_attempts: int = 10,
    ) -> Tuple[Optional[object], bool]:
        """
        Plan motion to target end-effector pose
        
        Args:
            current_joint_pos: Current joint positions
            current_joint_vel: Current joint velocities
            joint_names: Joint names
            target_pos: Target position [x, y, z]
            target_quat: Target quaternion [w, x, y, z]
            enable_graph: Enable graph-based planning
            max_attempts: Maximum planning attempts
        
        Returns:
            Tuple of (planned trajectory, success flag)
        """
        if np.any(np.isnan(current_joint_pos)):
            print("    Invalid joint positions (NaN)")
            return None, False
        
        cu_js = self.JointState(
            position=self.tensor_args.to_device(current_joint_pos),
            velocity=self.tensor_args.to_device(current_joint_vel) * 0.0,
            acceleration=self.tensor_args.to_device(current_joint_vel) * 0.0,
            jerk=self.tensor_args.to_device(current_joint_vel) * 0.0,
            joint_names=joint_names,
        )
        
        cu_js = cu_js.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)

        # UR5e / Kinova ee_link frame differs from the upper-level grasp
        # frame by 90 degrees around Z axis.
        if (
            "ur5e" in self.robot_cfg_name
            or "kinova" in self.robot_cfg_name
        ):
            r_target = R.from_quat([target_quat[1], target_quat[2], target_quat[3], target_quat[0]])
            r_z90 = R.from_euler('z', 90, degrees=True)
            q_corrected = (r_target * r_z90).as_quat()  # scipy xyzw
            target_quat = np.array([q_corrected[3], q_corrected[0], q_corrected[1], q_corrected[2]])

        ik_goal = self.Pose(
            position=self.tensor_args.to_device(target_pos),
            quaternion=self.tensor_args.to_device(target_quat),
        )

        # print(f"  Planning to pose: pos={target_pos}, quat={target_quat}")

        graph_attempt = 3 if enable_graph else 0

        plan_config = self.MotionGenPlanConfig(
            enable_graph=enable_graph,
            enable_graph_attempt=graph_attempt,
            max_attempts=max_attempts,
            enable_finetune_trajopt=True,
            time_dilation_factor=1.0,
            finetune_attempts=5,
        )

        result = self.motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, plan_config)

        # print(f"   Planning result details:")
        # print(f"      - success: {result.success.item()}")
        # print(f"      - status: {result.status}")
        # print(f"      - valid_query: {result.valid_query}")
        # print(f"      - ik_success: {result.ik_success if hasattr(result, 'ik_success') else 'N/A'}")
        # print(f"      - graph_success: {result.graph_success if hasattr(result, 'graph_success') else 'N/A'}")
        # print(f"      - trajopt_success: {result.trajopt_success if hasattr(result, 'trajopt_success') else 'N/A'}")
        #
        # if hasattr(result, 'interpolated_plan') and result.interpolated_plan is not None:
        #     print(f"      - interpolated_plan exists")
        # else:
        #     print(f"      - interpolated_plan is None")

        if result.success.item():
            cmd_plan = result.get_interpolated_plan()
            cmd_plan = self.motion_gen.get_full_js(cmd_plan)
            return cmd_plan, True
        else:
            print(f"    Planning failed: {result.status}")
            # print(f"    Debug info:")
            # print(f"      - valid_query: {result.valid_query}")
            # print(f"      - optimized_dt: {result.optimized_dt if hasattr(result, 'optimized_dt') else 'N/A'}")
            # if hasattr(result, 'debug_info'):
            #     print(f"      - debug_info: {result.debug_info}")
            return None, False
    
    def plan_to_pose_batch(
        self,
        current_joint_pos_batch: np.ndarray,
        current_joint_vel_batch: np.ndarray,
        joint_names: List[str],
        target_pos_batch: np.ndarray,
        target_quat_batch: np.ndarray,
        enable_graph: bool = False,
        max_attempts: int = 10,
        env_indices: Optional[List[int]] = None,
        target_prim_path: Optional[str] = None,
    ) -> Tuple[List[Optional[object]], List[bool]]:
        """
        Plan motion to target end-effector poses for multiple environments (batch)
        
        Args:
            current_joint_pos_batch: Current joint positions for all envs [batch_size, num_joints]
            current_joint_vel_batch: Current joint velocities for all envs [batch_size, num_joints]
            joint_names: Joint names
            target_pos_batch: Target positions for all envs [batch_size, 3]
            target_quat_batch: Target quaternions for all envs [batch_size, 4] (w,x,y,z)
            enable_graph: Enable graph-based planning
            max_attempts: Maximum planning attempts
            env_indices: Original environment index for each row. Defaults to row index
                only when the batch covers all environments.
            target_prim_path: Target object prim path to ignore while planning.
        
        Returns:
            Tuple of (list of planned trajectories, list of success flags)
        """
        batch_size = current_joint_pos_batch.shape[0]
        if env_indices is None:
            if self._env_context_configured and batch_size != self.num_envs:
                raise ValueError(
                    "plan_to_pose_batch() received a subset batch without env_indices. "
                    f"batch_size={batch_size}, num_envs={self.num_envs}. "
                    "Pass env_indices so each row updates the correct obstacle world."
                )
            env_indices = list(range(batch_size))
        else:
            env_indices = [int(env_idx) for env_idx in env_indices]
            if len(env_indices) != batch_size:
                raise ValueError(
                    f"env_indices length {len(env_indices)} does not match batch_size {batch_size}"
                )
        
        if np.any(np.isnan(current_joint_pos_batch)):
            print("    Invalid joint positions (NaN) in batch")
            return [None] * batch_size, [False] * batch_size
        
        print(f"  Planning sequentially for {batch_size} environments...")
        
        cmd_plans = []
        successes = []
        success_count = 0
        if target_prim_path is None:
            target_prim_path = self._last_target_prim_path
        else:
            self._last_target_prim_path = target_prim_path
        
        for i in range(batch_size):
            env_idx = env_indices[i]
            if self._env_context_configured and 0 <= env_idx < self.num_envs:
                self.update_obstacles_for_env(env_idx, target_prim_path)

            plan, success = self.plan_to_pose(
                current_joint_pos=current_joint_pos_batch[i],
                current_joint_vel=current_joint_vel_batch[i],
                joint_names=joint_names,
                target_pos=target_pos_batch[i],
                target_quat=target_quat_batch[i],
                enable_graph=enable_graph,
                max_attempts=max_attempts,
            )
            cmd_plans.append(plan)
            successes.append(success)
            if success:
                success_count += 1
        
        print(f"   Planning complete: {success_count}/{batch_size} succeeded")
        return cmd_plans, successes

    # =========================================================================
    # IK Solver
    # =========================================================================

    def init_ik_solver(self):
        """Initialize CuRobo IK solver - collision-check-disabled version, used for execute_ik_motion interpolation steps"""
        print("  [INFO] Creating IK solver (no collision check)...")

        from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
        from curobo.types.math import Pose as CuroboPose

        robot_cfg = self.robot_cfg
        tensor_args = self.tensor_args

        ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            None,
            rotation_threshold=0.05,
            position_threshold=0.005,
            num_seeds=20,
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=tensor_args,
            use_cuda_graph=False,
        )
        self.ik_solver = IKSolver(ik_config)
        self._ik_pose_cls = CuroboPose
        self._ik_tensor_args = tensor_args
        print(" CuRobo IK solver initialized (no world collision, self-collision enabled)")

    @staticmethod
    def _apply_robot_quat_correction(
        quaternions: np.ndarray,
        robot_cfg_name: str,
    ) -> np.ndarray:
        """Apply robot-specific quaternion correction for batch IK.

        Args:
            quaternions: (batch, 4) wxyz
            robot_cfg_name: Robot config name string

        Returns:
            Corrected quaternions (batch, 4) wxyz
        """
        if (
            "ur5e" not in robot_cfg_name
            and "kinova" not in robot_cfg_name
        ):
            return quaternions.copy()
        # wxyz  to  xyzw for scipy
        q_scipy = quaternions[:, [1, 2, 3, 0]].astype(np.float64)
        r_target = R.from_quat(q_scipy)
        r_corr = R.from_euler('z', 90, degrees=True)
        q_corrected = (r_target * r_corr).as_quat()  # (batch, 4) xyzw
        # xyzw  to  wxyz
        return q_corrected[:, [3, 0, 1, 2]]

    @staticmethod
    def _as_batched_ik_candidates(
        values: torch.Tensor,
        batch_size: int,
        name: str,
    ) -> torch.Tensor:
        """Normalize IK result tensors to ``[batch, candidates, dof]``.

        CuRobo result tensors can be returned either as one solution per env or
        as multiple seed/candidate solutions.  Keeping the candidate dimension
        lets us choose the joint-space-continuous branch explicitly.
        """
        tensor = values.detach()
        while tensor.dim() > 3 and tensor.shape[-2] == 1:
            tensor = tensor.squeeze(-2)

        if tensor.dim() == 1:
            return tensor.reshape(1, 1, -1)

        if tensor.dim() == 2:
            if tensor.shape[0] == batch_size:
                return tensor.unsqueeze(1)
            if batch_size == 1:
                return tensor.unsqueeze(0)
            raise ValueError(
                f"Unexpected {name} IK result shape {tuple(tensor.shape)} "
                f"for batch_size={batch_size}"
            )

        if tensor.dim() == 3:
            if tensor.shape[0] == batch_size:
                return tensor
            if tensor.shape[1] == batch_size:
                return tensor.transpose(0, 1)
            raise ValueError(
                f"Unexpected {name} IK result shape {tuple(tensor.shape)} "
                f"for batch_size={batch_size}"
            )

        raise ValueError(f"Unexpected {name} IK result rank {tensor.dim()}")

    @staticmethod
    def _success_to_candidate_mask(
        success_tensor: torch.Tensor,
        batch_size: int,
        num_candidates: int,
    ) -> np.ndarray:
        """Normalize CuRobo success flags to ``[batch, candidates]``."""
        success = success_tensor.detach().cpu().numpy().astype(bool)
        success = np.squeeze(success)

        if success.ndim == 0:
            return np.full((batch_size, num_candidates), bool(success))

        if success.ndim == 1:
            if success.shape[0] == batch_size:
                return np.repeat(success[:, None], num_candidates, axis=1)
            if batch_size == 1 and success.shape[0] == num_candidates:
                return success[None, :]
            if success.shape[0] == batch_size * num_candidates:
                return success.reshape(batch_size, num_candidates)
            raise ValueError(
                f"Unexpected IK success shape {tuple(success.shape)} "
                f"for batch_size={batch_size}, num_candidates={num_candidates}"
            )

        if success.ndim == 2:
            if success.shape == (batch_size, num_candidates):
                return success
            if success.shape == (batch_size, 1):
                return np.repeat(success, num_candidates, axis=1)
            if success.shape == (1, num_candidates) and batch_size == 1:
                return success
            if success.shape == (num_candidates, batch_size):
                return success.T

        raise ValueError(
            f"Unexpected IK success shape {tuple(success.shape)} "
            f"for batch_size={batch_size}, num_candidates={num_candidates}"
        )

    def _get_ik_joint_bounds(self, n_ik: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return IK joint position bounds in IK joint order when available."""
        try:
            lows = self.ik_solver.solver.safety_rollout.action_bound_lows
            highs = self.ik_solver.solver.safety_rollout.action_bound_highs
            if hasattr(lows, "detach"):
                lows = lows.detach().cpu().numpy()
            if hasattr(highs, "detach"):
                highs = highs.detach().cpu().numpy()
            lows = np.asarray(lows, dtype=np.float64).reshape(-1)[:n_ik]
            highs = np.asarray(highs, dtype=np.float64).reshape(-1)[:n_ik]
            if lows.shape[0] != n_ik or highs.shape[0] != n_ik:
                return None, None
            return lows, highs
        except Exception:
            return None, None

    def _build_local_ik_seed_config(
        self,
        seed_joint_pos: np.ndarray,
        num_seeds: int,
    ) -> np.ndarray:
        """Build a local seed cloud around the previous IK solution.

        CuRobo fills missing seeds with broad samples.  For servo-style IK this
        can pull the returned solution to another branch, so provide all seeds
        explicitly near the previous joint configuration.
        """
        seed = np.asarray(seed_joint_pos, dtype=np.float64)
        if seed.ndim != 2:
            raise ValueError(f"seed_joint_pos must be [batch, dof], got {seed.shape}")

        batch_size, n_ik = seed.shape
        num_seeds = max(1, int(num_seeds))
        seeds = np.repeat(seed[:, None, :], num_seeds, axis=1)

        if num_seeds == 1:
            return seeds

        offsets: List[np.ndarray] = []

        # Single-joint local probes preserve the current nullspace branch when
        # a nearby solution exists.
        single_joint_radius = 0.04
        for joint_idx in range(n_ik):
            plus = np.zeros(n_ik, dtype=np.float64)
            minus = np.zeros(n_ik, dtype=np.float64)
            plus[joint_idx] = single_joint_radius
            minus[joint_idx] = -single_joint_radius
            offsets.extend([plus, minus])

        # A few coupled probes help when the continuous branch needs coordinated
        # joint motion rather than a single-axis perturbation.
        combo_radius = 0.07
        k = 0
        while len(offsets) < num_seeds - 1:
            joint_ids = np.arange(1, n_ik + 1, dtype=np.float64)
            direction = np.sin((k + 1) * joint_ids * 1.61803398875)
            max_abs = np.max(np.abs(direction))
            if max_abs > 1e-8:
                direction = direction / max_abs
            offsets.append(combo_radius * direction)
            k += 1

        for seed_idx, offset in enumerate(offsets[: num_seeds - 1], start=1):
            seeds[:, seed_idx, :] = seed + offset[None, :]

        lows, highs = self._get_ik_joint_bounds(n_ik)
        if lows is not None and highs is not None:
            seeds = np.clip(seeds, lows[None, None, :], highs[None, None, :])

        return seeds

    def _solve_ik_batch_parallel(
        self,
        target_positions: np.ndarray,
        target_quaternions: np.ndarray,
        seed_joint_pos: np.ndarray,
        arm_joint_names: List[str],
        n_arm: int,
    ) -> Tuple[np.ndarray, np.ndarray, List[bool]]:
        """Solve IK for all envs in one GPU batch call.

        Args:
            target_positions: (batch, 3)
            target_quaternions: (batch, 4) wxyz - before robot-specific correction
            seed_joint_pos: (batch, n_ik_joints) in IK joint order
            arm_joint_names: Arm joint names in robot order
            n_arm: Number of arm joints

        Returns:
            joint_positions_arm: (batch, n_arm) in arm_joint_names order
            joint_positions_ik: (batch, n_ik) in IK joint order (for seed chaining)
            successes: List[bool]
        """
        batch_size = target_positions.shape[0]
        n_ik = seed_joint_pos.shape[1]

        corrected_quats = self._apply_robot_quat_correction(
            target_quaternions, self.robot_cfg_name
        )

        with torch.enable_grad():
            goal_pose = self._ik_pose_cls(
                position=self._ik_tensor_args.to_device(target_positions),
                quaternion=self._ik_tensor_args.to_device(corrected_quats),
            )

            num_ik_seeds = int(getattr(self.ik_solver, "num_seeds", 1))
            seed_tensor = self._ik_tensor_args.to_device(seed_joint_pos)  # (batch, dof)
            seed_config = self._ik_tensor_args.to_device(
                self._build_local_ik_seed_config(seed_joint_pos, num_ik_seeds)
            )  # (batch, num_seeds, dof)
            retract_config = seed_tensor              # (batch, dof)

            result = self.ik_solver.solve_batch(
                goal_pose,
                retract_config=retract_config,
                seed_config=seed_config,
                return_seeds=num_ik_seeds,
            )

        # IK-ordered candidate positions for seed chaining.
        ik_candidates_t = self._as_batched_ik_candidates(
            result.js_solution.position, batch_size, "IK-ordered"
        )
        ik_candidates = ik_candidates_t.detach().cpu().numpy()[:, :, :n_ik]

        # Arm-ordered candidate positions for trajectory output.
        ordered_js = result.js_solution.get_ordered_joint_state(arm_joint_names)
        arm_candidates_t = self._as_batched_ik_candidates(
            ordered_js.position, batch_size, "arm-ordered"
        )
        arm_candidates = arm_candidates_t.detach().cpu().numpy()[:, :, :n_arm]

        num_candidates = min(ik_candidates.shape[1], arm_candidates.shape[1])
        ik_candidates = ik_candidates[:, :num_candidates, :]
        arm_candidates = arm_candidates[:, :num_candidates, :]

        success_mask = self._success_to_candidate_mask(
            result.success, batch_size, num_candidates
        )
        success_mask = success_mask[:, :num_candidates]
        successes = [bool(v) for v in np.any(success_mask, axis=1).tolist()]

        # Explicitly choose the IK branch closest to the previous seed in joint space.
        seed_np = np.asarray(seed_joint_pos, dtype=np.float64)[:, None, :n_ik]
        candidate_dist = np.linalg.norm(ik_candidates.astype(np.float64) - seed_np, axis=-1)
        masked_dist = np.where(success_mask, candidate_dist, np.inf)
        chosen_candidate = np.argmin(masked_dist, axis=1)

        # If no candidate succeeded for an env, keep the nearest candidate only
        # as a fallback output; the caller will mark that env as failed.
        failed_envs = ~np.any(success_mask, axis=1)
        if np.any(failed_envs):
            chosen_candidate[failed_envs] = np.argmin(candidate_dist[failed_envs], axis=1)

        env_indices = np.arange(batch_size)
        joint_positions_ik = ik_candidates[env_indices, chosen_candidate, :]
        joint_positions_arm = arm_candidates[env_indices, chosen_candidate, :]

        return joint_positions_arm, joint_positions_ik, successes

    def solve_ik_trajectory(
        self,
        waypoint_positions: np.ndarray,
        waypoint_quaternions: np.ndarray,
        initial_joint_pos: np.ndarray,
        robot: Any,
    ) -> Tuple[np.ndarray, np.ndarray, List[bool]]:
        """Pre-solve IK for entire trajectory using chain-seeded GPU batch solving.

        Args:
            waypoint_positions: (num_envs, num_steps, 3)
            waypoint_quaternions: (num_envs, num_steps, 4) wxyz
            initial_joint_pos: (num_envs, num_joints) full robot joint positions
            robot: Isaac Lab robot articulation

        Returns:
            joint_trajectory: (num_envs, num_steps, n_arm) in arm_joint_names order
            step_successes: (num_envs, num_steps) bool
            env_successes: List[bool] - True if all steps succeeded for that env
        """
        if not hasattr(self, 'ik_solver') or self.ik_solver is None:
            raise RuntimeError("IK solver must be initialized before solve_ik_trajectory")

        num_envs, num_steps = waypoint_positions.shape[:2]

        # Joint name mappings
        curobo_joints = set(
            self.robot_cfg['kinematics']['cspace']['joint_names']
        )
        arm_joint_names = [j for j in robot.joint_names if j in curobo_joints]
        n_arm = len(arm_joint_names)
        ik_joint_names = list(self.ik_solver.joint_names)

        # Index mappings (precomputed once)
        ik_ordered_idx = [robot.joint_names.index(j) for j in ik_joint_names]
        arm_idx_in_robot = [robot.joint_names.index(j) for j in arm_joint_names]

        # Initial seeds and arm positions
        seed_ik = initial_joint_pos[:, ik_ordered_idx].copy()  # (num_envs, n_ik)
        initial_arm = initial_joint_pos[:, arm_idx_in_robot]   # (num_envs, n_arm)

        joint_trajectory = np.zeros((num_envs, num_steps, n_arm))
        step_successes = np.zeros((num_envs, num_steps), dtype=bool)
        env_successes = [True] * num_envs

        for step in range(num_steps):
            pos_step = waypoint_positions[:, step, :]    # (num_envs, 3)
            quat_step = waypoint_quaternions[:, step, :] # (num_envs, 4)

            ik_arm, ik_ik, successes = self._solve_ik_batch_parallel(
                pos_step, quat_step, seed_ik, arm_joint_names, n_arm,
            )

            failed_envs = []
            for env_idx in range(num_envs):
                if not env_successes[env_idx]:
                    # Already frozen - repeat last good position
                    prev = joint_trajectory[env_idx, step - 1, :] if step > 0 else initial_arm[env_idx]
                    joint_trajectory[env_idx, step, :] = prev
                    continue

                if not successes[env_idx]:
                    env_successes[env_idx] = False
                    failed_envs.append(env_idx)
                    prev = joint_trajectory[env_idx, step - 1, :] if step > 0 else initial_arm[env_idx]
                    joint_trajectory[env_idx, step, :] = prev
                    continue

                # Joint jump check
                prev_arm = joint_trajectory[env_idx, step - 1, :] if step > 0 else initial_arm[env_idx]
                ik_delta = np.abs(ik_arm[env_idx] - prev_arm).max()
                if ik_delta > IK_MAX_JOINT_JUMP:
                    env_successes[env_idx] = False
                    failed_envs.append(env_idx)
                    joint_trajectory[env_idx, step, :] = prev_arm
                    continue

                # Success - record and update seed
                step_successes[env_idx, step] = True
                joint_trajectory[env_idx, step, :] = ik_arm[env_idx]
                seed_ik[env_idx] = ik_ik[env_idx]

            if failed_envs:
                print(
                    f"   IK trajectory pre-solve failed at step {step}/{num_steps} "
                    f"for envs={failed_envs}"
                )

        return joint_trajectory, step_successes, env_successes

    def solve_ik_batch(
        self,
        target_positions: np.ndarray,
        target_quaternions: np.ndarray,
        current_joint_pos: np.ndarray,
        current_joint_vel: np.ndarray,
        robot: Any,
    ) -> Tuple[np.ndarray, List[bool]]:
        """
        Solve target joint angles using pure IK solver (no collision check).
        Compatibility wrapper around _solve_ik_batch_parallel.

        Args:
            target_positions: [num_envs, 3]
            target_quaternions: [num_envs, 4] wxyz
            current_joint_pos: [num_envs, num_joints]
            current_joint_vel: [num_envs, num_joints] (unused, kept for API compat)
            robot: Isaac Lab robot articulation
        """
        if not hasattr(self, 'ik_solver') or self.ik_solver is None:
            raise RuntimeError("IK solver must be initialized before solve_ik_batch")

        num_envs = target_positions.shape[0]

        curobo_joints = set(
            self.robot_cfg['kinematics']['cspace']['joint_names']
        )
        arm_joint_names = [j for j in robot.joint_names if j in curobo_joints]
        n_arm = len(arm_joint_names)
        ik_joint_names = list(self.ik_solver.joint_names)

        # Map current_joint_pos (robot order)  to  IK order for seeding
        ik_ordered_idx = [robot.joint_names.index(j) for j in ik_joint_names]
        seed_ik = current_joint_pos[:, ik_ordered_idx]

        joint_positions_arm, _, successes = self._solve_ik_batch_parallel(
            target_positions, target_quaternions, seed_ik, arm_joint_names, n_arm,
        )

        # For failed envs, fall back to current arm joint positions
        for i in range(num_envs):
            if not successes[i]:
                joint_positions_arm[i] = current_joint_pos[i, :n_arm]

        return joint_positions_arm, successes

    # =========================================================================
    # Trajectory Execution
    # =========================================================================

    @staticmethod
    def _slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """Quaternion spherical linear interpolation, input and output in [w, x, y, z] format"""
        q1 = np.asarray(q1).flatten()
        q2 = np.asarray(q2).flatten()
        if len(q1) != 4 or len(q2) != 4:
            raise ValueError(f"Quaternion must have 4 elements, got q1: {len(q1)}, q2: {len(q2)}")

        q1_scipy = np.array([q1[1], q1[2], q1[3], q1[0]])
        q2_scipy = np.array([q2[1], q2[2], q2[3], q2[0]])

        if np.dot(q1_scipy, q2_scipy) < 0:
            q2_scipy = -q2_scipy

        dot = np.clip(np.dot(q1_scipy, q2_scipy), -1.0, 1.0)
        if dot > 0.9995:
            result = q1_scipy + t * (q2_scipy - q1_scipy)
            result = result / np.linalg.norm(result)
            return np.array([result[3], result[0], result[1], result[2]])

        rotations = R.from_quat([q1_scipy, q2_scipy])
        slerp = Slerp([0, 1], rotations)
        interp_rot = slerp(t)
        q_scipy = interp_rot.as_quat()
        return np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])

    @staticmethod
    def _build_interpolated_waypoints(
        start_pos: np.ndarray,
        start_quat: np.ndarray,
        end_pos: np.ndarray,
        end_quat: np.ndarray,
        num_steps: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build linearly interpolated position + slerp quaternion waypoints.

        Args:
            start_pos: (num_envs, 3)
            start_quat: (num_envs, 4) wxyz
            end_pos: (num_envs, 3)
            end_quat: (num_envs, 4) wxyz
            num_steps: Number of interpolation steps

        Returns:
            positions: (num_envs, num_steps, 3)
            quaternions: (num_envs, num_steps, 4) wxyz
        """
        num_envs = start_pos.shape[0]
        positions = np.zeros((num_envs, num_steps, 3))
        quaternions = np.zeros((num_envs, num_steps, 4))

        for step in range(num_steps):
            alpha = (step + 1) / num_steps
            positions[:, step, :] = (1 - alpha) * start_pos + alpha * end_pos
            for env_idx in range(num_envs):
                quaternions[env_idx, step, :] = MotionPlanner._slerp(
                    start_quat[env_idx], end_quat[env_idx], alpha
                )

        return positions, quaternions

    def get_current_ee_pose(
        self,
        env: Any,
        num_envs: int,
        robot_type: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current end-effector pose (relative to environment origin)

        Args:
            env: Gym environment (unwrapped via env.env)
            num_envs: Number of parallel environments
            robot_type: Optional robot type string for UR5e/Kinova correction

        Returns:
            positions: [num_envs, 3]
            quaternions: [num_envs, 4] wxyz
        """
        from .tools.pose_utils import _quat_multiply_wxyz

        ee_frame = env.env.scene["ee_frame"]
        ee_pos_w = ee_frame.data.target_pos_w[:num_envs].cpu().numpy()
        ee_quat_w = ee_frame.data.target_quat_w[:num_envs].cpu().numpy()

        ee_pos_w = np.squeeze(ee_pos_w)
        ee_quat_w = np.squeeze(ee_quat_w)

        if ee_pos_w.ndim == 1:
            ee_pos_w = ee_pos_w.reshape(1, -1)
        if ee_quat_w.ndim == 1:
            ee_quat_w = ee_quat_w.reshape(1, -1)

        env_origins = env.env.scene.env_origins[:num_envs].cpu().numpy()
        if env_origins.ndim == 1:
            env_origins = env_origins.reshape(1, -1)

        ee_pos_rel = ee_pos_w - env_origins
        if robot_type in {"ur5e", "kinova_robotiq85"}:
            qz_neg90 = np.array([0.7071068, 0.0, 0.0, -0.7071068])
            for i in range(ee_quat_w.shape[0]):
                ee_quat_w[i] = _quat_multiply_wxyz(ee_quat_w[i], qz_neg90)

        return ee_pos_rel, ee_quat_w

    def execute_trajectory_batch(
        self,
        cmd_plans: List[Optional[object]],
        gripper_states: List[float],
        env: Any,
        robot: Any,
        num_envs: int,
        recorder: Any = None,
        desc: str = "Executing",
        active_mask: Optional[List[bool]] = None,
    ):
        """Execute planned trajectories for all environments simultaneously."""
        # for env_idx, plan in enumerate(cmd_plans):
        #     if plan is not None:
        #         n_points = plan.position.shape[0]
        #         first_joints = plan.position[0, :4].cpu().numpy()
        #         last_joints = plan.position[-1, :4].cpu().numpy()
        #         print(f"  [DEBUG] Env {env_idx}: Trajectory has {n_points} points")
        #         print(f"  [DEBUG] Env {env_idx}: First joints: {first_joints}")
        #         print(f"  [DEBUG] Env {env_idx}: Last joints: {last_joints}")

        common_js_names = []
        idx_list = []
        robot_joint_names = robot.joint_names

        if active_mask is None:
            active_mask = [True] * num_envs
        else:
            active_mask = [bool(v) for v in list(active_mask)]
            if len(active_mask) != num_envs:
                active_mask = [True] * num_envs

        valid_plan = next((p for p in cmd_plans if p is not None), None)
        if valid_plan is None:
            return

        for x in robot_joint_names:
            if x in valid_plan.joint_names:
                idx_list.append(robot_joint_names.index(x))
                common_js_names.append(x)

        action_dim = env.env.action_manager.total_action_dim
        idx_tensor = torch.tensor(idx_list[:action_dim], dtype=torch.long)

        ordered_plans = []
        plan_lengths = []
        for plan in cmd_plans:
            if plan is not None:
                ordered_plan = plan.get_ordered_joint_state(common_js_names)
                ordered_plans.append(ordered_plan)
                plan_lengths.append(len(ordered_plan.position))
            else:
                ordered_plans.append(None)
                plan_lengths.append(0)

        max_steps = max(plan_lengths)

        for i in tqdm(range(max_steps), desc=desc, leave=False):
            action = torch.zeros(num_envs, action_dim)

            for env_idx in range(num_envs):
                if not active_mask[env_idx]:
                    action[env_idx, :] = robot.data.joint_pos[env_idx, :action_dim].cpu()
                    continue

                if ordered_plans[env_idx] is not None and i < plan_lengths[env_idx]:
                    cmd_state = ordered_plans[env_idx][i]
                    action[env_idx, idx_tensor] = cmd_state.position[:len(idx_tensor)].cpu()
                    action[env_idx, -1] = gripper_states[env_idx]
                else:
                    action[env_idx, :] = robot.data.joint_pos[env_idx, :action_dim].cpu()
                    action[env_idx, -1] = gripper_states[env_idx]

            next_obs, _, term, trunc, _ = env.step(action)

            if recorder is not None:
                recorder.add_step(next_obs, action, term, trunc, step_index=i, plan_lengths=plan_lengths)

    def control_gripper_batch(
        self,
        gripper_actions: List[float],
        env: Any,
        robot: Any,
        num_envs: int,
        recorder: Any = None,
        steps: int = 3,
        active_mask: Optional[List[bool]] = None,
    ):
        """Control grippers for all environments."""
        if active_mask is None:
            active_mask = [True] * num_envs
        else:
            active_mask = [bool(v) for v in list(active_mask)]
            if len(active_mask) != num_envs:
                active_mask = [True] * num_envs

        action_dim = env.env.action_manager.total_action_dim
        for _ in range(steps):
            action = torch.zeros(num_envs, action_dim)
            action[:, :] = robot.data.joint_pos[:num_envs, :action_dim].cpu()
            for env_idx in range(num_envs):
                if active_mask[env_idx]:
                    action[env_idx, -1] = gripper_actions[env_idx]

            obs, _, term, trunc, _ = env.step(action)

            if recorder is not None:
                recorder.add_step(obs, action, term, trunc)

    def execute_ik_motion(
        self,
        target_positions: np.ndarray,
        target_quaternions: np.ndarray,
        gripper_states: List[float],
        env: Any,
        robot: Any,
        num_envs: int,
        robot_type: Optional[str] = None,
        recorder: Any = None,
        interpolation_steps: int = 30,
        desc: str = "IK Motion",
        active_mask: Optional[List[bool]] = None,
        step_debug_fn: Any = None,
    ) -> Tuple[bool, List[bool]]:
        """Execute fine motion using pre-solved IK trajectory + joint control.

        Phase A: pre-solve the entire IK trajectory (one batch call per step).
        Phase B: execute the pre-solved joint commands step by step.
        """
        target_positions = np.atleast_2d(np.array(target_positions))
        target_quaternions = np.atleast_2d(np.array(target_quaternions))

        if target_positions.shape[0] != num_envs:
            if target_positions.shape[0] == 3 and num_envs == 1:
                target_positions = target_positions.reshape(1, 3)
        if target_quaternions.shape[0] != num_envs:
            if target_quaternions.shape[0] == 4 and num_envs == 1:
                target_quaternions = target_quaternions.reshape(1, 4)

        current_ee_pos, current_ee_quat = self.get_current_ee_pose(env, num_envs, robot_type)
        current_ee_pos = np.atleast_2d(current_ee_pos)
        current_ee_quat = np.atleast_2d(current_ee_quat)

        current_joint_pos = robot.data.joint_pos[:num_envs].cpu().numpy()
        action_dim = env.env.action_manager.total_action_dim

        if active_mask is None:
            env_successes = [True] * num_envs
        else:
            env_successes = [bool(v) for v in list(active_mask)]
            if len(env_successes) != num_envs:
                env_successes = [True] * num_envs

        # === Phase A: Pre-solve entire IK trajectory ===
        waypoints_pos, waypoints_quat = self._build_interpolated_waypoints(
            current_ee_pos, current_ee_quat,
            target_positions, target_quaternions,
            interpolation_steps,
        )

        joint_traj, _step_successes, traj_env_successes = self.solve_ik_trajectory(
            waypoints_pos, waypoints_quat, current_joint_pos, robot,
        )

        # Merge trajectory successes with active_mask
        for env_idx in range(num_envs):
            if not traj_env_successes[env_idx]:
                env_successes[env_idx] = False

        # === Phase B: Execute pre-solved trajectory ===
        n_arm = joint_traj.shape[2]
        n_arm_action = min(n_arm, action_dim - 1)

        for step in tqdm(range(interpolation_steps), desc=desc):
            if not any(env_successes):
                print(f"   No active envs remaining at IK step {step}/{interpolation_steps}, stopping early")
                break

            action = robot.data.joint_pos[:num_envs, :action_dim].cpu().clone()

            for env_idx in range(num_envs):
                if env_successes[env_idx]:
                    action[env_idx, :n_arm_action] = torch.tensor(
                        joint_traj[env_idx, step, :n_arm_action], dtype=action.dtype
                    )
                    action[env_idx, -1] = gripper_states[env_idx]

            obs, _, term, trunc, _ = env.step(action)

            if recorder is not None:
                recorder.add_step(obs, action, term, trunc)

            if step_debug_fn is not None:
                step_debug_fn(step + 1, interpolation_steps, list(env_successes))

        return any(env_successes), env_successes
