"""
DataCollector utility methods module

Contains obstacle management, trajectory execution, IK solving, pose computation and other utility methods.
Path: isaaclab/motion_collect/collector_utils.py
"""

import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
from scipy.spatial.transform import Rotation as R

from . import scene_query as sq
from .tools.pose_utils import (
    _quat_multiply_wxyz,
    compute_face_target_place_quat,
    normalize_ee_orientation_wxyz,
    parse_facing_axis_local,
)

import random
import os
import omni.kit.commands
import omni.usd
from pxr import UsdShade, Sdf, UsdGeom
import omni.kit.app


class CollectorUtilsMixin:
    """
    Utility methods Mixin class for DataCollector.
    Provides obstacle management, trajectory execution, IK solving, pose computation, etc.
    """

    # =========================================================================
    # Container Registry (semi-enclosed container registry)
    # =========================================================================

    def _load_containers_registry(self) -> Dict[str, dict]:
        """Load the containers.yaml semi-enclosed container registry"""
        task_cfg_path = Path(self.task_config_path)
        search_dirs = [task_cfg_path.parent] + list(task_cfg_path.parents)
        containers_path = None
        for directory in search_dirs:
            candidate = directory / "containers.yaml"
            if candidate.exists():
                containers_path = candidate
                break
        if containers_path is None:
            # Evaluation copies task YAMLs into per-run output directories.
            # They still use the registry bundled with this checkout unless
            # an explicit registry exists beside the task (or an ancestor).
            project_root = Path(__file__).resolve().parents[4]
            candidate = project_root / "source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/containers.yaml"
            if candidate.is_file():
                containers_path = candidate
        if containers_path is None:
            print(
                f"  [INFO] containers.yaml not found when searching from {task_cfg_path.parent} upward, "
                "no semi-enclosed containers registered."
            )
            return {}
        with open(containers_path, 'r', encoding='utf-8') as f:
            registry = yaml.safe_load(f) or {}
        print(f"  [INFO] Loaded {len(registry)} semi-enclosed container(s) from {containers_path}: {list(registry.keys())}")
        return registry

    def is_semi_enclosed_container(self, container_name: str) -> bool:
        """Check if a container is semi-enclosed (requires both dimensions and face_center)."""
        if not hasattr(self, '_containers_registry'):
            self._containers_registry = self._load_containers_registry()
        cfg = self._containers_registry.get(container_name)
        if cfg is None:
            return False
        return "dimensions" in cfg and "face_center" in cfg

    def get_container_config(self, container_name: str) -> Optional[dict]:
        """Get semi-enclosed container configuration"""
        if not hasattr(self, '_containers_registry'):
            self._containers_registry = self._load_containers_registry()
        return self._containers_registry.get(container_name)

    def compute_semi_enclosed_placement_poses(
        self,
        container_name: str,
        container_position: np.ndarray,
        container_quat_wxyz: np.ndarray,
        reference_quat_wxyz: np.ndarray = None,
    ) -> Tuple[dict, dict]:
        """
        Compute the two-step placement poses for a semi-enclosed container.

        Args:
            container_name: Container name (key in containers.yaml)
            container_position: Container position [x, y, z]. Frame consistent with caller
            container_quat_wxyz: Container orientation quaternion [w, x, y, z]
            reference_quat_wxyz: Current EE orientation quaternion [w, x, y, z], used to resolve 180-degree ambiguity

        Returns:
            pre_place_pose: {'position': [x,y,z], 'orientation': [w,x,y,z]}
            place_pose: {'position': [x,y,z], 'orientation': [w,x,y,z]}
        """
        cfg = self.get_container_config(container_name)
        if cfg is None:
            raise ValueError(f"Container '{container_name}' not found in registry")

        face_center_local = np.array(cfg['face_center'], dtype=np.float64)
        place_offset_local = np.array(cfg['place_offset'], dtype=np.float64)
        q_base = np.array(cfg['q_base'], dtype=np.float64)

        # Container rotation matrix (scipy uses xyzw)
        w, qx, qy, qz = container_quat_wxyz
        rot = R.from_quat([qx, qy, qz, w])

        # pre-place: transform face_center to world coordinates
        pre_place_world = rot.apply(face_center_local) + container_position

        # place: transform geometric center + place_offset to world coordinates
        place_world = rot.apply(place_offset_local) + container_position

        # facing_axis_local: prefer explicit field, otherwise derive by normalizing face_center
        facing_axis_local = parse_facing_axis_local(cfg.get('facing_axis_local'))
        if facing_axis_local is None:
            norm = float(np.linalg.norm(face_center_local))
            if norm < 1e-8:
                raise ValueError(
                    f"Container '{container_name}' has zero-length face_center "
                    "and no facing_axis_local; cannot derive orientation."
                )
            facing_axis_local = face_center_local / norm

        # EE orientation: computed via unified helper (semi-enclosed q_base branch)
        orientation = compute_face_target_place_quat(
            target_quat_wxyz=np.asarray(container_quat_wxyz, dtype=np.float64),
            facing_axis_local=facing_axis_local,
            q_base_wxyz=q_base,
        )
        orientation = self.normalize_ee_orientation(orientation, reference_quat_wxyz, mode="place")

        pre_place_pose = {
            'position': pre_place_world.tolist(),
            'orientation': orientation.tolist(),
        }
        place_pose = {
            'position': place_world.tolist(),
            'orientation': orientation.tolist(),
        }

        return pre_place_pose, place_pose

    # --- Container-aware pick (grasping from inside a container) ---

    def detect_object_in_container(
        self,
        target_label: str,
        env_idx: int,
    ) -> Tuple[Optional[str], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect whether the target object is inside a semi-enclosed container.

        Transforms the target object's global coordinates into each container's local frame
        and performs bounding box detection using dimensions.

        Returns:
            (container_key, container_pos_global, container_quat_wxyz)
            Returns (None, None, None) if not inside any container
        """
        if not hasattr(self, '_containers_registry'):
            self._containers_registry = self._load_containers_registry()

        if not self._containers_registry:
            return None, None, None

        target_pos, _ = self._get_object_pose_global(target_label, env_idx)
        if target_pos is None:
            return None, None, None

        for container_key, cfg in self._containers_registry.items():
            if "dimensions" not in cfg:
                # affordance-only target has no dimensions, skip bounding box detection
                continue
            cont_pos, cont_quat = self._get_object_pose_global(container_key, env_idx)
            if cont_pos is None:
                continue

            # Transform to container local frame
            w, qx, qy, qz = cont_quat
            rot = R.from_quat([qx, qy, qz, w])
            local_pos = rot.inv().apply(target_pos - cont_pos)

            # Bounding box check
            dims = np.array(cfg['dimensions'])
            half_dims = dims / 2.0
            if np.all(np.abs(local_pos) <= half_dims):
                print(f"    Env {env_idx}: target '{target_label}' detected INSIDE container '{container_key}'")
                return container_key, cont_pos, cont_quat

        return None, None, None

    def get_container_face_center_world(
        self,
        container_name: str,
        container_pos_world: np.ndarray,
        container_quat_wxyz: np.ndarray,
    ) -> np.ndarray:
        """
        Compute world coordinates of the container opening center (face_center).

        Returns:
            face_center world coordinates [x, y, z]
        """
        cfg = self.get_container_config(container_name)
        face_center_local = np.array(cfg['face_center'], dtype=np.float64)
        w, qx, qy, qz = container_quat_wxyz
        rot = R.from_quat([qx, qy, qz, w])
        return rot.apply(face_center_local) + container_pos_world

    def _detect_and_adjust_for_container_pick(
        self,
        target_label: str,
        all_pre_pick_pos: list,
    ) -> list:
        """
        Detect whether the target object is inside a container; if so, replace pre-pick
        position with face_center. Modifies all_pre_pick_pos in place.

        Args:
            target_label: Target object label
            all_pre_pick_pos: Pre-pick position list (modified in place)

        Returns:
            container_picks: list of (container_key, cont_pos_global, cont_quat_wxyz) or None per env
        """
        container_picks = [None] * self.num_envs
        env_origins = self._get_env_origins()

        for i in range(self.num_envs):
            cname, cpos, cquat = self.detect_object_in_container(target_label, i)
            if cname is not None:
                container_picks[i] = (cname, cpos, cquat)
                face_center_world = self.get_container_face_center_world(cname, cpos, cquat)
                all_pre_pick_pos[i] = face_center_world - env_origins[i]

        return container_picks


    # --- Container-aware placement (placing into container) ---


    # =========================================================================
    # Initialization & Configuration
    # =========================================================================

    def _get_ignored_objects_for_env(self, env_idx: int) -> List[str]:
        return self.motion_planner.get_ignored_objects_for_env(env_idx, self.target_prim_path)

    def _update_obstacles_for_env(self, env_idx: int):
        if self.motion_planner is None:
            return
        self.motion_planner.update_obstacles_for_env(env_idx, self.target_prim_path)

    def _update_all_env_obstacles(self):
        if self.motion_planner is None:
            return
        self.motion_planner.update_all_env_obstacles(self.target_prim_path)

    def _init_motion_planning(self):
        if self.motion_planner is None:
            return
        self.motion_planner.configure_env_context(
            world_stage=self.world.stage,
            robot_prim_path=self.robot_prim_path,
            scene=self.scene,
        )
        self.motion_planner.init_motion_planning(self.target_prim_path)


    # =========================================================================
    # Reset Logic
    # =========================================================================

    def _reset_and_stabilize(self):
        """
        Execute a robust reset:
        1. Reset Gym environment
        2. Force zero joint velocity
        3. Reset Motion Planner
        4. Physics warmup
        """
        # 1. Standard Gym Reset
        self.env.reset()
        
        # 2. Force Zero Velocity
        if self.robot:
            default_pos = self.robot.data.default_joint_pos
            
            if default_pos.ndim == 1:
                default_pos = default_pos.unsqueeze(0).repeat(self.num_envs, 1)
            
            self.robot.data.joint_pos[:] = default_pos
            self.robot.data.joint_vel[:] = 0.0
            self.robot.data.root_vel_w[:] = 0.0
            self.robot.write_data_to_sim()

        # 3. Reset Motion Planner
        if self.motion_planner and hasattr(self.motion_planner, 'reset'):
            self.motion_planner.reset()

        # 4. Physics Warmup
        action_dim = self.env.env.action_manager.total_action_dim
        zero_action = torch.zeros(self.num_envs, action_dim, device=self.device)
        
        if self.robot:
            current_pos = self.robot.data.joint_pos[:, :action_dim].clone()
            limit = min(action_dim, current_pos.shape[1])
            zero_action[:, :limit] = current_pos[:, :limit]

        for _ in range(10):
            self.env.step(zero_action)

    # =========================================================================
    # Label Resolution (Simplified based on YAML format)
    # =========================================================================

    def _find_object_in_scene(self, label: str) -> Optional[str]:
        return sq.find_object_in_scene(label, self.scene)

    def _get_object_entry_from_scene_desc(self, label: str, scene_desc: List[dict]) -> Optional[dict]:
        return sq.get_object_entry_from_scene_desc(label, scene_desc)

    # =========================================================================
    # Helper Utilities
    # =========================================================================

    def clear_texture_randomization_record(self):
        """Reset per-episode texture randomization record."""
        self._latest_texture_randomization = {
            "enabled": False,
            "assignments": [],
        }

    def _capture_light_state_snapshot(self) -> Dict[str, Any]:
        """Capture current world light states from USD stage."""
        snapshot: Dict[str, Any] = {}
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return snapshot

        candidate_paths = {
            "/World/light_sphere",
            "/World/light_disk",
            "/World/light_distant",
            "/World/dome_light",
            "/World/defaultDomeLight",
        }
        world_prim = stage.GetPrimAtPath("/World")
        if world_prim.IsValid():
            for child in world_prim.GetChildren():
                type_name = child.GetTypeName()
                if "Light" in str(type_name):
                    candidate_paths.add(str(child.GetPath()))

        def _to_jsonable(value: Any) -> Any:
            if value is None:
                return None
            if isinstance(value, (str, int, float, bool)):
                return value
            if hasattr(value, "tolist"):
                return value.tolist()
            try:
                return list(value)
            except TypeError:
                return str(value)

        for prim_path in sorted(candidate_paths):
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue

            intensity_attr = prim.GetAttribute("inputs:intensity")
            color_attr = prim.GetAttribute("inputs:color")
            translate_attr = prim.GetAttribute("xformOp:translate")
            rotate_attr = prim.GetAttribute("xformOp:rotateXYZ")

            snapshot[prim_path] = {
                "type": str(prim.GetTypeName()),
                "intensity": _to_jsonable(intensity_attr.Get() if intensity_attr.IsValid() else None),
                "color": _to_jsonable(color_attr.Get() if color_attr.IsValid() else None),
                "translate": _to_jsonable(translate_attr.Get() if translate_attr.IsValid() else None),
                "rotateXYZ": _to_jsonable(rotate_attr.Get() if rotate_attr.IsValid() else None),
            }

        return snapshot

    def _capture_episode_initial_env_states(self, task_yaml: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Capture per-env initial pose snapshot for all task objects/articulations."""
        object_names = list((task_yaml or {}).get("objects", {}).keys())
        articulation_names = list((task_yaml or {}).get("articulations", {}).keys())
        env_origins = self._get_env_origins()

        all_env_states: List[Dict[str, Any]] = []
        for env_idx in range(self.num_envs):
            env_state: Dict[str, Any] = {
                "env_idx": env_idx,
                "env_origin": env_origins[env_idx].tolist() if env_idx < len(env_origins) else None,
                "objects": {},
                "articulations": {},
            }

            for name in object_names:
                pos, quat = self._get_object_pose_global(name, env_idx)
                env_state["objects"][name] = {
                    "position_world": None if pos is None else np.array(pos, dtype=float).tolist(),
                    "orientation_wxyz_world": None if quat is None else np.array(quat, dtype=float).tolist(),
                }

            for name in articulation_names:
                pos, quat = self._get_object_pose_global(name, env_idx)
                env_state["articulations"][name] = {
                    "position_world": None if pos is None else np.array(pos, dtype=float).tolist(),
                    "orientation_wxyz_world": None if quat is None else np.array(quat, dtype=float).tolist(),
                }

            all_env_states.append(env_state)

        return all_env_states

    def build_episode_initial_metadata(self, task_yaml: Dict[str, Any]) -> Dict[str, Any]:
        """Build metadata snapshot that will be written into per-episode JSON."""
        texture_cfg = dict((task_yaml or {}).get("texture_randomization", {}) or {})
        texture_record = dict(getattr(self, "_latest_texture_randomization", {}) or {})
        if not texture_record:
            texture_record = {"enabled": False, "assignments": []}

        randomization: Dict[str, Any] = {
            "enable_random_lights": bool((task_yaml or {}).get("enable_random_lights", False)),
            "enable_objects_randomization": bool((task_yaml or {}).get("enable_objects_randomization", False)),
            "enable_objects_randomization_curriculum": bool(
                (task_yaml or {}).get("enable_objects_randomization_curriculum", False)
            ),
            "texture_randomization_config": texture_cfg,
            "texture_randomization_applied": texture_record,
        }
        if randomization["enable_random_lights"]:
            randomization["light_state_snapshot"] = self._capture_light_state_snapshot()

        return {
            "initial_env_states": self._capture_episode_initial_env_states(task_yaml),
            "randomization": randomization,
        }

    def _get_env_origins(self) -> np.ndarray:
        try:
            return sq.get_env_origins(self.scene)
        except AttributeError:
            return sq.get_env_origins(self.env.env.scene)

    def get_object_pose(self, object_name: str, env_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        return sq.get_object_pose(object_name, self.scene, env_idx)

    def _get_object_pose_global(self, label: str, env_idx: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return sq.get_object_pose_global(label, env_idx, self.scene)

    def rebuild_scene_desc_from_yaml(self) -> Tuple[List[dict], str]:
        return sq.rebuild_scene_desc_from_yaml(self.task_config_path, self.yaml)

    # =========================================================================
    # Obstacle Management (Multi-Environment Support)
    # =========================================================================

    def _get_object_poses_for_env(self, env_idx: int) -> Dict[str, np.ndarray]:
        return sq.get_object_poses_for_env(env_idx, self.scene)


    def _settle_sim_and_update_obstacles(self):
        """
        Run physics steps to settle simulation and update obstacles for all environments.
        """
        action_dim = self.env.env.action_manager.total_action_dim
        
        # Settle physics
        for _ in range(10):
            action = torch.zeros(self.num_envs, action_dim, device=self.device)
            action[:, :] = self.robot.data.joint_pos[:self.num_envs, :action_dim]
            self.env.step(action)
        
        # Update obstacles for all environments
        print(f"  [INFO] Updating obstacles for {self.num_envs} environments...")
        for env_idx in range(self.num_envs):
            self._update_obstacles_for_env(env_idx)



    def execute_trajectory_batch(self, cmd_plans, gripper_states, desc="Executing", active_mask=None):
        self.motion_planner.execute_trajectory_batch(
            cmd_plans, gripper_states, self.env, self.robot, self.num_envs,
            recorder=self.recorder, desc=desc, active_mask=active_mask,
        )

    def control_gripper_batch(self, gripper_actions, steps=3, active_mask=None):
        self.motion_planner.control_gripper_batch(
            gripper_actions, self.env, self.robot, self.num_envs,
            recorder=self.recorder, steps=steps, active_mask=active_mask,
        )

    # =========================================================================
    # Vision Pipeline Methods
    # =========================================================================

    def capture_camera_data_batch(self) -> bool:
        """Capture and save RGB-D data for all environments."""
        return self.vision_pipeline.capture_camera_data(self.scene, self.num_envs)

    def generate_bbox_batch(self, target_label: str) -> bool:
        """Generate 3D BBoxes for all environments."""
        env_origins = self._get_env_origins()
        return self.vision_pipeline.generate_bbox(
            target_label=target_label,
            scene=self.scene,
            num_envs=self.num_envs,
            env_origins=env_origins,
        )

    def run_anygrasp_process(self) -> bool:
        """Run external AnyGrasp process via subprocess."""
        return self.vision_pipeline.run_anygrasp(self.num_envs)

    # =========================================================================
    # Coordinate system calculations
    # =========================================================================


    def quaternion_multiply_wxyz(self, q1, q2):
        """Quaternion multiplication q1 * q2, format [w, x, y, z], delegated to top-level function"""
        return _quat_multiply_wxyz(q1, q2)


    def normalize_ee_orientation(self, quat_wxyz: np.ndarray, reference_quat_wxyz: np.ndarray = None, mode: str = "pick") -> np.ndarray:
        """Resolve 180-degree gripper z-axis ambiguity; pick mode chooses closer rotation, place mode ensures x-axis z-component same sign"""
        return normalize_ee_orientation_wxyz(quat_wxyz, reference_quat_wxyz, mode=mode)

    # =========================================================================
    # Execution layer proxies (delegated to MotionPlanner)
    # =========================================================================

    def _init_ik_solver(self):
        self.motion_planner.init_ik_solver()

    def solve_ik_batch(self, target_positions, target_quaternions, current_joint_pos, current_joint_vel):
        return self.motion_planner.solve_ik_batch(
            target_positions, target_quaternions, current_joint_pos, current_joint_vel, self.robot,
        )

    def _slerp(self, q1, q2, t):
        from .motion_planner import MotionPlanner
        return MotionPlanner._slerp(q1, q2, t)

    def get_current_ee_pose(self):
        return self.motion_planner.get_current_ee_pose(
            self.env, self.num_envs, getattr(self, 'robot_type', None),
        )

    def execute_ik_motion(self, target_positions, target_quaternions, gripper_states,
                          interpolation_steps=30, desc="IK Motion", active_mask=None, step_debug_fn=None):
        return self.motion_planner.execute_ik_motion(
            target_positions, target_quaternions, gripper_states,
            self.env, self.robot, self.num_envs,
            robot_type=getattr(self, 'robot_type', None),
            recorder=self.recorder,
            interpolation_steps=interpolation_steps, desc=desc,
            active_mask=active_mask, step_debug_fn=step_debug_fn,
        )

    
    
    def apply_random_textures(self, texture_dir: str, targets: Dict[str, str], avoid_repeats: bool = False):
        """
        Randomly assign textures to specified meshes for each environment, supporting multiple objects.

        Args:
            texture_dir: Default texture directory path
            targets: Mapping from object name to mesh path template, using {i} for environment index
                Values can be a string (prim path) or dict (with prim_path and optional texture_dir)
                e.g.: {"table": "/World/envs/env_{i}/table/.../Cube"}
                or:   {"table": {"prim_path": "/World/envs/env_{i}/table/.../Cube",
                                 "texture_dir": "/path/to/table_textures"}}
            avoid_repeats: Shuffle each target's texture pool and exhaust it before
                reusing a texture. Useful for comparing repeated evaluation trials.
        """
        stage = omni.usd.get_context().get_stage()
        assignments: List[Dict[str, Any]] = []

        if not stage.GetPrimAtPath("/Looks").IsValid():
            UsdGeom.Scope.Define(stage, "/Looks")

        for target_name, target_cfg in targets.items():
            if isinstance(target_cfg, dict):
                mesh_prim_template = target_cfg["prim_path"]
                target_texture_dir = target_cfg.get("texture_dir", texture_dir)
            else:
                mesh_prim_template = target_cfg
                target_texture_dir = texture_dir

            texture_files = sorted(f for f in os.listdir(target_texture_dir) if f.endswith(".png"))
            if not texture_files:
                print(f"[WARNING] No textures found: {target_texture_dir} (target: {target_name})")
                assignments.append(
                    {
                        "target": target_name,
                        "status": "no_texture_files",
                        "texture_dir": target_texture_dir,
                    }
                )
                continue

            for i in range(self.num_envs):
                if avoid_repeats:
                    if not hasattr(self, "_texture_shuffle_bags"):
                        self._texture_shuffle_bags = {}
                    pool_key = (target_name, os.path.abspath(target_texture_dir), tuple(texture_files))
                    pool = self._texture_shuffle_bags.setdefault(pool_key, {"remaining": [], "last": None})
                    if not pool["remaining"]:
                        pool["remaining"] = list(texture_files)
                        random.shuffle(pool["remaining"])
                        # Avoid an adjacent repeat across two shuffled cycles.
                        if len(pool["remaining"]) > 1 and pool["remaining"][-1] == pool["last"]:
                            pool["remaining"][0], pool["remaining"][-1] = pool["remaining"][-1], pool["remaining"][0]
                    tex_file = pool["remaining"].pop()
                    pool["last"] = tex_file
                else:
                    tex_file = random.choice(texture_files)
                texture_path = os.path.join(target_texture_dir, tex_file)

                mtl_path = f"/Looks/RandomMaterial_{target_name}_env_{i}"
                shader_path = f"{mtl_path}/Shader"

                material = UsdShade.Material.Define(stage, mtl_path)
                shader = UsdShade.Shader.Define(stage, shader_path)
                shader.SetSourceAsset("OmniPBR.mdl", "mdl")
                shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")

                diffuse_input = shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset)
                diffuse_input.Set(Sdf.AssetPath(texture_path))

                shader_output = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
                material.CreateSurfaceOutput("mdl").ConnectToSource(shader_output)
                material.CreateVolumeOutput("mdl").ConnectToSource(shader_output)
                material.CreateDisplacementOutput("mdl").ConnectToSource(shader_output)

                mesh_prim_path = mesh_prim_template.format(i=i)
                mesh_prim = stage.GetPrimAtPath(mesh_prim_path)
                if mesh_prim.IsValid():
                    UsdShade.MaterialBindingAPI(mesh_prim).Bind(
                        material, UsdShade.Tokens.strongerThanDescendants
                    )
                    assignments.append(
                        {
                            "target": target_name,
                            "env_idx": i,
                            "prim_path": mesh_prim_path,
                            "texture_dir": target_texture_dir,
                            "texture_file": tex_file,
                            "texture_path": texture_path,
                            "status": "bound",
                        }
                    )
                else:
                    print(f"[WARNING] Prim not found: {mesh_prim_path}")
                    assignments.append(
                        {
                            "target": target_name,
                            "env_idx": i,
                            "prim_path": mesh_prim_path,
                            "texture_dir": target_texture_dir,
                            "texture_file": tex_file,
                            "texture_path": texture_path,
                            "status": "prim_not_found",
                        }
                    )

            print(f"[INFO] Randomly assigned textures for {target_name} across {self.num_envs} environments")

        self._latest_texture_randomization = {
            "enabled": True,
            "default_texture_dir": texture_dir,
            "assignments": assignments,
        }
        return self._latest_texture_randomization
