from __future__ import annotations
from typing import TYPE_CHECKING, Optional
import torch
import numpy as np
import random
import re
import isaacsim.core.utils.stage as stage_utils

# Import Semantics API
try:
    from pxr import Sdf, Semantics
except ImportError:
    from pxr import Semantics

from isaaclab.sim.utils import safe_set_attribute_on_usd_prim

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.assets import RigidObject, Articulation
    
def quat_mul(q1, q2):
    """Quaternion multiplication, format [w, x, y, z] (Isaac Lab convention)."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)

def add_semantic_labels_to_scene(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    prim_path_patterns: dict[str, str],
):
    """Add semantic labels to prims in the scene based on path patterns.
    
    Args:
        env: The environment instance.
        env_ids: Environment IDs to apply labels to. None for startup mode (all envs).
        prim_path_patterns: Dictionary mapping regex path patterns to semantic labels.
                           Example: {r".*/table.*": "table", r".*/block.*": "cube"}
    """
    stage = stage_utils.get_current_stage()
    
    # If env_ids is None (startup mode), process all environments
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    
    label_count = {}
    
    for env_id in env_ids:
        env_id_int = int(env_id)
        env_path = f"/World/envs/env_{env_id_int}"
        
        # Iterate through all prims in the environment
        env_prim = stage.GetPrimAtPath(env_path)
        if not env_prim.IsValid():
            continue
            
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            
            # Check if prim is under current environment
            if not prim_path.startswith(env_path):
                continue
            
            # Match against all patterns
            for pattern, label in prim_path_patterns.items():
                # Replace ENV_REGEX_NS if present in pattern
                actual_pattern = pattern.replace("{ENV_REGEX_NS}", env_path)
                
                if re.search(actual_pattern, prim_path):
                    # Set semantic label using Semantics API (same as Isaac Lab's implementation)
                    # Sanitize label (replace spaces with underscores)
                    label_sanitized = label.replace(" ", "_")
                    # Create instance name: class_<label>
                    instance_name = f"class_{label_sanitized}"
                    
                    # Apply Semantics API
                    sem = Semantics.SemanticsAPI.Apply(prim, instance_name)
                    # Create and set semantic type and data attributes
                    sem.CreateSemanticTypeAttr()
                    sem.CreateSemanticDataAttr()
                    sem.GetSemanticTypeAttr().Set("class")
                    sem.GetSemanticDataAttr().Set(label)
                    
                    # Count labels
                    label_count[label] = label_count.get(label, 0) + 1
                    break  # Only apply first matching label

def set_rigid_object_pose(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    position: Optional[np.ndarray] = None,
    orientation: Optional[dict] = None,
    position_range: Optional[dict] = None,
    orientation_range: Optional[dict] = None,
    curriculum_config: Optional[dict] = None,
):
    rigid_object: RigidObject = env.scene[asset_cfg.name]
    num_resets = len(env_ids)
    env_origins = env.scene.env_origins[env_ids]  # Shape: [num_resets, 3]
    
    if curriculum_config is not None:
        current_it = env.common_step_counter // 24
        initial_ranges = curriculum_config['initial_ranges']
        final_ranges = curriculum_config['final_ranges']
        start_it = curriculum_config['start_it']
        end_it = curriculum_config['end_it']
        if current_it < start_it:
            progress = 0.0
        elif current_it >= end_it:
            progress = 1.0
        else:
            progress = (current_it - start_it) / (end_it - start_it)
        
        position_range = {}
        for axis in ['x', 'y', 'z']:
            if axis in initial_ranges and axis in final_ranges:
                init_range = initial_ranges[axis]
                final_range = final_ranges[axis]
                current_min = init_range[0] + progress * (final_range[0] - init_range[0])
                current_max = init_range[1] + progress * (final_range[1] - init_range[1])
                position_range[axis] = (current_min, current_max)
        print("current_it", current_it)

    if position is not None:
        if isinstance(position, np.ndarray):
            position = torch.from_numpy(position).float().to(env.device)
        elif isinstance(position, (list, tuple)):
            position = torch.tensor(position, dtype=torch.float32, device=env.device)
        
        new_pos = position.unsqueeze(0).expand(num_resets, -1) + env_origins
    
    elif position_range is not None:
        print("Applying random position within range:", position_range)
        new_pos = torch.zeros(num_resets, 3, device=env.device)
        new_pos[:, 0] = torch.rand(num_resets, device=env.device) * \
                        (position_range['x'][1] - position_range['x'][0]) + position_range['x'][0]
        new_pos[:, 1] = torch.rand(num_resets, device=env.device) * \
                        (position_range['y'][1] - position_range['y'][0]) + position_range['y'][0]
        new_pos[:, 2] = torch.rand(num_resets, device=env.device) * \
                        (position_range['z'][1] - position_range['z'][0]) + position_range['z'][0]
        new_pos = new_pos + env_origins
    else:
        new_pos = rigid_object.data.default_root_state[env_ids, :3].clone() + env_origins
    
    if orientation is not None:
        # Use specified orientation
        quat_orig = rigid_object.data.default_root_state[env_ids, 3:7].clone()
        half_yaw = orientation.get('yaw', 0.0) * 0.5
        if isinstance(half_yaw, (int, float)):
            half_yaw = torch.tensor(half_yaw, dtype=torch.float32, device=env.device)
        q_rot = torch.zeros_like(quat_orig)
        q_rot[:, 0] = torch.cos(half_yaw)
        q_rot[:, 3] = torch.sin(half_yaw)
        quat = quat_mul(quat_orig, q_rot)
        
        quat = quat.unsqueeze(0).expand(num_resets, -1) if quat.dim() == 1 else quat
        
    elif orientation_range is not None:
        quat_orig = rigid_object.data.default_root_state[env_ids, 3:7].clone()
        yaw_range = orientation_range.get('yaw', (0.0, 0.0))
        delta_yaw = torch.rand_like(quat_orig[:, 0]) * (yaw_range[1] - yaw_range[0]) + yaw_range[0]
        half_yaw = delta_yaw * 0.5
        q_rot = torch.zeros_like(quat_orig)
        q_rot[:, 0] = torch.cos(half_yaw)
        q_rot[:, 3] = torch.sin(half_yaw)
        quat = quat_mul(quat_orig, q_rot)
    else:
        quat = rigid_object.data.default_root_state[env_ids, 3:7].clone()
    
    # Print object 6D pose after initialization (world frame)
    from isaaclab.utils.math import euler_xyz_from_quat
    roll, pitch, yaw = euler_xyz_from_quat(quat)
    rad2deg = 180.0 / np.pi
    for i, eid in enumerate(env_ids.tolist()):
        p = new_pos[i].tolist()
        q = quat[i].tolist()
        r_deg = (roll[i].item() * rad2deg, pitch[i].item() * rad2deg, yaw[i].item() * rad2deg)
        print(
            f"[init pose] env={eid} obj={asset_cfg.name} "
            f"pos=({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}) "
            f"quat_wxyz=({q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}) "
            f"rpy_deg=({r_deg[0]:.2f}, {r_deg[1]:.2f}, {r_deg[2]:.2f})"
        )

    rigid_object.write_root_pose_to_sim(
        torch.cat([new_pos, quat], dim=-1),
        env_ids=env_ids
    )

    # Save object initial position to env (used for reward computation)
    # Note: new_pos here is an absolute position in the world frame
    # and needs to be converted to a position relative to the robot base
    if hasattr(env, 'object_init_x'):
        robot = env.scene["robot"]
        # Get robot base world position
        robot_pos_w = robot.data.root_pos_w[env_ids]
        # Compute relative position
        rel_pos = new_pos - robot_pos_w
        env.object_init_x[env_ids] = rel_pos[:, 0]
        env.object_init_y[env_ids] = rel_pos[:, 1]
        env.object_init_z[env_ids] = new_pos[:, 2]  # z uses absolute height


def set_articulation_pose(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    position: Optional[np.ndarray] = None,
    orientation: Optional[dict] = None,
    position_range: Optional[dict] = None,
    orientation_range: Optional[dict] = None,
):
    """Set or randomize the pose of an articulation object (e.g. microwave).

    Similar to set_rigid_object_pose but operates on Articulation assets.
    Also resets joint positions and velocities to their default values.
    """
    articulation: Articulation = env.scene[asset_cfg.name]
    num_resets = len(env_ids)
    env_origins = env.scene.env_origins[env_ids]

    # --- Position ---
    if position is not None:
        if isinstance(position, np.ndarray):
            position = torch.from_numpy(position).float().to(env.device)
        elif isinstance(position, (list, tuple)):
            position = torch.tensor(position, dtype=torch.float32, device=env.device)
        new_pos = position.unsqueeze(0).expand(num_resets, -1) + env_origins
    elif position_range is not None:
        new_pos = torch.zeros(num_resets, 3, device=env.device)
        new_pos[:, 0] = torch.rand(num_resets, device=env.device) * \
                        (position_range['x'][1] - position_range['x'][0]) + position_range['x'][0]
        new_pos[:, 1] = torch.rand(num_resets, device=env.device) * \
                        (position_range['y'][1] - position_range['y'][0]) + position_range['y'][0]
        new_pos[:, 2] = torch.rand(num_resets, device=env.device) * \
                        (position_range['z'][1] - position_range['z'][0]) + position_range['z'][0]
        new_pos = new_pos + env_origins
    else:
        new_pos = articulation.data.default_root_state[env_ids, :3].clone() + env_origins

    # --- Orientation ---
    if orientation is not None:
        quat_orig = articulation.data.default_root_state[env_ids, 3:7].clone()
        half_yaw = orientation.get('yaw', 0.0) * 0.5
        if isinstance(half_yaw, (int, float)):
            half_yaw = torch.tensor(half_yaw, dtype=torch.float32, device=env.device)
        q_rot = torch.zeros_like(quat_orig)
        q_rot[:, 0] = torch.cos(half_yaw)
        q_rot[:, 3] = torch.sin(half_yaw)
        quat = quat_mul(quat_orig, q_rot)
        quat = quat.unsqueeze(0).expand(num_resets, -1) if quat.dim() == 1 else quat
    elif orientation_range is not None:
        quat_orig = articulation.data.default_root_state[env_ids, 3:7].clone()
        yaw_range = orientation_range.get('yaw', (0.0, 0.0))
        delta_yaw = torch.rand_like(quat_orig[:, 0]) * (yaw_range[1] - yaw_range[0]) + yaw_range[0]
        half_yaw = delta_yaw * 0.5
        q_rot = torch.zeros_like(quat_orig)
        q_rot[:, 0] = torch.cos(half_yaw)
        q_rot[:, 3] = torch.sin(half_yaw)
        quat = quat_mul(quat_orig, q_rot)
    else:
        quat = articulation.data.default_root_state[env_ids, 3:7].clone()

    # Write root pose
    articulation.write_root_pose_to_sim(
        torch.cat([new_pos, quat], dim=-1),
        env_ids=env_ids
    )
    articulation.write_root_velocity_to_sim(
        articulation.data.default_root_state[env_ids, 7:13].clone(), env_ids=env_ids
    )

    # Reset joint positions and velocities to defaults
    joint_pos = articulation.data.default_joint_pos[env_ids].clone()
    joint_vel = articulation.data.default_joint_vel[env_ids].clone()
    articulation.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    articulation.set_joint_position_target(joint_pos, env_ids=env_ids)
    articulation.set_joint_velocity_target(joint_vel, env_ids=env_ids)


def configure_task_lighting(env, env_ids=None):
    """Keep only the requested task light; disable lights embedded in assets."""
    import omni.usd
    from pxr import UsdLux
    stage = omni.usd.get_context().get_stage()
    active = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdLux.LightAPI):
            continue
        if prim.GetName() in {"task_point_light", "task_dome_light"}:
            active.append({"path": str(prim.GetPath()), "type": prim.GetTypeName(),
                           "intensity": UsdLux.LightAPI(prim).GetIntensityAttr().Get()})
        else:
            UsdLux.LightAPI(prim).GetIntensityAttr().Set(0.0)
    print(f"[TaskLighting] Active sources: {active}")


def randomize_lights(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    light_configs: list[dict],
):
    """Randomize lighting by activating one random light type per reset.

    Each reset picks one light from the list, randomizes its intensity/color/position,
    and disables all others (intensity=0). Since lights are global (/World/), they
    affect all envs uniformly.

    Args:
        env: The environment instance.
        env_ids: Environment IDs (unused, lights are global).
        light_configs: List of dicts, each with:
            - prim_path: str, e.g. "/World/light_sphere"
            - intensity_range: (min, max)
            - color_range: ((r_min, g_min, b_min), (r_max, g_max, b_max))
            - position_range: ((x_min, y_min, z_min), (x_max, y_max, z_max)) (optional)
    """
    stage = stage_utils.get_current_stage()

    # Pick one light to activate
    chosen_idx = random.randint(0, len(light_configs) - 1)

    for idx, config in enumerate(light_configs):
        prim_path = config["prim_path"]
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"[randomize_lights] WARNING: prim not found: {prim_path}")
            continue

        # Disable non-chosen lights
        if idx != chosen_idx:
            safe_set_attribute_on_usd_prim(prim, "inputs:intensity", 0.0, camel_case=True)
            continue

        # Random intensity
        intensity_range = config["intensity_range"]
        intensity = random.uniform(intensity_range[0], intensity_range[1])
        safe_set_attribute_on_usd_prim(prim, "inputs:intensity", intensity, camel_case=True)

        # Random color
        c_min, c_max = config["color_range"]
        color = (
            random.uniform(c_min[0], c_max[0]),
            random.uniform(c_min[1], c_max[1]),
            random.uniform(c_min[2], c_max[2]),
        )
        safe_set_attribute_on_usd_prim(prim, "inputs:color", color, camel_case=True)

        # Random position (if specified)
        # Use safe_set_attribute_on_usd_prim so changes propagate through Fabric
        position_range = config.get("position_range", None)
        pos = None
        if position_range is not None:
            p_min, p_max = position_range
            pos = (
                random.uniform(p_min[0], p_max[0]),
                random.uniform(p_min[1], p_max[1]),
                random.uniform(p_min[2], p_max[2]),
            )
            safe_set_attribute_on_usd_prim(prim, "xformOp:translate", pos, camel_case=False)

        # Random direction / rotation (if specified, useful for distant lights)
        direction_range = config.get("direction_range", None)
        direction = None
        if direction_range is not None:
            d_min, d_max = direction_range
            direction = (
                random.uniform(d_min[0], d_max[0]),
                random.uniform(d_min[1], d_max[1]),
                random.uniform(d_min[2], d_max[2]),
            )
            safe_set_attribute_on_usd_prim(prim, "xformOp:rotateXYZ", direction, camel_case=False)

        # Read back actual values from prim for verification
        actual_intensity = prim.GetAttribute("inputs:intensity").Get()
        actual_color = prim.GetAttribute("inputs:color").Get()
        actual_pos = prim.GetAttribute("xformOp:translate").Get()
        actual_rot = prim.GetAttribute("xformOp:rotateXYZ").Get()
        print(f"[randomize_lights] Activated: {prim_path}")
        print(f"  intensity:  set={intensity:.1f}, actual={actual_intensity}")
        print(f"  color:      set=({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f}), actual={actual_color}")
        print(f"  position:   set={pos}, actual={actual_pos}")
        print(f"  direction:  set={direction}, actual={actual_rot}")
