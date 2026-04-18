"""
Usage:
python collect_data.py --episodes 10 --headless --scene 1 --robot franka_panda_robotiq85.yml
"""

import tyro
import os
import argparse
import gymnasium as gym
import torch
import time
import numpy as np
import cv2
import mediapy
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import sys
from isaaclab_tasks.real2sim import REAL2SIM_ROOT_DIR
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="CuRobo motion planning in Real2Sim environment.")
AppLauncher.add_app_launcher_args(parser)


# NOW import CuRobo modules (after Isaac Sim is initialized)
CUROBO_PATH = Path(REAL2SIM_ROOT_DIR) / "curobo" / "src"
sys.path.insert(0, str(CUROBO_PATH))

def get_object_pose(env, object_name: str):
    """Get object position and quaternion in world frame"""
    from pxr import UsdGeom
    
    scene_path = f"/World/envs/env_0/scene/{object_name}"
    stage = env.env.sim.stage
    prim = stage.GetPrimAtPath(scene_path)
    
    if not prim.IsValid():
        print(f"Warning: Object '{object_name}' not found at {scene_path}")
        return None, None
    
    # 获取世界坐标系下的变换
    xform = UsdGeom.Xformable(prim)
    world_transform = xform.ComputeLocalToWorldTransform(0)
    
    # 提取位置（世界坐标系）
    translation = world_transform.ExtractTranslation()
    position = np.array([translation[0], translation[1], translation[2]])
    
    # 提取四元数（世界坐标系）
    rotation = world_transform.ExtractRotationQuat()
    quaternion = np.array([
        rotation.GetReal(),           # w
        rotation.GetImaginary()[0],   # x
        rotation.GetImaginary()[1],   # y
        rotation.GetImaginary()[2]    # z
    ])
    
    return position, quaternion

def compute_pick_pose(object_pos, object_quat, grasp_offset: np.ndarray = np.array([0, 0, 0.15])):
    """
    Compute pick pose based on object position
    
    Args:
        object_pos: Object position [x, y, z]
        object_quat: Object quaternion [w, x, y, z]
        grasp_offset: Offset from object center to grasp point (in object frame)
    
    Returns:
        pick_position, pick_quaternion
    """
    # 简单方式：在物体上方一定距离，夹爪朝下
    pick_position = object_pos.copy()
    pick_position[2] += grasp_offset[2]  # 上方偏移
    
    # 夹爪朝下的四元数 (gripper pointing down)
    pick_quaternion = np.array([0, 1, 0, 0])  # [w, x, y, z]
    
    return pick_position, pick_quaternion


def compute_place_pose(container_pos, container_quat, place_offset: np.ndarray = np.array([0, 0, 0.2])):
    """
    Compute place pose based on container position
    
    Args:
        container_pos: Container position [x, y, z]
        container_quat: Container quaternion [w, x, y, z]
        place_offset: Offset from container center to place point
    
    Returns:
        place_position, place_quaternion
    """
    place_position = container_pos.copy()
    place_position[2] += place_offset[2]  # 在容器上方
    
    # 夹爪朝下
    place_quaternion = np.array([0, 1, 0, 0])  # [w, x, y, z]
    
    return place_position, place_quaternion

def main(
    episodes: int = 10,
    headless: bool = True,
    scene: int = 1,
    robot_cfg_name: str = "franka_fr3.yml",
    use_motion_planning: bool = True,
):  
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    from curobo.geom.sdf.world import CollisionCheckerType
    from curobo.geom.types import WorldConfig
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.types.robot import JointState
    from curobo.types.state import JointState
    from curobo.util.logger import log_error, setup_curobo_logger
    from curobo.util.usd_helper import UsdHelper
    from curobo.util_file import (
        get_robot_configs_path,
        get_world_configs_path,
        join_path,
        load_yaml,
    )
    from curobo.wrap.reacher.motion_gen import (
        MotionGen,
        MotionGenConfig,
        MotionGenPlanConfig,
        PoseCostMetric,
    )

    def update_obstacles(motion_gen, world, robot_prim_path: str, ignored_objects: list):
        usd_help = UsdHelper()
        usd_help.load_stage(world.stage)
        ignore_list = ignored_objects
        obstacles = usd_help.get_obstacles_from_stage(
            only_paths=["/World"],
            reference_prim_path=robot_prim_path,
            ignore_substring=ignore_list,
        ).get_collision_check_world()

        print("len(obstacles.objects)", len(obstacles.objects))
        print("obstacles.objects", obstacles.objects)

        if obstacles is not None:
            motion_gen.update_world(obstacles)

    def setup_curobo_motion_gen(robot_cfg_name: str, world, robot, robot_prim_path: str, target_object_name: str):
        setup_curobo_logger("warn")
        
        tensor_args = TensorDeviceType()
        robot_cfg_path = get_robot_configs_path()
        robot_cfg = load_yaml(join_path(robot_cfg_path, robot_cfg_name))["robot_cfg"]
        
        # Setup world configuration
        # world_cfg_table = WorldConfig.from_dict(
        #     load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
        # )
        # world_cfg_table.cuboid[0].pose[2] -= 0.02
        # world_cfg1 = WorldConfig.from_dict(
        #     load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
        # ).get_mesh_world()
        # world_cfg1.mesh[0].name += "_mesh"
        # world_cfg1.mesh[0].pose[2] = -10.5
        # world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)
        world_cfg = WorldConfig()
        
        # Motion generation configuration
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            world_cfg,
            tensor_args,
            collision_checker_type=CollisionCheckerType.MESH,
            num_trajopt_seeds=12,
            num_graph_seeds=12,
            interpolation_dt=0.05,
            collision_cache={"obb": 30, "mesh": 100},
            # collision_cache={"obb": 30},
            optimize_dt=True,
            trajopt_tsteps=16,
            collision_activation_distance=0.08,
        )
        
        motion_gen = MotionGen(motion_gen_config)
        motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)

        target_object_name_path = f"/World/envs/env_0/scene/{target_object_name}"
        ignored_objects = [robot_prim_path,"/World/defaultGroundPlane", "/World/envs/env_0/fr3"]
        
        update_obstacles(motion_gen, world, robot_prim_path, ignored_objects)
        
        return motion_gen, robot_cfg, tensor_args

    def plan_to_pose(
        motion_gen,
        robot,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        tensor_args,
        robot_cfg,
    ):
        """Plan motion to target pose using CuRobo"""
        joint_pos = robot.data.joint_pos[0].cpu().numpy()
        joint_vel = robot.data.joint_vel[0].cpu().numpy()
        
        if np.any(np.isnan(joint_pos)):
            return None, False
        
        # Get joint names from robot
        joint_names = robot.joint_names
        
        cu_js = JointState(
            position=tensor_args.to_device(joint_pos),
            velocity=tensor_args.to_device(joint_vel) * 0.0,
            acceleration=tensor_args.to_device(joint_vel) * 0.0,
            jerk=tensor_args.to_device(joint_vel) * 0.0,
            joint_names=joint_names,
        )
        
        cu_js = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)
        
        # Create target pose
        ik_goal = Pose(
            position=tensor_args.to_device(target_pos),
            quaternion=tensor_args.to_device(target_quat),
        )
        
        # Plan
        plan_config = MotionGenPlanConfig(
            enable_graph=False,
            enable_graph_attempt=2,
            max_attempts=4,
            enable_finetune_trajopt=True,
            time_dilation_factor=0.5,
        )
        
        result = motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, plan_config)
        
        if result.success.item():
            cmd_plan = result.get_interpolated_plan()
            cmd_plan = motion_gen.get_full_js(cmd_plan)
            return cmd_plan, True
        else:
            print(f"Planning failed: {result.status}")
            return None, False

    import src.environments
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(
        "Real2Sim-v0",
        device=args_cli.device,
        num_envs=1,
        use_fabric=True,
    )

    env_cfg.disable_initialization()

    env_cfg.set_scene(scene)
    env = gym.make("Real2Sim-v0", cfg=env_cfg)

    obs, _ = env.reset()
    obs, _ = env.reset()

    scene_objects = {
        1: {"object": "_10_potted_meat_can", 
            "container": "_25_mug",
            "grasp_offset": np.array([0, 0, 0.1]),  # 抓取点偏移
            "place_offset": np.array([0, 0, 0.2]),  # 放置点偏移
            },
        2: {"object": "_10_potted_meat_can",
            "container": "_37_bowl",
            "grasp_offset": np.array([0, 0, 0.15]),
            "place_offset": np.array([0, 0, 0.18]),
            },
        3: {"object": "_13_banana",
            "container": "_51_large_clamp",
            "grasp_offset": np.array([0, 0, 0.10]),
            "place_offset": np.array([0, 0, 0.15]),
            },
    }
    
    config = scene_objects.get(scene)
    if not config:
        raise ValueError(f"Scene {scene} not configured")
    
    target_object = config["object"]
    target_container = config["container"]
    
    # Get robot and world references
    world = env.env.sim
    robot_articulation = env.env.scene["robot"]
    robot_prim_path = str(robot_articulation.cfg.prim_path).replace(".*", "0")
    
    # Initialize CuRobo motion generation
    motion_gen = None
    robot_cfg = None
    tensor_args = None
    
    if use_motion_planning:
        print("Initializing CuRobo motion planning...")
        motion_gen, robot_cfg, tensor_args = setup_curobo_motion_gen(
            robot_cfg_name, world, robot_articulation, robot_prim_path, target_object_name=target_object
        )
        print("CuRobo motion planning ready!")
        j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    
    # pick_pos, pick_quat = np.array([0.5, 0.0, 0.3]), np.array([0, 1, 0, 0])
    # place_pos, place_quat = np.array([0.2, 0.0, 0.4]), np.array([0, 1, 0, 0])

    pick_pos, pick_quat = get_object_pose(env, target_object)
    place_pos, place_quat = get_object_pose(env, target_container)
    pick_position, pick_quaternion = compute_pick_pose(
        pick_pos, 
        pick_quat, 
        grasp_offset=config["grasp_offset"])
    place_position, place_quaternion = compute_place_pose(
        place_pos,
        place_quat,
        place_offset=config["place_offset"])
    pick_pose = {
        "position": pick_position,
        "quaternion": pick_quaternion,
    }
    place_pose = {
        "position": place_position,
        "quaternion": place_quaternion,
    }
    print(f"Pick pose: {pick_pose}")
    print(f"Place pose: {place_pose}")

    # Video recording setup
    video_dir = Path("runs") / datetime.now().strftime("%Y-%m-%d") / datetime.now().strftime("%H-%M-%S")
    video_dir.mkdir(parents=True, exist_ok=True)
    video = []
    
    max_steps = env.env.max_episode_length
    
    with torch.no_grad():
        for ep in range(episodes):
            obs, _ = env.reset()
            
            if use_motion_planning and motion_gen is not None:
                # Phase 1: Move to pick position
                print(f"Episode {ep+1}: Planning to pick position...")
                cmd_plan, success = plan_to_pose(
                    motion_gen,
                    robot_articulation,
                    pick_pose["position"],
                    pick_pose["quaternion"],
                    tensor_args,
                    robot_cfg,
                )
                
                if success:
                    # Execute planned trajectory
                    common_js_names = []
                    idx_list = []
                    robot_joint_names = robot_articulation.joint_names
                    
                    for x in robot_joint_names:
                        if x in cmd_plan.joint_names:
                            idx_list.append(robot_joint_names.index(x))
                            common_js_names.append(x)
                    idx_tensor = torch.tensor(idx_list[:env.env.action_manager.total_action_dim], 
                          dtype=torch.long)
                    
                    cmd_plan = cmd_plan.get_ordered_joint_state(common_js_names)
                    
                    for i in tqdm(range(len(cmd_plan.position)), desc=f"Episode {ep+1} - Moving to pick"):
                        cmd_state = cmd_plan[i]
                        past_cmd = cmd_state.clone()
                        
                        # Convert to action
                        action = torch.zeros(1, env.env.action_manager.total_action_dim)
                        action[0, idx_tensor] = past_cmd.position[:len(idx_tensor)].cpu()
                        action[0, -1] = 0.0  # Open gripper
                        # action = action[0,:-1].unsqueeze(0)
                        obs, _, term, trunc, _ = env.step(action)
                        
                        if term or trunc:
                            break
                    
                    # Close gripper
                    print("Closing gripper...")
                    for _ in range(10):
                        action = torch.zeros(1, env.env.action_manager.total_action_dim)
                        action[0, idx_tensor] = past_cmd.position[:len(idx_tensor)].cpu()
                        action[0, -1] = np.pi / 4 # Close gripper
                        obs, _, term, trunc, _ = env.step(action)

                    # Phase 2: Move to place position
                    print(f"Episode {ep+1}: Planning to place position...")
                    cmd_plan, success = plan_to_pose(
                        motion_gen,
                        robot_articulation,
                        place_pose["position"],
                        place_pose["quaternion"],
                        tensor_args,
                        robot_cfg,
                    )
                    
                    if success:
                        cmd_plan = cmd_plan.get_ordered_joint_state(common_js_names)
                        for i in tqdm(range(len(cmd_plan.position)), desc=f"Episode {ep+1} - Moving to place"):
                            cmd_state = cmd_plan[i]
                            past_cmd = cmd_state.clone()
                            
                            action = torch.zeros(1, env.env.action_manager.total_action_dim)
                            action[0, idx_tensor] = past_cmd.position[:len(idx_tensor)].cpu()
                            action[0, -1] = np.pi / 4  # Keep gripper closed
                            
                            obs, _, term, trunc, _ = env.step(action)
                            
                            if term or trunc:
                                break
                        
                        # Open gripper
                        print("Opening gripper...")
                        for _ in range(10):
                            action = torch.zeros(1, env.env.action_manager.total_action_dim)
                            action[0, idx_tensor] = past_cmd.position[:len(idx_tensor)].cpu()
                            action[0, -1] = 0.0 # Open gripper
                            obs, _, term, trunc, _ = env.step(action)
                
            else:
                # Fallback to manual actions
                actions_test = torch.tensor(
                    [[0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04]],
                    device=env.env.device
                )
                
                for i in tqdm(range(max_steps), desc=f"Episode {ep+1}/{episodes}"):
                    obs, _, term, trunc, _ = env.step(actions_test)
                    
                    if term or trunc:
                        break
            
            print(f"Episode {ep+1} completed!")
    
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    tyro.cli(main)