"""
Real2Sim2Real Diffusion Policy 服务器 (GPU 机器).

架构:
    [真机 client] --图像--> [本脚本: 模型推理 + 仿真 env.step] --关节角度--> [真机 client]

流程:
1. 接收真机客户端发来的图像 (front + wrist)
2. 用真机图像 + 仿真 qpos/qvel 输入模型, 得到 action
3. action 在仿真 env.step() 中执行 (IK 解算 + 物理仿真)
4. 读取仿真中的 7-DOF 关节角度 + gripper 状态
5. 返回给真机客户端, 真机按 JOINT_ABSOLUTE 运动

注意: Isaac Sim 的 env.step() 必须在主线程执行,
      所以 WebSocket 通信在后台线程, 通过队列与主线程同步.

用法:
    python scripts/imitation_learning/diffusion_policy/play_real2sim.py \
        --checkpoint /path/to/model_best.pth \
        --stat /path/to/dataset_stat.json \
        --task_config /path/to/generated_task_config.yaml \
        --task Pick_Place_banana-v0 \
        --horizon 300 \
        --port 8070
"""

import argparse
import json
import logging
import os
import sys
import time
import threading
import queue
from collections import deque

# Isaac Lab 启动
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Real2Sim2Real DP Server")
parser.add_argument("--task", type=str, default="Pick_Place_banana-v0", help="环境名称")
parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路径")
parser.add_argument("--stat", type=str, default=None, help="dataset_stat.json 路径")
parser.add_argument("--horizon", type=int, default=500, help="最大步数")
parser.add_argument("--port", type=int, default=8070, help="WebSocket 端口")
parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
parser.add_argument("--seed", type=int, default=None, help="随机种子")
parser.add_argument(
    "--task_config", type=str,
    default="source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/task_gen/config/sceneconfig/generated_task_config.yaml",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- 以下在 Isaac Sim 启动后导入 ---
import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn.functional as F
import yaml

import isaaclab_tasks.real2sim.tasks  # noqa: F401

# Diffusion Policy
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from network import DiffusionPolicy
from train import DDPMScheduler

# WebSocket (同步版, 不用 asyncio)
_deploy_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "lmy_roby", "deployment")
sys.path.insert(0, os.path.abspath(_deploy_dir))
from websocket import msgpack_numpy


# ============================================================================
# 工具函数
# ============================================================================

def load_norm_params(stat_path):
    with open(stat_path) as f:
        stat = json.load(f)
    params = stat["_minmax_params"]
    result = {}
    for key in params:
        result[key] = {
            "min": np.array(params[key]["min"], dtype=np.float32),
            "max": np.array(params[key]["max"], dtype=np.float32),
        }
    return result


def minmax_normalize(data, dmin, dmax):
    drange = dmax - dmin
    drange = np.where(drange < 1e-8, 1.0, drange)
    return 2.0 * (data - dmin) / drange - 1.0


def minmax_denormalize(data, dmin, dmax):
    drange = dmax - dmin
    return (data + 1.0) / 2.0 * drange + dmin


def process_real_image(image_np: np.ndarray, target_size: int = 256) -> torch.Tensor:
    """处理真机 RGB 图像: center crop + resize -> (3, H, W) float [0,1]."""
    image = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    C, H, W = image.shape
    crop_size = min(H, W)
    y0 = (H - crop_size) // 2
    x0 = (W - crop_size) // 2
    image = image[:, y0:y0 + crop_size, x0:x0 + crop_size]
    if crop_size != target_size:
        image = F.interpolate(
            image.unsqueeze(0), size=(target_size, target_size),
            mode="bilinear", align_corners=False,
        ).squeeze(0)
    return image


@torch.no_grad()
def ddpm_sample(model, scheduler, obs_images, obs_lowdim, device):
    B = 1
    T_a = model.pred_horizon
    D_act = model.action_dim
    global_cond = model.encode_obs(obs_images, obs_lowdim)
    x = torch.randn(B, T_a, D_act, device=device)
    for t in reversed(range(scheduler.num_steps)):
        timestep = torch.full((B,), t, device=device, dtype=torch.long)
        noise_pred = model.noise_pred_net(x, timestep, global_cond)
        x = scheduler.step(noise_pred, t, x)
    return x.squeeze(0).cpu().numpy()


# ============================================================================
# 同步 WebSocket 服务器线程 (仅做通信, 不做推理)
# ============================================================================

def ws_server_thread(host, port, obs_queue, result_queue, stop_event):
    """WebSocket 服务器线程: 接收 obs 放入队列, 等待主线程处理结果后发回客户端.

    使用 websockets.sync.server, 不涉及 asyncio.
    """
    import websockets.sync.server as ws_server

    packer = msgpack_numpy.Packer()

    def handler(websocket):
        print(f"[WS] 客户端连接: {websocket.remote_address}")
        # 发送 metadata (与 WebsocketClientPolicy 协议一致)
        websocket.send(packer.pack({}))

        while not stop_event.is_set():
            try:
                raw = websocket.recv(timeout=1.0)
            except TimeoutError:
                continue
            except Exception:
                break

            obs = msgpack_numpy.unpackb(raw)

            # 把 obs 放入队列, 等待主线程处理
            obs_queue.put(obs)

            # 等待主线程返回结果
            try:
                result = result_queue.get(timeout=30.0)
            except queue.Empty:
                print("[WS] 等待主线程超时!")
                result = {"error": "timeout"}

            websocket.send(packer.pack(result))

        print(f"[WS] 客户端断开")

    print(f"[WS] 同步 WebSocket 服务器启动: {host}:{port}")
    with ws_server.serve(handler, host, port, compression=None, max_size=None) as server:
        server.serve_forever()


# ============================================================================
# Main
# ============================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 加载模型 ---
    print(f"[INFO] 加载 checkpoint: {args_cli.checkpoint}")
    ckpt = torch.load(args_cli.checkpoint, map_location=device, weights_only=False)
    model_args = ckpt.get("args", {})
    model = DiffusionPolicy(
        action_dim=model_args.get("action_dim", 7),
        obs_horizon=model_args.get("obs_horizon", 2),
        pred_horizon=model_args.get("pred_horizon", 16),
        lowdim_dim=model_args.get("lowdim_dim", 14),
        img_feature_dim=model_args.get("img_feature_dim", 256),
        lowdim_feature_dim=model_args.get("lowdim_feature_dim", 64),
        down_dims=tuple(model_args.get("down_dims", [256, 512, 1024])),
        pretrained_resnet=False,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[INFO] 模型加载完成, epoch={ckpt.get('epoch', 'unknown')}")

    scheduler = DDPMScheduler(
        num_diffusion_steps=model_args.get("num_diffusion_steps", 100),
    )
    action_horizon = model_args.get("action_horizon", 3)
    obs_horizon = model_args.get("obs_horizon", 2)

    if args_cli.seed is not None:
        torch.manual_seed(args_cli.seed)
        np.random.seed(args_cli.seed)
        random.seed(args_cli.seed)

    norm_params = None
    if args_cli.stat is not None:
        norm_params = load_norm_params(args_cli.stat)

    # --- 仿真环境 ---
    from isaaclab_tasks.real2sim.tasks.pick_r2s_env_cfg import create_pick_env_cfg_from_yaml
    from isaaclab_tasks.real2sim.tasks.config.franka_fr3.ourgripper.ik_rel_vis_env_cfg import FrankaFr3OurGripperIKRelVisPickEnvCfg

    env_cfg = create_pick_env_cfg_from_yaml(
        yaml_path=args_cli.task_config,
        env_cfg_class=FrankaFr3OurGripperIKRelVisPickEnvCfg,
        num_envs=1,
        device=args_cli.device,
    )
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.terminations.time_out = None
    if hasattr(env_cfg.terminations, "object_dropping"):
        env_cfg.terminations.object_dropping = None
    env_cfg.recorders = None
    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    print("[INFO] 仿真环境创建完成")

    # --- 观测缓冲区 ---
    obs_img_buffer = deque(maxlen=obs_horizon)
    obs_low_buffer = deque(maxlen=obs_horizon)

    # --- 仿真状态 ---
    obs_dict, _ = env.reset()
    obs_dict, _, _, _, _ = env.step(
        torch.zeros(1, env.action_space.shape[1], device=device)
    )
    step = 1
    initialized = True
    print("[INFO] 仿真环境初始化完成")

    # --- action buffer (每次推理填充 action_horizon 步, 每收到一帧消耗 1 步) ---
    action_buffer = deque()

    # --- 启动 WebSocket 通信线程 ---
    obs_q = queue.Queue(maxsize=1)
    result_q = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    ws_t = threading.Thread(
        target=ws_server_thread,
        args=(args_cli.host, args_cli.port, obs_q, result_q, stop_event),
        daemon=True,
    )
    ws_t.start()

    # --- 主循环: 在主线程处理 obs → 推理 → env.step → 返回 joint ---
    print("[INFO] 主线程进入处理循环, 等待客户端发送 obs...")

    try:
        while not stop_event.is_set():
            # 从 WS 线程获取 obs (阻塞, 带超时)
            try:
                obs = obs_q.get(timeout=1.0)
            except queue.Empty:
                continue

            # --- 处理 reset ---
            if obs.get("reset", False):
                obs_img_buffer.clear()
                obs_low_buffer.clear()
                action_buffer.clear()
                step = 0
                obs_dict, _ = env.reset()
                obs_dict, _, _, _, _ = env.step(
                    torch.zeros(1, env.action_space.shape[1], device=device)
                )
                step = 1
                initialized = True
                sim_qpos = obs_dict["policy"]["qpos"].squeeze().cpu().numpy()
                print(f"[INFO] 仿真已重置, 初始 qpos: {np.round(sim_qpos, 4)}")
                result_q.put({
                    "joint_positions": sim_qpos.reshape(1, -1).astype(np.float32),
                    "gripper_actions": np.array([0.0], dtype=np.float32),
                })
                continue

            # --- 达到最大步数 ---
            if step >= args_cli.horizon:
                sim_qpos = obs_dict["policy"]["qpos"].squeeze().cpu().numpy()
                result_q.put({
                    "joint_positions": sim_qpos.reshape(1, -1).astype(np.float32),
                    "gripper_actions": np.array([0.0], dtype=np.float32),
                    "done": True,
                })
                continue

            # --- 1. 处理真机图像 ---
            front_img = process_real_image(obs["front_image"], 256)
            wrist_img = process_real_image(obs["wrist_image"], 256)
            obs_img = torch.stack([front_img, wrist_img], dim=0).to(device)

            # --- 2. 仿真 qpos/qvel ---
            sim_obs = obs_dict["policy"]
            qpos = sim_obs["qpos"].squeeze().cpu().numpy()
            qvel = sim_obs["qvel"].squeeze().cpu().numpy()

            if norm_params is not None:
                qpos_n = minmax_normalize(qpos, norm_params["qpos"]["min"], norm_params["qpos"]["max"])
                qvel_n = minmax_normalize(qvel, norm_params["qvel"]["min"], norm_params["qvel"]["max"])
            else:
                qpos_n, qvel_n = qpos, qvel
            obs_low = torch.from_numpy(np.concatenate([qpos_n, qvel_n])).float().to(device)

            # 更新缓冲区
            obs_img_buffer.append(obs_img)
            obs_low_buffer.append(obs_low)
            while len(obs_img_buffer) < obs_horizon:
                obs_img_buffer.appendleft(obs_img_buffer[0])
                obs_low_buffer.appendleft(obs_low_buffer[0])

            # --- 3. 如果 action buffer 空了, 重新推理 ---
            if len(action_buffer) == 0:
                obs_images = torch.stack(list(obs_img_buffer), dim=0).unsqueeze(0)
                obs_lowdim = torch.stack(list(obs_low_buffer), dim=0).unsqueeze(0)
                action_chunk = ddpm_sample(model, scheduler, obs_images, obs_lowdim, device)
                # 取前 action_horizon 步放入 buffer
                for j in range(min(action_horizon, action_chunk.shape[0])):
                    action_buffer.append(action_chunk[j].copy())
                print(f"[INFO] 新推理, 填充 {len(action_buffer)} 步到 action buffer")

            # --- 4. 从 buffer 取 1 步, 仿真执行 1 步, 返回 1 组关节角度 ---
            raw_action = action_buffer.popleft()
            print(f"[Step {step:04d}] raw_action (model output): {np.round(raw_action, 4)}")

            action = raw_action.copy()

            # 反归一化 ee_delta
            if norm_params is not None and "action" in norm_params:
                action[:-1] = minmax_denormalize(
                    action[:-1],
                    norm_params["action"]["min"][:-1],
                    norm_params["action"]["max"][:-1],
                )

            # Gripper 二值化
            gripper = 1.0 if action[-1] > 0.0 else -1.0
            action[-1] = gripper

            # 仿真 step 1 步 (在主线程!)
            action_tensor = torch.from_numpy(action).float().to(device).view(1, -1)
            obs_dict, _, terminated, truncated, _ = env.step(action_tensor)
            step += 1

            # 读取仿真关节角度
            sim_qpos = obs_dict["policy"]["qpos"].squeeze().cpu().numpy()
            print(f"         -> sim_qpos: {np.round(sim_qpos, 4)}, gripper: {gripper}")

            if terminated or truncated:
                print(f"[INFO] Episode ended at step {step}")

            # 返回 1 步关节角度给真机
            result_q.put({
                "joint_positions": sim_qpos.reshape(1, -1).astype(np.float32),
                "gripper_actions": np.array([-gripper], dtype=np.float32),  # 仿真→真机 gripper 取反
            })

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt, 停止...")
    finally:
        stop_event.set()
        env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
    simulation_app.close()
