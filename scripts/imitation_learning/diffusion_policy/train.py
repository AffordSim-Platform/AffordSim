"""
Diffusion Policy 训练脚本.

DDPM (Denoising Diffusion Probabilistic Models) 训练流程:
1. 从数据集采样 (obs, action_chunk)
2. 对 action_chunk 加随机噪声 (forward diffusion)
3. 模型预测噪声 (conditioned on obs + timestep)
4. 用 MSE loss 优化噪声预测

用法:
    python scripts/imitation_learning/diffusion_policy/train.py \
        --dataset /home/lmy/code/Real2Sim2Real-Pipeline-auto_task/dataset/r2s2r_data/task_gen/pick_place/pick_banana_and_place_on_plate/hdf5/merged_trajectories_robomimic.hdf5 \
        --stat /home/lmy/code/Real2Sim2Real-Pipeline-auto_task/dataset/r2s2r_data/task_gen/pick_place/pick_banana_and_place_on_plate/hdf5/dataset_stat.json \
        --epochs 1000 \
        --batch_size 64 \
        --pred_horizon 16 \
        --obs_horizon 2
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import DiffusionPolicyDataset
from network import DiffusionPolicy


# ============================================================================
# DDPM Noise Schedule
# ============================================================================

class DDPMScheduler:
    """DDPM 噪声调度器."""

    def __init__(self, num_diffusion_steps: int = 100, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.num_steps = num_diffusion_steps

        # Linear beta schedule
        betas = torch.linspace(beta_start, beta_end, num_diffusion_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # 用于采样
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    def add_noise(self, x0, noise, timestep):
        """Forward diffusion: q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise.

        Args:
            x0: (B, T_a, D) 原始动作序列
            noise: (B, T_a, D) 高斯噪声
            timestep: (B,) diffusion timestep

        Returns:
            x_t: (B, T_a, D) 加噪后的动作序列
        """
        sqrt_alpha = self.sqrt_alphas_cumprod[timestep.cpu()].view(-1, 1, 1).to(x0.device)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timestep.cpu()].view(-1, 1, 1).to(x0.device)
        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise

    def step(self, noise_pred, timestep, x_t):
        """Reverse diffusion: 从 x_t 采样 x_{t-1}.

        Args:
            noise_pred: (B, T_a, D) 模型预测的噪声
            timestep: int, 当前 timestep
            x_t: (B, T_a, D) 当前噪声样本

        Returns:
            x_{t-1}: (B, T_a, D)
        """
        alpha = self.alphas[timestep]
        alpha_cumprod = self.alphas_cumprod[timestep]
        beta = self.betas[timestep]

        # 预测均值
        pred_mean = (1 / math.sqrt(alpha)) * (
            x_t - (beta / math.sqrt(1 - alpha_cumprod)) * noise_pred
        )

        if timestep > 0:
            noise = torch.randn_like(x_t)
            variance = self.posterior_variance[timestep]
            return pred_mean + math.sqrt(variance) * noise
        else:
            return pred_mean


# ============================================================================
# 训练
# ============================================================================

def run_eval_subprocess(args, ckpt_path: str, log_dir: str, epoch: int) -> dict | None:
    """每次保存 ckpt 后同步拉起 eval_rollout.py 子进程, 解析 json 结果.

    返回 dict (内含 stage_success_rate / overall_success_rate), 或 None 表示失败.
    """
    if not args.eval_task_config:
        return None

    eval_dir = os.path.join(log_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    output_json = os.path.join(eval_dir, f"eval_epoch_{epoch}.json")

    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_rollout.py")
    cmd = [
        sys.executable, script_path,
        "--checkpoint", ckpt_path,
        "--task_config", args.eval_task_config,
        "--task", args.eval_task,
        "--num_rollouts", str(args.eval_num_rollouts),
        "--horizon", str(args.eval_horizon),
        "--seed", str(args.eval_seed),
        "--output_json", output_json,
        "--pick_lift_threshold", str(args.eval_pick_lift_threshold),
        "--headless",
        "--enable_cameras",
    ]
    if args.stat:
        cmd += ["--stat", args.stat]
    if args.eval_env_cfg_class:
        cmd += ["--env_cfg_class", args.eval_env_cfg_class]

    print(f"[EVAL] launching subprocess: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    print(f"[EVAL] subprocess returncode={result.returncode}")
    if result.returncode != 0:
        print(f"[EVAL] subprocess failed; skipping this epoch's eval")
        return None

    if not os.path.exists(output_json):
        print(f"[EVAL] expected json not found: {output_json}")
        return None
    with open(output_json) as f:
        return json.load(f)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用设备: {device}")

    # --- 数据集 ---
    train_dataset = DiffusionPolicyDataset(
        hdf5_path=args.dataset,
        pred_horizon=args.pred_horizon,
        obs_horizon=args.obs_horizon,
        filter_key="train",
    )
    valid_dataset = DiffusionPolicyDataset(
        hdf5_path=args.dataset,
        pred_horizon=args.pred_horizon,
        obs_horizon=args.obs_horizon,
        filter_key="valid",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"[INFO] 训练样本: {len(train_dataset)}, 验证样本: {len(valid_dataset)}")

    # --- 模型 ---
    model = DiffusionPolicy(
        action_dim=args.action_dim,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        lowdim_dim=args.lowdim_dim,
        img_feature_dim=args.img_feature_dim,
        lowdim_feature_dim=args.lowdim_feature_dim,
        down_dims=tuple(args.down_dims),
        pretrained_resnet=args.pretrained,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] 模型参数量: {num_params / 1e6:.2f}M")

    # --- DDPM 调度器 ---
    scheduler = DDPMScheduler(
        num_diffusion_steps=args.num_diffusion_steps,
    )

    # --- 优化器 ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Cosine annealing scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    # --- 日志 ---
    timestamp = time.strftime("%Y%m%d%H%M%S")
    task_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", args.eval_task) if args.eval_task else "no_task"
    log_dir = os.path.join(args.output_dir, f"diffusion_policy_{task_tag}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "models"), exist_ok=True)
    writer = SummaryWriter(log_dir)

    # 保存配置
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"[INFO] 日志目录: {log_dir}")

    # --- 加载 checkpoint ---
    start_epoch = 1
    if args.resume is not None:
        print(f"[INFO] 加载 checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        print(f"[INFO] 从 epoch {start_epoch} 继续训练")

    # --- 训练循环 ---
    best_valid_loss = float("inf")
    global_step = 0

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            obs_images = batch["obs_images"].to(device)   # (B, T_o, 2, 3, H, W)
            obs_lowdim = batch["obs_lowdim"].to(device)   # (B, T_o, 14)
            actions = batch["actions"].to(device)          # (B, T_a, 7)

            B = actions.shape[0]

            # 采样随机 timestep
            timesteps = torch.randint(0, args.num_diffusion_steps, (B,), device=device).long()

            # 采样噪声
            noise = torch.randn_like(actions)

            # Forward diffusion: 给动作加噪
            noisy_actions = scheduler.add_noise(actions, noise, timesteps)

            # 预测噪声
            noise_pred = model(noisy_actions, timesteps, obs_images, obs_lowdim)

            # MSE loss
            loss = nn.MSELoss()(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            if global_step % 100 == 0:
                writer.add_scalar("Train/loss", loss.item(), global_step)
                writer.add_scalar("Train/lr", optimizer.param_groups[0]["lr"], global_step)

        lr_scheduler.step()

        avg_train_loss = epoch_loss / max(num_batches, 1)

        # --- 验证 ---
        model.eval()
        valid_loss = 0.0
        valid_batches = 0
        with torch.no_grad():
            for batch in valid_loader:
                obs_images = batch["obs_images"].to(device)
                obs_lowdim = batch["obs_lowdim"].to(device)
                actions = batch["actions"].to(device)
                B = actions.shape[0]

                timesteps = torch.randint(0, args.num_diffusion_steps, (B,), device=device).long()
                noise = torch.randn_like(actions)
                noisy_actions = scheduler.add_noise(actions, noise, timesteps)
                noise_pred = model(noisy_actions, timesteps, obs_images, obs_lowdim)
                loss = nn.MSELoss()(noise_pred, noise)

                valid_loss += loss.item()
                valid_batches += 1

        avg_valid_loss = valid_loss / max(valid_batches, 1)

        # 日志
        writer.add_scalar("Epoch/train_loss", avg_train_loss, epoch)
        writer.add_scalar("Epoch/valid_loss", avg_valid_loss, epoch)
        writer.add_scalar("Epoch/lr", optimizer.param_groups[0]["lr"], epoch)

        print(f"Epoch {epoch:4d} | Train Loss: {avg_train_loss:.6f} | Valid Loss: {avg_valid_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # --- 保存 checkpoint ---
        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(log_dir, "models", f"model_epoch_{epoch}.pth")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "args": vars(args),
            }, ckpt_path)
            print(f"  -> 保存 checkpoint: {ckpt_path}")

            # --- 训练期评估 (subprocess.run 同步阻塞) ---
            if args.eval_task_config:
                eval_result = run_eval_subprocess(args, ckpt_path, log_dir, epoch)
                if eval_result is not None:
                    overall = float(eval_result.get("overall_success_rate", 0.0))
                    writer.add_scalar("Eval/overall_success_rate", overall, epoch)
                    for stage_key, rate in (eval_result.get("stage_success_rate") or {}).items():
                        writer.add_scalar(f"Eval/stage/{stage_key}", float(rate), epoch)
                    print(
                        f"  -> Eval overall={overall:.2%}, "
                        + ", ".join(
                            f"{k}={float(v):.2%}"
                            for k, v in (eval_result.get("stage_success_rate") or {}).items()
                        )
                    )

                    # 将 eval GIF 写入 TensorBoard (每个 trial 一个视频)
                    for gi, gif_path in enumerate(eval_result.get("gif_paths") or []):
                        if not os.path.exists(gif_path):
                            continue
                        try:
                            gif = Image.open(gif_path)
                            frames = []
                            for frame_idx in range(gif.n_frames):
                                gif.seek(frame_idx)
                                frames.append(np.array(gif.convert("RGB")))
                            # TensorBoard add_video 期望 (N, T, C, H, W), uint8
                            vid = np.stack(frames)  # (T, H, W, 3)
                            vid = vid.transpose(0, 3, 1, 2)  # (T, 3, H, W)
                            vid_tensor = torch.from_numpy(vid).unsqueeze(0)  # (1, T, 3, H, W)
                            writer.add_video(f"Eval/rollout_trial_{gi}", vid_tensor, epoch, fps=6)
                        except Exception as e:
                            print(f"  -> [WARN] failed to log GIF to tensorboard: {e}")

        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            ckpt_path = os.path.join(log_dir, "models", "model_best.pth")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "args": vars(args),
            }, ckpt_path)
            print(f"  -> 最佳模型 (valid loss: {best_valid_loss:.6f})")

    writer.close()
    train_dataset.close()
    valid_dataset.close()
    print("[INFO] 训练完成!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion Policy Training")

    # 数据
    parser.add_argument("--dataset", type=str, required=True, help="HDF5 数据集路径")
    parser.add_argument("--stat", type=str, default=None, help="dataset_stat.json 路径")
    parser.add_argument("--output_dir", type=str, default="logs/diffusion_policy", help="输出目录")

    # 训练
    parser.add_argument("--epochs", type=int, default=300, help="训练轮次")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="权重衰减")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint 路径")
    parser.add_argument("--save_every", type=int, default=1, help="每 N 个 epoch 保存一次")

    # Diffusion
    parser.add_argument("--num_diffusion_steps", type=int, default=100, help="Diffusion steps")
    parser.add_argument("--pred_horizon", type=int, default=16, help="预测动作序列长度 T_a")
    parser.add_argument("--obs_horizon", type=int, default=2, help="观测历史长度 T_o")
    parser.add_argument("--action_horizon", type=int, default=8, help="实际执行的动作步数 T_exec")

    # 模型
    parser.add_argument("--action_dim", type=int, default=7, help="动作维度")
    parser.add_argument("--lowdim_dim", type=int, default=14, help="低维观测维度 (qpos+qvel)")
    parser.add_argument("--img_feature_dim", type=int, default=256, help="图像特征维度")
    parser.add_argument("--lowdim_feature_dim", type=int, default=64, help="低维特征维度")
    parser.add_argument("--down_dims", type=int, nargs="+", default=[256, 512, 1024], help="U-Net 通道")
    parser.add_argument("--pretrained", action="store_true", default=True, help="使用预训练 ResNet18")

    # 训练期评估 (每次 save ckpt 后同步拉起 eval_rollout.py 子进程)
    parser.add_argument("--eval_task_config", type=str, default=None,
                        help="task yaml 路径; 设置后才会启用训练期评估")
    parser.add_argument("--eval_task", type=str, default="Pick_Place_banana-v0",
                        help="gym 环境 id, 用于评估子进程")
    parser.add_argument("--eval_num_rollouts", type=int, default=5)
    parser.add_argument("--eval_horizon", type=int, default=300)
    parser.add_argument("--eval_seed", type=int, default=0)
    parser.add_argument("--eval_pick_lift_threshold", type=float, default=0.05)
    parser.add_argument("--eval_env_cfg_class", type=str, default=None,
                        help="可选: 'pkg.mod:ClassName' 形式覆盖 eval 子进程的 env_cfg 类")

    args = parser.parse_args()
    train(args)
