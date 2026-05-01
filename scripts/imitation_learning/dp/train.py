"""
Diffusion Policy 训练 (RSS 原版风格).

要点:
- EMA 权重 (每个 opt step 更新), ckpt 里同时保存 main + ema, eval/inference 用 ema
- 余弦 β schedule + clip_sample (由 diffusers.DDPMScheduler 提供)
- 训练期评估: 每次 save ckpt 后同步拉起 eval_rollout.py 子进程

用法:
    python scripts/imitation_learning/dp_rss/train.py \
        --dataset /path/to/merged_trajectories_robo.hdf5 \
        --stat    /path/to/dataset_stat.json \
        --epochs 1000 --batch_size 64 \
        --pred_horizon 16 --obs_horizon 2 --action_horizon 8 \
        --eval_task_config /path/to/generated_task_config.yaml \
        --eval_task Pick_Place_banana-v0
"""

import argparse
import copy
import json
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

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dataset import DiffusionPolicyDataset
from ema_model import EMAModel
from policy import DiffusionPolicy


# ===========================================================================
# 训练期评估 (subprocess)
# ===========================================================================

def run_eval_subprocess(args, ckpt_path: str, log_dir: str, epoch: int):
    if not args.eval_task_config:
        return None

    eval_dir = os.path.join(log_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    output_json = os.path.join(eval_dir, f"eval_epoch_{epoch}.json")

    script_path = os.path.join(_HERE, "eval_rollout.py")
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
    if args.eval_warmup_action:
        cmd += ["--warmup_action", args.eval_warmup_action]
    cmd += ["--joint_delta_mode", args.eval_joint_delta_mode]
    if args.eval_action_horizon is not None:
        cmd += ["--action_horizon", str(args.eval_action_horizon)]

    print(f"[EVAL] launching: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    print(f"[EVAL] subprocess returncode={result.returncode}")
    if result.returncode != 0 or not os.path.exists(output_json):
        return None
    with open(output_json) as f:
        return json.load(f)


# ===========================================================================
# Checkpoint helpers
# ===========================================================================

def save_ckpt(path, policy, ema, optimizer, lr_scheduler, epoch, args):
    torch.save(
        {
            "model": policy.state_dict(),
            "ema_model": ema.averaged_model.state_dict() if ema is not None else None,
            "ema_step": ema.optimization_step if ema is not None else 0,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
            "epoch": epoch,
            "args": vars(args),
        },
        path,
    )


# ===========================================================================
# Training
# ===========================================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    # --- dataset ---
    train_ds = DiffusionPolicyDataset(
        hdf5_path=args.dataset,
        pred_horizon=args.pred_horizon,
        obs_horizon=args.obs_horizon,
        filter_key="train",
        load_count=args.load_count,
        use_qvel=args.use_qvel,
    )
    valid_ds = DiffusionPolicyDataset(
        hdf5_path=args.dataset,
        pred_horizon=args.pred_horizon,
        obs_horizon=args.obs_horizon,
        filter_key="valid",
        load_count=args.load_count,
        use_qvel=args.use_qvel,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    print(f"[INFO] train={len(train_ds)}, valid={len(valid_ds)}")

    # 把 dataset 解析出的真实 use_qvel 回填到 args (保证 ckpt 里存的是实际值, play 时一致).
    args.use_qvel = train_ds.use_qvel
    # lowdim_dim 必须随 use_qvel 调整 (qpos=7, qvel=7).
    expected_lowdim = 14 if args.use_qvel else 7
    if args.lowdim_dim != expected_lowdim:
        print(f"[INFO] lowdim_dim 自动调整: {args.lowdim_dim} -> {expected_lowdim} (use_qvel={args.use_qvel})")
        args.lowdim_dim = expected_lowdim

    # --- policy ---
    policy = DiffusionPolicy(
        action_dim=args.action_dim,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        lowdim_dim=args.lowdim_dim,
        img_feature_dim=args.img_feature_dim,
        lowdim_feature_dim=args.lowdim_feature_dim,
        resize_shape=tuple(args.resize_shape),
        crop_shape=tuple(args.crop_shape),
        num_views=args.num_views,
        pretrained_resnet=args.pretrained,
        use_group_norm=True,
        image_aug=args.image_aug,
        down_dims=tuple(args.down_dims),
        kernel_size=args.kernel_size,
        diffusion_step_embed_dim=args.diffusion_step_embed_dim,
        cond_predict_scale=True,
        num_train_timesteps=args.num_diffusion_steps,
        beta_schedule=args.beta_schedule,
        prediction_type="epsilon",
        clip_sample=True,
        num_inference_steps=args.num_inference_steps,
        inference_scheduler=args.inference_scheduler,
    ).to(device)

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"[INFO] params: {n_params / 1e6:.2f}M")

    # --- EMA ---
    ema = None
    if args.use_ema:
        ema_policy = copy.deepcopy(policy).to(device)
        ema = EMAModel(
            ema_policy,
            update_after_step=args.ema_update_after_step,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            min_value=args.ema_min,
            max_value=args.ema_max,
        )
        print("[INFO] EMA enabled")

    # --- optim ---
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.lr,
        betas=(0.95, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * max(1, len(train_loader))
    warmup_steps = min(args.warmup_steps, total_steps - 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- logging ---
    task_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", args.eval_task) if args.eval_task else "no_task"
    log_dir = os.path.join(args.output_dir, f"dp_rss_{task_tag}_{time.strftime('%Y%m%d%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "models"), exist_ok=True)
    writer = SummaryWriter(log_dir)
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"[INFO] log_dir: {log_dir}")

    # --- resume ---
    start_epoch = 1
    if args.resume is not None:
        print(f"[INFO] resume from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt["model"])
        if ema is not None and ckpt.get("ema_model") is not None:
            ema.averaged_model.load_state_dict(ckpt["ema_model"])
            ema.optimization_step = int(ckpt.get("ema_step", 0))
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("lr_scheduler") is not None:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1

    # --- training loop ---
    best_valid_loss = float("inf")
    global_step = 0

    for epoch in range(start_epoch, args.epochs + 1):
        policy.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            loss = policy.compute_loss(batch)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()

            if ema is not None:
                ema.step(policy)

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if global_step % 100 == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                if ema is not None:
                    writer.add_scalar("train/ema_decay", ema.decay, global_step)

        avg_train = epoch_loss / max(n_batches, 1)

        # --- valid (use EMA if present) ---
        eval_policy = ema.averaged_model if ema is not None else policy
        eval_policy.eval()
        v_loss, v_n = 0.0, 0
        with torch.no_grad():
            for batch in valid_loader:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                v_loss += eval_policy.compute_loss(batch).item()
                v_n += 1
        avg_valid = v_loss / max(v_n, 1)

        writer.add_scalar("epoch/train_loss", avg_train, epoch)
        writer.add_scalar("epoch/valid_loss", avg_valid, epoch)
        writer.add_scalar("epoch/lr", optimizer.param_groups[0]["lr"], epoch)
        print(
            f"Epoch {epoch:4d} | train {avg_train:.6f} | valid {avg_valid:.6f} "
            f"| lr {optimizer.param_groups[0]['lr']:.2e}"
        )

        # --- save + eval rollout ---
        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(log_dir, "models", f"model_epoch_{epoch}.pth")
            save_ckpt(ckpt_path, policy, ema, optimizer, lr_scheduler, epoch, args)
            print(f"  -> saved {ckpt_path}")

            if args.eval_task_config:
                res = run_eval_subprocess(args, ckpt_path, log_dir, epoch)
                if res is not None:
                    overall = float(res.get("overall_success_rate", 0.0))
                    writer.add_scalar("eval/overall_success_rate", overall, epoch)
                    for sk, rate in (res.get("stage_success_rate") or {}).items():
                        writer.add_scalar(f"eval/stage/{sk}", float(rate), epoch)
                    print(
                        f"  -> Eval overall={overall:.2%}, "
                        + ", ".join(
                            f"{k}={float(v):.2%}"
                            for k, v in (res.get("stage_success_rate") or {}).items()
                        )
                    )
                    for gi, gif in enumerate(res.get("gif_paths") or []):
                        if not os.path.exists(gif):
                            continue
                        try:
                            im = Image.open(gif)
                            frames = []
                            for fi in range(im.n_frames):
                                im.seek(fi)
                                frames.append(np.array(im.convert("RGB")))
                            vid = np.stack(frames).transpose(0, 3, 1, 2)[None]
                            writer.add_video(
                                f"eval/rollout_trial_{gi}",
                                torch.from_numpy(vid), epoch, fps=6,
                            )
                        except Exception as e:
                            print(f"  -> [WARN] gif->tb failed: {e}")

        if avg_valid < best_valid_loss:
            best_valid_loss = avg_valid
            ckpt_path = os.path.join(log_dir, "models", "model_best.pth")
            save_ckpt(ckpt_path, policy, ema, optimizer, lr_scheduler, epoch, args)
            print(f"  -> best (valid {best_valid_loss:.6f})")

    writer.close()
    train_ds.close()
    valid_ds.close()
    print("[INFO] training done.")


# ===========================================================================
# CLI
# ===========================================================================

def build_parser():
    p = argparse.ArgumentParser(description="Diffusion Policy (RSS) Training")

    # data
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--stat", type=str, default=None, help="dataset_stat.json (仅传给 eval 子进程)")
    p.add_argument("--output_dir", type=str, default="logs/dp_rss")
    p.add_argument("--load_count", type=int, default=None)

    # qvel 是否进入 obs_lowdim. 默认 None=按 hdf5 自动探测;
    # --no_qvel 强制只用 qpos (lowdim_dim=7); --use_qvel 强制要 qvel (缺则报错).
    qvel_grp = p.add_mutually_exclusive_group()
    qvel_grp.add_argument("--no_qvel", dest="use_qvel", action="store_false", default=None,
                          help="不使用 qvel, obs_lowdim 只含 qpos (lowdim_dim 自动设 7)")
    qvel_grp.add_argument("--use_qvel", dest="use_qvel", action="store_true", default=None,
                          help="强制使用 qvel; hdf5 没 qvel 时报错")

    # training
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--save_every", type=int, default=50)

    # EMA
    p.add_argument("--use_ema", action="store_true", default=True)
    p.add_argument("--no_ema", dest="use_ema", action="store_false")
    p.add_argument("--ema_update_after_step", type=int, default=0)
    p.add_argument("--ema_inv_gamma", type=float, default=1.0)
    p.add_argument("--ema_power", type=float, default=0.75)
    p.add_argument("--ema_min", type=float, default=0.0)
    p.add_argument("--ema_max", type=float, default=0.9999)

    # horizons
    p.add_argument("--pred_horizon", type=int, default=16)
    p.add_argument("--obs_horizon", type=int, default=2)
    p.add_argument("--action_horizon", type=int, default=8)

    # obs encoder
    p.add_argument("--lowdim_dim", type=int, default=14)
    p.add_argument("--img_feature_dim", type=int, default=128)
    p.add_argument("--lowdim_feature_dim", type=int, default=64)
    p.add_argument("--resize_shape", type=int, nargs=2, default=[240, 240])
    p.add_argument("--crop_shape", type=int, nargs=2, default=[224, 224])
    p.add_argument("--num_views", type=int, default=2)
    p.add_argument("--pretrained", action="store_true", default=True)
    # 图像增强 (训练随机 crop, 推理 center crop). 默认开;
    # --no_image_aug 关掉 (resize 直接到 crop_shape, train/eval 行为一致, 风险: 易过拟合).
    p.add_argument("--image_aug", action="store_true", default=True,
                   help="启用 RandomCrop 图像增强 (默认开)")
    p.add_argument("--no_image_aug", dest="image_aug", action="store_false",
                   help="关闭图像增强, 直接 resize 到 crop_shape, train/eval 完全一致")

    # U-Net
    p.add_argument("--down_dims", type=int, nargs="+", default=[256, 512, 1024])
    p.add_argument("--kernel_size", type=int, default=5)
    p.add_argument("--diffusion_step_embed_dim", type=int, default=256)

    # diffusion
    p.add_argument("--num_diffusion_steps", type=int, default=100)
    p.add_argument("--beta_schedule", type=str, default="squaredcos_cap_v2",
                   choices=["linear", "squaredcos_cap_v2"])
    p.add_argument("--inference_scheduler", type=str, default="ddim", choices=["ddpm", "ddim"])
    p.add_argument("--num_inference_steps", type=int, default=16)

    # action semantics
    p.add_argument("--action_dim", type=int, default=None)
    p.add_argument("--action_type", type=str, default="ee_delta", choices=["ee_delta", "joint_delta"])

    # eval (subprocess)
    p.add_argument("--eval_task_config", type=str, default=None)
    p.add_argument("--eval_task", type=str, default="Pick_Place_banana-v0")
    p.add_argument("--eval_num_rollouts", type=int, default=5)
    p.add_argument("--eval_horizon", type=int, default=300)
    p.add_argument("--eval_seed", type=int, default=0)
    p.add_argument("--eval_pick_lift_threshold", type=float, default=0.05)
    p.add_argument(
        "--eval_action_horizon", type=int, default=None,
        help="评测时的开环执行步数; 不传则用模型训练时的 action_horizon. 调小可减少累积误差.",
    )
    p.add_argument("--eval_env_cfg_class", type=str, default=None)
    p.add_argument("--eval_warmup_action", type=str, default=None, choices=["zero", "hold"])
    p.add_argument(
        "--eval_joint_delta_mode", type=str, default="cumulative", choices=["current", "cumulative"],
        help=(
            "joint_delta -> 绝对位置的展开方式 (传给 eval 子进程, 仅 action_type=joint_delta 时生效). "
            "'current'=arm_target(k)=cur_qpos(k)+delta(k); "
            "'cumulative'=arm_target(k)=arm_target(k-1)+delta(k), 初值=首帧 arm qpos."
        ),
    )

    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    # 自动推断 action_dim 和 eval_warmup_action (与现有 diffusion_policy/train.py 语义一致)
    _DIM_BY_TYPE = {"ee_delta": 7, "joint_delta": 8}
    _WARMUP_BY_TYPE = {"ee_delta": "zero", "joint_delta": "hold"}
    expected_dim = _DIM_BY_TYPE[args.action_type]
    if args.action_dim is None:
        args.action_dim = expected_dim
        print(f"[INFO] --action_dim 推断为 {expected_dim} (action_type={args.action_type})")
    elif args.action_dim != expected_dim:
        print(
            f"[WARN] --action_dim={args.action_dim} != 推断值 {expected_dim}, "
            f"沿用显式值 (请自行确保数据 action 维度一致)."
        )
    if args.eval_warmup_action is None:
        args.eval_warmup_action = _WARMUP_BY_TYPE[args.action_type]
        print(f"[INFO] --eval_warmup_action 推断为 '{args.eval_warmup_action}'")

    train(args)
