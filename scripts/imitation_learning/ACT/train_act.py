"""
ACT (Action Chunking with Transformers) training script.

CVAE + Transformer training pipeline:
1. Sample (obs, action_chunk) from the dataset
2. CVAE Encoder: encode (obs, action) -> latent z
3. Transformer Decoder: predict action chunk conditioned on obs + z
4. Loss = L1 reconstruction + beta * KL divergence

Data formats (two supported):
  robomimic format: data/demo_X/actions, data/demo_X/obs/qpos, .../front, ...
  raw format: traj_X/action, traj_X/observations/qpos, .../images/front, ...

Usage:
    python scripts/imitation_learning/ACT/train_act.py \
        --dataset /path/to/merged_trajectories_robomimic.hdf5 \
        --stat /path/to/dataset_stat.json \
        --total_iters 200000 \
        --batch_size 64 \
        --action_chunk_size 16 \
        --action_horizon 8
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from network_act import ACTPolicy


# ============================================================================
# Normalization Utilities
# ============================================================================

def compute_minmax(all_data):
    return all_data.min(axis=0), all_data.max(axis=0)


def minmax_normalize(data, dmin, dmax):
    """Min-max normalize to [-1, 1]."""
    drange = dmax - dmin
    drange = np.where(drange < 1e-8, 1.0, drange)
    return 2.0 * (data - dmin) / drange - 1.0


def minmax_denormalize(data, dmin, dmax):
    """Denormalize from [-1, 1] back to original scale."""
    drange = dmax - dmin
    return (data + 1.0) / 2.0 * drange + dmin


# ============================================================================
# Gripper Conversion: Binary -> Continuous
# ============================================================================

def convert_binary_gripper_to_continuous(gripper_seq):
    """
    Convert binary gripper signal to continuous values and normalize.

    Input: gripper_seq (N,) -- values are -1 (open) or +1 (close)
    Output: (N,) -- values in [0, 1], 0=close, 1=open
    """
    N = len(gripper_seq)
    is_open = gripper_seq < 0
    physical = np.where(is_open, 0.04, 0.02).astype(np.float64)

    for i in range(1, N):
        if gripper_seq[i] != gripper_seq[i - 1]:
            if gripper_seq[i] > 0:
                for j in range(5):
                    idx = i + j
                    if idx < N and gripper_seq[idx] > 0:
                        t = (j + 1) / 5.0
                        physical[idx] = 0.04 * (1 - t) + 0.02 * t
            else:
                for j in range(5):
                    idx = i + j
                    if idx < N and gripper_seq[idx] < 0:
                        t = (j + 1) / 5.0
                        physical[idx] = 0.02 * (1 - t) + 0.04 * t

    normalized = (physical - 0.02) / 0.02
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


# ============================================================================
# Dataset
# ============================================================================

def get_active_mask(action):
    """Return mask of steps where EE delta is non-zero."""
    return np.abs(action[:, :6]).sum(axis=1) > 1e-8


class ACTDataset(Dataset):
    """
    Load HDF5 data, auto-detecting format:
      - Raw format: traj_X/action, traj_X/observations/qpos, .../images/front, ...
      - robomimic format: data/demo_X/actions, data/demo_X/obs/qpos, .../front, ...

    action_chunk_size: returns action sequence of the next chunk_size steps.
    """

    def __init__(self, dataset_file: str, load_count=None, remove_idle=True,
                 action_chunk_size=1):
        self.dataset_file = dataset_file
        self.action_chunk_size = action_chunk_size
        data = h5py.File(dataset_file, "r")

        top_keys = list(data.keys())
        is_robomimic = "data" in top_keys
        if is_robomimic:
            print(f"[Dataset] Detected robomimic format (data/demo_X/...)")
            self._load_robomimic(data, load_count, remove_idle)
        else:
            print(f"[Dataset] Detected raw format (traj_X/...)")
            self._load_raw(data, load_count, remove_idle)

        data.close()

        # Build valid indices
        valid = []
        for i in range(len(self.states)):
            act_end = i + action_chunk_size
            if act_end > len(self.states):
                continue
            if action_chunk_size > 1 and self.traj_ids[i] != self.traj_ids[act_end - 1]:
                continue
            valid.append(i)
        self.valid_indices = np.array(valid, dtype=np.int64)

        print(f"[Dataset] action_chunk_size={action_chunk_size}, "
              f"valid frames: {len(self.valid_indices)}/{len(self.states)}")
        print(f"  state dim: {self.states.shape[1]}")
        print(f"  action dim: {self.actions.shape[1]} (6 ee + 1 gripper)")
        print(f"  front image shape: {self.front_images.shape[1:]}")
        print(f"  wrist image shape: {self.wrist_images.shape[1:]}")

    def _load_robomimic(self, data, load_count, remove_idle):
        data_grp = data["data"]
        demo_keys = sorted(
            [k for k in data_grp.keys() if k.startswith("demo_")],
            key=lambda x: int(x.split("_")[1]),
        )
        if load_count is not None:
            demo_keys = demo_keys[:load_count]

        print(f"[Dataset] Loading {len(demo_keys)} demos from robomimic HDF5")

        self.front_images = []
        self.wrist_images = []
        self.states = []
        self.actions = []
        self.traj_ids = []

        for traj_idx, demo_key in enumerate(tqdm(demo_keys, desc="Loading demos")):
            demo = data_grp[demo_key]
            action = np.array(demo["actions"])
            qpos = np.array(demo["obs/qpos"])
            qvel = np.array(demo["obs/qvel"])
            front = np.array(demo["obs/front"])
            wrist = np.array(demo["obs/wrist"])

            if remove_idle:
                mask = get_active_mask(action)
                if mask.sum() == 0:
                    continue
                action = action[mask]
                qpos = qpos[mask]
                qvel = qvel[mask]
                front = front[mask]
                wrist = wrist[mask]

            norm_ee = (action[:, :6] * 100.0).astype(np.float32)
            gripper_cont = convert_binary_gripper_to_continuous(action[:, -1])
            full_action = np.concatenate([norm_ee, gripper_cont[:, None]], axis=-1)

            state = np.concatenate([qpos, qvel], axis=-1).astype(np.float32)
            front_norm = front.astype(np.float32) / 255.0
            wrist_norm = wrist.astype(np.float32) / 255.0

            self.front_images.append(front_norm)
            self.wrist_images.append(wrist_norm)
            self.states.append(state)
            self.actions.append(full_action)
            self.traj_ids.append(np.full(len(full_action), traj_idx, dtype=np.int64))

        self._concat_all()

        self.action_ee_min = -np.ones(6, dtype=np.float32)
        self.action_ee_max = np.ones(6, dtype=np.float32)
        self.qpos_min = -np.ones(self.states.shape[1] // 2, dtype=np.float32)
        self.qpos_max = np.ones(self.states.shape[1] // 2, dtype=np.float32)
        self.qvel_min = -np.ones(self.states.shape[1] // 2, dtype=np.float32)
        self.qvel_max = np.ones(self.states.shape[1] // 2, dtype=np.float32)

    def _load_raw(self, data, load_count, remove_idle):
        traj_keys = sorted(
            [k for k in data.keys() if k.startswith("traj_")],
            key=lambda x: int(x.split("_")[1]),
        )
        if load_count is not None:
            traj_keys = traj_keys[:load_count]

        print(f"[Dataset] Loading {len(traj_keys)} trajectories from raw HDF5")

        all_actions, all_qpos, all_qvel = [], [], []
        traj_data = []

        for traj_key in tqdm(traj_keys, desc="Loading trajectories"):
            traj = data[traj_key]
            action = np.array(traj["action"])
            qpos = np.array(traj["observations/qpos"])
            qvel = np.array(traj["observations/qvel"])
            front = np.array(traj["observations/images/front"])
            wrist = np.array(traj["observations/images/wrist"])

            if remove_idle:
                mask = get_active_mask(action)
                if mask.sum() == 0:
                    continue
                action = action[mask]
                qpos = qpos[mask]
                qvel = qvel[mask]
                front = front[mask]
                wrist = wrist[mask]

            all_actions.append(action)
            all_qpos.append(qpos)
            all_qvel.append(qvel)
            traj_data.append((action, qpos, qvel, front, wrist))

        all_actions_cat = np.concatenate(all_actions, axis=0)
        all_qpos_cat = np.concatenate(all_qpos, axis=0)
        all_qvel_cat = np.concatenate(all_qvel, axis=0)

        self.action_ee_min, self.action_ee_max = compute_minmax(all_actions_cat[:, :6])
        self.qpos_min, self.qpos_max = compute_minmax(all_qpos_cat)
        self.qvel_min, self.qvel_max = compute_minmax(all_qvel_cat)

        self.front_images = []
        self.wrist_images = []
        self.states = []
        self.actions = []
        self.traj_ids = []

        for traj_idx, (action, qpos, qvel, front, wrist) in enumerate(traj_data):
            norm_ee = (minmax_normalize(action[:, :6], self.action_ee_min, self.action_ee_max) * 100.0).astype(np.float32)
            gripper_cont = convert_binary_gripper_to_continuous(action[:, -1])
            full_action = np.concatenate([norm_ee, gripper_cont[:, None]], axis=-1)

            norm_qpos = minmax_normalize(qpos, self.qpos_min, self.qpos_max).astype(np.float32)
            norm_qvel = minmax_normalize(qvel, self.qvel_min, self.qvel_max).astype(np.float32)
            state = np.concatenate([norm_qpos, norm_qvel], axis=-1)

            front_norm = front.astype(np.float32) / 255.0
            wrist_norm = wrist.astype(np.float32) / 255.0

            self.front_images.append(front_norm)
            self.wrist_images.append(wrist_norm)
            self.states.append(state)
            self.actions.append(full_action)
            self.traj_ids.append(np.full(len(full_action), traj_idx, dtype=np.int64))

        self._concat_all()

    def _concat_all(self):
        self.front_images = np.concatenate(self.front_images, axis=0)
        self.wrist_images = np.concatenate(self.wrist_images, axis=0)
        self.states = np.concatenate(self.states, axis=0)
        self.actions = np.concatenate(self.actions, axis=0)
        self.traj_ids = np.concatenate(self.traj_ids, axis=0)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        C = self.action_chunk_size
        return {
            "front": torch.from_numpy(self.front_images[i]).float(),
            "wrist": torch.from_numpy(self.wrist_images[i]).float(),
            "state": torch.from_numpy(self.states[i]).float(),
            "action": torch.from_numpy(self.actions[i:i + C]).float(),
        }

    def get_norm_params(self):
        return {
            "action_ee": {"min": self.action_ee_min.tolist(), "max": self.action_ee_max.tolist()},
            "qpos": {"min": self.qpos_min.tolist(), "max": self.qpos_max.tolist()},
            "qvel": {"min": self.qvel_min.tolist(), "max": self.qvel_max.tolist()},
        }


# ============================================================================
# Sampler (infinite iteration)
# ============================================================================

class IterationBasedBatchSampler(BatchSampler):
    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations


# ============================================================================
# Loss
# ============================================================================

def act_loss(pred_actions, target_actions, mu, log_var, kl_weight=10.0):
    """ACT Loss = L1 reconstruction + KL divergence.

    Args:
        pred_actions: (B, C, action_dim) predicted action chunk
        target_actions: (B, C, action_dim) target action chunk
        mu: (B, latent_dim) mean output of CVAE encoder
        log_var: (B, latent_dim) log variance output of CVAE encoder
        kl_weight: KL divergence weight (beta in beta-VAE)

    Returns:
        total_loss, l1_loss, kl_loss
    """
    # L1 reconstruction loss
    l1_loss = F.l1_loss(pred_actions, target_actions)

    # KL divergence: D_KL(q(z|x,a) || p(z)) = -0.5 * sum(1 + log_var - mu^2 - var)
    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    total_loss = l1_loss + kl_weight * kl_loss
    return total_loss, l1_loss, kl_loss


# ============================================================================
# Training Utilities
# ============================================================================

def save_checkpoint(policy, optimizer, iteration, run_dir, tag, norm_params):
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"{tag}.pt")
    torch.save({
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "norm_params": norm_params,
        "model_config": {
            "state_dim": policy.front_encoder.fc.out_features * 0 + \
                         policy.state_proj.in_features,  # state_dim
            "action_dim": policy.action_dim,
            "action_chunk_size": policy.action_chunk_size,
            "hidden_dim": policy.hidden_dim,
            "latent_dim": policy.latent_dim,
        },
    }, path)
    print(f"[Checkpoint] Saved: {path}")
    return path


def run_eval_subprocess(args, ckpt_path: str, run_dir: str, iteration: int) -> dict | None:
    """Synchronously launch eval_rollout_act.py subprocess after each checkpoint save."""
    if not args.eval_task_config or not args.eval_task:
        return None

    eval_dir = os.path.join(run_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    output_json = os.path.join(eval_dir, f"eval_iter_{iteration}.json")

    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_rollout_act.py")
    cmd = [
        sys.executable, script_path,
        "--checkpoint", ckpt_path,
        "--task", args.eval_task,
        "--task_config", args.eval_task_config,
        "--num_rollouts", str(args.eval_num_rollouts),
        "--horizon", str(args.eval_horizon),
        "--seed", str(args.seed),
        "--output_json", output_json,
        "--pick_lift_threshold", str(args.eval_pick_lift_threshold),
        "--headless",
        "--enable_cameras",
    ]
    if args.stat:
        cmd += ["--dataset_stat", args.stat]
    if args.eval_env_cfg_class:
        cmd += ["--env_cfg_class", args.eval_env_cfg_class]
    if args.action_chunk_size and args.action_chunk_size > 1:
        cmd += ["--action_horizon", str(args.action_horizon)]
        cmd += ["--action_chunk_size", str(args.action_chunk_size)]

    print(f"[EVAL] launching subprocess: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    print(f"[EVAL] subprocess returncode={result.returncode}")
    if result.returncode != 0:
        print(f"[EVAL] subprocess failed; skipping this eval")
        return None
    if not os.path.exists(output_json):
        print(f"[EVAL] expected json not found: {output_json}")
        return None
    with open(output_json) as f:
        return json.load(f)


# ============================================================================
# Args
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="ACT Training (CVAE + Transformer)")
    parser.add_argument("--dataset", type=str, required=True, help="Path to HDF5 demo dataset")
    parser.add_argument("--num_demos", type=int, default=None, help="Number of demos to load")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--total_iters", type=int, default=200_000, help="Total training iterations")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--no_remove_idle", action="store_true",
                        help="Disable removing idle (zero EE delta) steps")

    # Action chunking
    parser.add_argument("--action_chunk_size", type=int, default=16,
                        help="Action chunk size: plan m future action steps")
    parser.add_argument("--action_horizon", type=int, default=8,
                        help="Action horizon: actually execute n steps each time (n <= action_chunk_size)")

    # ACT architecture
    parser.add_argument("--hidden_dim", type=int, default=512, help="Transformer hidden dim")
    parser.add_argument("--latent_dim", type=int, default=32, help="CVAE latent dim")
    parser.add_argument("--n_encoder_layers", type=int, default=4, help="CVAE encoder layers")
    parser.add_argument("--n_decoder_layers", type=int, default=7, help="Transformer decoder layers")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--kl_weight", type=float, default=10.0, help="KL divergence weight (beta)")

    # Logging
    parser.add_argument("--log_freq", type=int, default=50, help="Logging frequency")
    parser.add_argument("--save_freq", type=int, default=2000, help="Checkpoint saving frequency")

    # Training-time evaluation
    parser.add_argument("--stat", type=str, default=None,
                        help="dataset_stat.json; used to override norm_params for denormalization")
    parser.add_argument("--eval_task_config", type=str, default=None,
                        help="task yaml path; training-time evaluation is enabled only when set")
    parser.add_argument("--eval_task", type=str, default=None,
                        help="gym environment id, used for evaluation subprocess")
    parser.add_argument("--eval_num_rollouts", type=int, default=5)
    parser.add_argument("--eval_horizon", type=int, default=300)
    parser.add_argument("--eval_pick_lift_threshold", type=float, default=0.05)
    parser.add_argument("--eval_env_cfg_class", type=str, default=None,
                        help="Optional 'pkg.mod:ClassName' to override default pick env cfg")

    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    assert args.action_horizon <= args.action_chunk_size, \
        f"action_horizon ({args.action_horizon}) must <= action_chunk_size ({args.action_chunk_size})"

    # Run name
    if args.exp_name is None:
        eval_task_tag = args.eval_task if args.eval_task else "noeval"
        args.exp_name = (f"act_chunk{args.action_chunk_size}"
                         f"_horizon{args.action_horizon}"
                         f"_h{args.hidden_dim}_z{args.latent_dim}"
                         f"_{os.path.basename(args.dataset).replace('.h5', '').replace('.hdf5', '')}"
                         f"_{eval_task_tag}_{int(time.time())}")
    run_dir = os.path.join("logs", args.exp_name)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    writer = SummaryWriter(run_dir)

    # Dataset
    ds = ACTDataset(
        args.dataset,
        load_count=args.num_demos,
        remove_idle=not args.no_remove_idle,
        action_chunk_size=args.action_chunk_size,
    )
    norm_params = ds.get_norm_params()

    norm_path = os.path.join(run_dir, "norm_params.json")
    with open(norm_path, "w") as f:
        json.dump(norm_params, f, indent=2)
    print(f"[INFO] Norm params saved to {norm_path}")

    # DataLoader
    sampler = RandomSampler(ds)
    batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    iter_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
    data_loader = DataLoader(ds, batch_sampler=iter_sampler, num_workers=4, pin_memory=True)

    # Model
    state_dim = ds.states.shape[1]
    policy = ACTPolicy(
        state_dim=state_dim,
        action_dim=7,
        action_chunk_size=args.action_chunk_size,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    ).to(device)

    n_total = sum(p.numel() for p in policy.parameters())
    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"[Model] ACTPolicy: CVAE + Transformer")
    print(f"[Model] chunk={args.action_chunk_size}, horizon={args.action_horizon}")
    print(f"[Model] hidden_dim={args.hidden_dim}, latent_dim={args.latent_dim}")
    print(f"[Model] encoder_layers={args.n_encoder_layers}, decoder_layers={args.n_decoder_layers}")
    print(f"[Model] total params: {n_total:,}, trainable: {n_trainable:,}")

    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, policy.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_iters, eta_min=1e-6)

    # Training loop
    best_eval_success = -1.0
    policy.train()

    for iteration, batch in enumerate(tqdm(data_loader, desc="Training", total=args.total_iters)):
        front = batch["front"].to(device)
        wrist = batch["wrist"].to(device)
        state = batch["state"].to(device)
        target = batch["action"].to(device)

        # Forward (training mode: returns pred_actions, mu, log_var)
        pred_actions, mu, log_var = policy(front, wrist, state, actions=target)

        # ACT Loss
        total_loss, l1_loss, kl_loss = act_loss(
            pred_actions, target, mu, log_var, kl_weight=args.kl_weight,
        )

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Logging
        if iteration % args.log_freq == 0:
            with torch.no_grad():
                # Per-dimension loss
                ee_l1 = F.l1_loss(pred_actions[:, :, :6], target[:, :, :6])
                gripper_l1 = F.l1_loss(pred_actions[:, :, 6:], target[:, :, 6:])

                # Gripper accuracy
                gripper_pred_binary = (pred_actions[:, :, 6] > 0.5).float()
                gripper_target_binary = (target[:, :, 6] > 0.5).float()
                gripper_acc = (gripper_pred_binary == gripper_target_binary).float().mean().item()

            print(f"[Iter {iteration}] loss={total_loss.item():.4f} "
                  f"l1={l1_loss.item():.4f} kl={kl_loss.item():.4f} "
                  f"ee_l1={ee_l1.item():.4f} gripper_l1={gripper_l1.item():.4f} "
                  f"gripper_acc={gripper_acc:.4f} "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")

            writer.add_scalar("loss/total", total_loss.item(), iteration)
            writer.add_scalar("loss/l1", l1_loss.item(), iteration)
            writer.add_scalar("loss/kl", kl_loss.item(), iteration)
            writer.add_scalar("loss/ee_l1", ee_l1.item(), iteration)
            writer.add_scalar("loss/gripper_l1", gripper_l1.item(), iteration)
            writer.add_scalar("metrics/gripper_accuracy", gripper_acc, iteration)
            writer.add_scalar("lr", scheduler.get_last_lr()[0], iteration)

        # Save checkpoint
        if args.save_freq and iteration > args.save_freq and iteration % args.save_freq == 0:
            ckpt_path = save_checkpoint(policy, optimizer, iteration, run_dir,
                                        f"iter_{iteration}", norm_params)

            eval_result = run_eval_subprocess(args, ckpt_path, run_dir, iteration)
            if eval_result is not None:
                overall = float(eval_result.get("overall_success_rate", 0.0))
                writer.add_scalar("Eval/overall_success_rate", overall, iteration)
                for stage_key, rate in (eval_result.get("stage_success_rate") or {}).items():
                    writer.add_scalar(f"Eval/stage/{stage_key}", float(rate), iteration)
                for gi, gif_path in enumerate(eval_result.get("gif_paths") or []):
                    if not os.path.exists(gif_path):
                        continue
                    try:
                        from PIL import Image
                        gif = Image.open(gif_path)
                        frames = []
                        for frame_idx in range(gif.n_frames):
                            gif.seek(frame_idx)
                            frames.append(np.array(gif.convert("RGB")))
                        vid = np.stack(frames).transpose(0, 3, 1, 2)
                        vid_tensor = torch.from_numpy(vid).unsqueeze(0)
                        writer.add_video(f"Eval/rollout_trial_{gi}", vid_tensor, iteration, fps=6)
                    except Exception as e:
                        print(f"  -> [WARN] failed to log GIF to tensorboard: {e}")
                print(f"  -> Eval overall={overall:.2%}, "
                      + ", ".join(f"{k}={float(v):.2%}"
                                  for k, v in (eval_result.get("stage_success_rate") or {}).items()))
                if overall > best_eval_success:
                    best_eval_success = overall
                    save_checkpoint(policy, optimizer, iteration, run_dir, "best_eval", norm_params)

    # Final checkpoint
    final_ckpt_path = save_checkpoint(policy, optimizer, args.total_iters, run_dir, "final", norm_params)
    eval_result = run_eval_subprocess(args, final_ckpt_path, run_dir, args.total_iters)
    if eval_result is not None:
        overall = float(eval_result.get("overall_success_rate", 0.0))
        writer.add_scalar("Eval/overall_success_rate", overall, args.total_iters)
        for stage_key, rate in (eval_result.get("stage_success_rate") or {}).items():
            writer.add_scalar(f"Eval/stage/{stage_key}", float(rate), args.total_iters)
        for gi, gif_path in enumerate(eval_result.get("gif_paths") or []):
            if not os.path.exists(gif_path):
                continue
            try:
                from PIL import Image
                gif = Image.open(gif_path)
                frames = []
                for frame_idx in range(gif.n_frames):
                    gif.seek(frame_idx)
                    frames.append(np.array(gif.convert("RGB")))
                vid = np.stack(frames).transpose(0, 3, 1, 2)
                vid_tensor = torch.from_numpy(vid).unsqueeze(0)
                writer.add_video(f"Eval/rollout_trial_{gi}", vid_tensor, args.total_iters, fps=6)
            except Exception as e:
                print(f"  -> [WARN] failed to log GIF to tensorboard: {e}")

    writer.close()
    print(f"[INFO] Training complete. Results saved to {run_dir}")


if __name__ == "__main__":
    main()
