"""
Behavior Cloning (BC)  - Gaussian NLL Loss.

:
  -  ResNet18 encoder (ImageNet pretrained)
  -  +  (qpos + qvel)
  - MLP  action  μ,  log_std σ
  - Loss: Gaussian Negative Log-Likelihood

:
  traj_X/action                      (N, 7)   [ee_delta(6) + gripper(1)]
  traj_X/observations/qpos           (N, 7)
  traj_X/observations/qvel           (N, 7)
  traj_X/observations/images/front   (N, 256, 256, 3)
  traj_X/observations/images/wrist   (N, 256, 256, 3)

:
  - :  (-1=open, +1=close)
  - : open=0.04, close=0.02,  5 
  -  [0, 1]: 0=close, 1=open

:
  python train_bc.py --dataset /path/to/data.h5 --task FrankaFr3-xxx-v0 \\
      --action_chunk_size 8 --action_horizon 4
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


# ============================================================================
# Normalization Utilities
# ============================================================================

def compute_minmax(all_data):
    return all_data.min(axis=0), all_data.max(axis=0)


def minmax_normalize(data, dmin, dmax):
    """min-max  [-1, 1]."""
    drange = dmax - dmin
    drange = np.where(drange < 1e-8, 1.0, drange)
    return 2.0 * (data - dmin) / drange - 1.0


def minmax_denormalize(data, dmin, dmax):
    """ [-1, 1] ."""
    drange = dmax - dmin
    return (data + 1.0) / 2.0 * drange + dmin


# ============================================================================
# Gripper Conversion: Binary  to  Continuous
# ============================================================================

def convert_binary_gripper_to_continuous(gripper_seq):
    """
    .

    : gripper_seq (N,) -  -1 (open)  +1 (close)
    : (N,) -  [0, 1], 0=close, 1=open

    :
      open  = 0.04
      close = 0.02
    :  5 
    : (physical - 0.02) / 0.02
    """
    N = len(gripper_seq)
    is_open = gripper_seq < 0  # -1 = open
    physical = np.where(is_open, 0.04, 0.02).astype(np.float64)

    # 
    for i in range(1, N):
        if gripper_seq[i] != gripper_seq[i - 1]:
            if gripper_seq[i] > 0:
                # open  to  close: 0.04  to  0.02 over 5 steps
                for j in range(5):
                    idx = i + j
                    if idx < N and gripper_seq[idx] > 0:
                        t = (j + 1) / 5.0
                        physical[idx] = 0.04 * (1 - t) + 0.02 * t
            else:
                # close  to  open: 0.02  to  0.04 over 5 steps
                for j in range(5):
                    idx = i + j
                    if idx < N and gripper_seq[idx] < 0:
                        t = (j + 1) / 5.0
                        physical[idx] = 0.02 * (1 - t) + 0.04 * t

    #  [0, 1]: 0.02  to  0, 0.04  to  1
    normalized = (physical - 0.02) / 0.02
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


# ============================================================================
# Dataset
# ============================================================================

def get_active_mask(action):
    """ EE delta  mask."""
    return np.abs(action[:, :6]).sum(axis=1) > 1e-8


class BCDataset(Dataset):
    """
     HDF5 , :
      - : traj_X/action, traj_X/observations/qpos, .../images/front, ...
      - robomimic : data/demo_X/actions, data/demo_X/obs/qpos, .../front, ...

    action_chunk_size:  chunk_size  action  (action chunking).
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

        # :  action_chunk_size ,  trajectory
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

        # Gripper distribution
        gripper = self.actions[:, -1]
        print(f"  gripper range: [{gripper.min():.4f}, {gripper.max():.4f}]")
        print(f"  gripper mean: {gripper.mean():.4f}")

    def _load_robomimic(self, data, load_count, remove_idle):
        """ robomimic ."""
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
            action = np.array(demo["actions"])        # (N, 7)
            qpos = np.array(demo["obs/qpos"])         # (N, 7)
            qvel = np.array(demo["obs/qvel"])         # (N, 7)
            front = np.array(demo["obs/front"])       # (N, 256, 256, 3)
            wrist = np.array(demo["obs/wrist"])       # (N, 256, 256, 3)

            if remove_idle and self.action_chunk_size <= 1:
                mask = get_active_mask(action)
                if mask.sum() == 0:
                    continue
                action = action[mask]
                qpos = qpos[mask]
                qvel = qvel[mask]
                front = front[mask]
                wrist = wrist[mask]
            elif remove_idle:
                print(f"[Dataset] WARN: remove_idle disabled for action_chunk_size={self.action_chunk_size} "
                      f"to preserve temporal continuity")

            # EE delta: ,  100 
            norm_ee = (action[:, :6] * 100.0).astype(np.float32)
            # Gripper:   to   [0, 1]
            gripper_cont = convert_binary_gripper_to_continuous(action[:, -1])
            #  7  action
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

        # robomimic : ee ,  [-1, 1]
        self.action_ee_min = -np.ones(6, dtype=np.float32)
        self.action_ee_max = np.ones(6, dtype=np.float32)
        self.qpos_min = -np.ones(self.states.shape[1] // 2, dtype=np.float32)
        self.qpos_max = np.ones(self.states.shape[1] // 2, dtype=np.float32)
        self.qvel_min = -np.ones(self.states.shape[1] // 2, dtype=np.float32)
        self.qvel_max = np.ones(self.states.shape[1] // 2, dtype=np.float32)

    def _load_raw(self, data, load_count, remove_idle):
        """,  min-max ."""
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

            if remove_idle and self.action_chunk_size <= 1:
                mask = get_active_mask(action)
                if mask.sum() == 0:
                    continue
                action = action[mask]
                qpos = qpos[mask]
                qvel = qvel[mask]
                front = front[mask]
                wrist = wrist[mask]
            elif remove_idle:
                print(f"[Dataset] WARN: remove_idle disabled for action_chunk_size={self.action_chunk_size} "
                      f"to preserve temporal continuity")

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

        print(f"  action_ee min: {self.action_ee_min}")
        print(f"  action_ee max: {self.action_ee_max}")

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
            "front": torch.from_numpy(self.front_images[i]).float(),      # (H, W, 3)
            "wrist": torch.from_numpy(self.wrist_images[i]).float(),      # (H, W, 3)
            "state": torch.from_numpy(self.states[i]).float(),            # (state_dim,)
            "action": torch.from_numpy(self.actions[i:i + C]).float(),    # (C, 7)
        }

    def get_norm_params(self):
        return {
            "action_ee": {"min": self.action_ee_min.tolist(), "max": self.action_ee_max.tolist()},
            "qpos": {"min": self.qpos_min.tolist(), "max": self.qpos_max.tolist()},
            "qvel": {"min": self.qvel_min.tolist(), "max": self.qvel_max.tolist()},
        }


# ============================================================================
# Sampler ()
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
# Model
# ============================================================================

def make_mlp(in_channels, mlp_channels, act_builder=nn.ReLU, last_act=True):
    c_in = in_channels
    module_list = []
    for idx, c_out in enumerate(mlp_channels):
        module_list.append(nn.Linear(c_in, c_out))
        if last_act or idx < len(mlp_channels) - 1:
            module_list.append(act_builder())
        c_in = c_out
    return nn.Sequential(*module_list)


class SpatialSoftmax(nn.Module):
    """Spatial Softmax:  (B, C, H, W)  (B, C*2) ."""

    def __init__(self, height, width, num_channels):
        super().__init__()
        x = torch.linspace(-1, 1, width)
        y = torch.linspace(-1, 1, height)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        self.register_buffer("xx", xx.reshape(1, 1, height, width))
        self.register_buffer("yy", yy.reshape(1, 1, height, width))

    def forward(self, features):
        # features: (B, C, H, W)
        B, C, H, W = features.shape
        weights = F.softmax(features.reshape(B, C, -1), dim=-1).reshape(B, C, H, W)
        expected_x = (weights * self.xx).sum(dim=[2, 3])  # (B, C)
        expected_y = (weights * self.yy).sum(dim=[2, 3])  # (B, C)
        return torch.cat([expected_x, expected_y], dim=-1)  # (B, C*2)


class ResNet18Encoder(nn.Module):
    """
    ResNet18 encoder with Spatial Softmax.

    : (B, H, W, C) float [0, 1]
    : (B, out_dim)

    finetune_from:  ('layer3', 'layer4', 'all')
    """

    def __init__(self, in_channels=3, out_dim=256, finetune_from="all"):
        super().__init__()
        import torchvision.models as models

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Spatial Softmax on layer4 output (512 channels, 8x8 for 256x256 input)
        self.spatial_softmax = SpatialSoftmax(8, 8, 512)
        # 512 * 2 = 1024  to  out_dim
        self.fc = nn.Linear(1024, out_dim)

        # ImageNet normalization
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # 
        for param in self.parameters():
            param.requires_grad = False

        unfreeze_layers = {
            "layer4": [self.layer4],
            "layer3": [self.layer3, self.layer4],
            "layer2": [self.layer2, self.layer3, self.layer4],
            "all": [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4],
        }
        for layer in unfreeze_layers.get(finetune_from, [self.layer3, self.layer4]):
            for param in layer.parameters():
                param.requires_grad = True

        # spatial_softmax  fc 
        for param in self.spatial_softmax.parameters():
            param.requires_grad = True
        for param in self.fc.parameters():
            param.requires_grad = True

    def forward(self, image):
        # image: (B, H, W, C) float [0, 1]  to  (B, C, H, W), ImageNet normalize
        x = image.permute(0, 3, 1, 2)
        x = (x - self.img_mean) / self.img_std

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)   # (B, 512, 8, 8)

        x = self.spatial_softmax(x)  # (B, 1024)
        x = self.fc(x)              # (B, out_dim)
        return x


class BCGaussianPolicy(nn.Module):
    """
    BC  - Gaussian .

     action  μ,  log_std.
    Loss: Gaussian Negative Log-Likelihood.

    action = [ee_delta(6), gripper(1)] = 7 
    """

    def __init__(self, state_dim, action_dim=7, visual_feat_dim=256,
                 action_chunk_size=1):
        super().__init__()
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size

        #  ResNet18 encoder
        self.front_encoder = ResNet18Encoder(in_channels=3, out_dim=visual_feat_dim)
        self.wrist_encoder = ResNet18Encoder(in_channels=3, out_dim=visual_feat_dim)

        # MLP: visual_feat * 2 + state  to  action mean
        feat_dim = visual_feat_dim * 2 + state_dim
        self.mlp = make_mlp(feat_dim, [1024, 512, 256], last_act=True)
        self.mean_head = nn.Linear(256, action_dim * action_chunk_size)

        #  log_std (input-independent,  action )
        self.log_std = nn.Parameter(torch.zeros(action_chunk_size, action_dim))

    def forward(self, front, wrist, state):
        """
        front: (B, H, W, C), wrist: (B, H, W, C), state: (B, state_dim)
        : mean (B, C, 7), log_std (B, C, 7)
        """
        feat_front = self.front_encoder(front)   # (B, 256)
        feat_wrist = self.wrist_encoder(wrist)   # (B, 256)
        feat = torch.cat([feat_front, feat_wrist, state], dim=-1)

        h = self.mlp(feat)
        mean = self.mean_head(h)  # (B, action_dim * C)

        B = front.shape[0]
        C = self.action_chunk_size
        mean = mean.view(B, C, self.action_dim)

        # log_std: clamp ,  σ 
        log_std = self.log_std.clamp(-5.0, 0.5)
        log_std = log_std.unsqueeze(0).expand(B, -1, -1)  # (B, C, 7)

        return mean, log_std

    def get_action(self, front, wrist, state):
        """
        ,  ().
        front: (1, H, W, C), wrist: (1, H, W, C), state: (1, D)
        : (1, C, 7) action chunk
        """
        with torch.no_grad():
            mean, _ = self.forward(front, wrist, state)
        return mean


# ============================================================================
# Loss
# ============================================================================

def gaussian_nll_loss(mean, log_std, target):
    """
    Gaussian Negative Log-Likelihood Loss.

    L = 0.5 * (log(2π) + 2*log_std + (target - mean)^2 / exp(2*log_std))

    mean:   (B, C, D)
    log_std: (B, C, D)
    target: (B, C, D)
    """
    var = torch.exp(2 * log_std)
    nll = 0.5 * (np.log(2 * np.pi) + 2 * log_std + (target - mean) ** 2 / var)
    return nll.mean()


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
            "state_dim": policy.mlp[0].in_features - 256 * 2,
            "action_dim": policy.action_dim,
            "action_chunk_size": policy.action_chunk_size,
        },
    }, path)
    print(f"[Checkpoint] Saved: {path}")
    return path


def run_eval_subprocess(args, ckpt_path: str, run_dir: str, iteration: int) -> dict | None:
    """ ckpt  eval_rollout_bc.py ."""
    if not args.eval_task_config or not args.eval_task:
        return None

    eval_dir = os.path.join(run_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    output_json = os.path.join(eval_dir, f"eval_iter_{iteration}.json")

    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_rollout_bc.py")
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
    parser = argparse.ArgumentParser(description="BC Training (Gaussian NLL)")
    parser.add_argument("--dataset", type=str, required=True, help="Path to HDF5 demo dataset")
    parser.add_argument("--num_demos", type=int, default=None, help="Number of demos to load")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--total_iters", type=int, default=200_000, help="Total training iterations")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--no_remove_idle", action="store_true",
                        help="Disable removing idle (zero EE delta) steps")
    parser.add_argument("--action_chunk_size", type=int, default=8,
                        help="Action chunk size:  m ")
    parser.add_argument("--action_horizon", type=int, default=4,
                        help="Action horizon:  n  (n <= action_chunk_size)")
    parser.add_argument("--log_freq", type=int, default=50, help="Logging frequency")
    parser.add_argument("--save_freq", type=int, default=200, help="Checkpoint saving frequency")

    # 
    parser.add_argument("--stat", type=str, default=None,
                        help="dataset_stat.json;  norm_params ")
    parser.add_argument("--eval_task_config", type=str, default=None,
                        help="task yaml ; ")
    parser.add_argument("--eval_task", type=str, default=None,
                        help="gym  id, ")
    parser.add_argument("--eval_num_rollouts", type=int, default=5)
    parser.add_argument("--eval_horizon", type=int, default=300)
    parser.add_argument("--eval_pick_lift_threshold", type=float, default=0.05)
    parser.add_argument("--eval_env_cfg_class", type=str, default=None,
                        help=" 'pkg.mod:ClassName'  pick env cfg")

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

    assert args.action_horizon <= args.action_chunk_size,\
        f"action_horizon ({args.action_horizon}) must <= action_chunk_size ({args.action_chunk_size})"

    # Run name
    if args.exp_name is None:
        eval_task_tag = args.eval_task if args.eval_task else "noeval"
        args.exp_name = (f"bc_gnll_chunk{args.action_chunk_size}"
                         f"_horizon{args.action_horizon}_{os.path.basename(args.dataset).replace('.h5', '')}"
                         f"_{eval_task_tag}_{int(time.time())}")
    run_dir = os.path.join("logs", args.exp_name)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    writer = SummaryWriter(run_dir)

    # Dataset
    ds = BCDataset(
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
    state_dim = ds.states.shape[1]  # 14 (qpos 7 + qvel 7)
    policy = BCGaussianPolicy(
        state_dim=state_dim,
        action_dim=7,  # 6 ee + 1 gripper
        action_chunk_size=args.action_chunk_size,
    ).to(device)

    n_total = sum(p.numel() for p in policy.parameters())
    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"[Model] BCGaussianPolicy: ResNet18 + SpatialSoftmax + MLP")
    print(f"[Model] chunk={args.action_chunk_size}, horizon={args.action_horizon}")
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
        front = batch["front"].to(device)       # (B, H, W, 3)
        wrist = batch["wrist"].to(device)       # (B, H, W, 3)
        state = batch["state"].to(device)       # (B, state_dim)
        target = batch["action"].to(device)     # (B, C, 7)

        # Forward
        pred_mean, pred_log_std = policy(front, wrist, state)
        # pred_mean: (B, C, 7), pred_log_std: (B, C, 7)

        # Gaussian NLL loss
        loss = gaussian_nll_loss(pred_mean, pred_log_std, target)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Logging
        if iteration % args.log_freq == 0:
            with torch.no_grad():
                #  loss
                ee_target = target[:, :, :6]
                ee_mean = pred_mean[:, :, :6]
                ee_log_std = pred_log_std[:, :, :6]
                gripper_target = target[:, :, 6:]
                gripper_mean = pred_mean[:, :, 6:]
                gripper_log_std = pred_log_std[:, :, 6:]

                loss_ee = gaussian_nll_loss(ee_mean, ee_log_std, ee_target)
                loss_gripper = gaussian_nll_loss(gripper_mean, gripper_log_std, gripper_target)

                # EE MSE (for monitoring)
                ee_mse = F.mse_loss(ee_mean, ee_target)

                # Gripper accuracy (threshold at 0.5)
                gripper_pred_binary = (gripper_mean > 0.5).float()
                gripper_target_binary = (gripper_target > 0.5).float()
                gripper_acc = (gripper_pred_binary == gripper_target_binary).float().mean().item()

                # learned std
                std_vals = torch.exp(policy.log_std.clamp(-5.0, 0.5))
                mean_std = std_vals.mean().item()

            print(f"[Iter {iteration}] loss={loss.item():.4f} "
                  f"nll_ee={loss_ee.item():.4f} nll_gripper={loss_gripper.item():.4f} "
                  f"ee_mse={ee_mse.item():.4f} gripper_acc={gripper_acc:.4f} "
                  f"mean_std={mean_std:.4f} lr={scheduler.get_last_lr()[0]:.6f}")

            writer.add_scalar("loss/total_nll", loss.item(), iteration)
            writer.add_scalar("loss/nll_ee", loss_ee.item(), iteration)
            writer.add_scalar("loss/nll_gripper", loss_gripper.item(), iteration)
            writer.add_scalar("metrics/ee_mse", ee_mse.item(), iteration)
            writer.add_scalar("metrics/gripper_accuracy", gripper_acc, iteration)
            writer.add_scalar("metrics/mean_std", mean_std, iteration)
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
                        print(f"   to  [WARN] failed to log GIF to tensorboard: {e}")
                print(f"   to  Eval overall={overall:.2%}, "
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
                print(f"   to  [WARN] failed to log GIF to tensorboard: {e}")

    writer.close()
    print(f"[INFO] Training complete. Results saved to {run_dir}")


if __name__ == "__main__":
    main()
