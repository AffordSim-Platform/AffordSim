"""
Diffusion Policy Dataset (同现有 diffusion_policy/dataset.py 的输出契约).

从 robomimic HDF5 加载:
  data/demo_X/actions        (N, D_act)  已在离线 min-max 归一化到 [-1, 1]
  data/demo_X/obs/qpos       (N, 7)      已归一化到 [-1, 1]
  data/demo_X/obs/qvel       (N, 7)      已归一化到 [-1, 1]
  data/demo_X/obs/front      (N, H, W, 3) uint8
  data/demo_X/obs/wrist      (N, H, W, 3) uint8
  mask/train, mask/valid     demo 名字列表

每个样本:
  obs_images: (T_o, 2, 3, H, W) [0,1]
  obs_lowdim: (T_o, 14)         [-1,1]
  actions:    (T_a, D_act)      [-1,1]
"""

from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class DiffusionPolicyDataset(Dataset):
    def __init__(
        self,
        hdf5_path: str,
        pred_horizon: int = 16,
        obs_horizon: int = 2,
        filter_key: str = "train",
        load_count: Optional[int] = None,
        use_qvel: Optional[bool] = None,
    ):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.hdf5_path = hdf5_path

        self.hdf5_file = h5py.File(hdf5_path, "r")

        # 自动探测 hdf5 里第一条 demo 的 obs 有没有 qvel.
        # use_qvel=None -> 自动; True/False -> 强制 (强制 True 但缺 qvel 会报错).
        first_demo = next(iter(self.hdf5_file["data"].values()))
        src_has_qvel = "qvel" in first_demo["obs"]
        if use_qvel is None:
            use_qvel = src_has_qvel
        elif use_qvel and not src_has_qvel:
            raise ValueError(f"use_qvel=True 但 hdf5 中没有 obs/qvel: {hdf5_path}")
        self.use_qvel = bool(use_qvel)

        if filter_key is not None and "mask" in self.hdf5_file:
            demo_names = [
                name.decode() if isinstance(name, bytes) else name
                for name in self.hdf5_file["mask"][filter_key][:]
            ]
        else:
            demo_names = sorted(self.hdf5_file["data"].keys())

        if load_count is not None:
            demo_names = demo_names[:load_count]

        # Receding-horizon sample indices with "padding at edges"
        # 与原版 SequenceSampler 一致: allow pad_before=obs_horizon-1, pad_after=pred_horizon-1 的
        # 边界 padding (通过重复首/尾帧). 这里简化为非边界采样, 跟原 dataset.py 对齐.
        self.indices = []
        for demo_name in demo_names:
            demo = self.hdf5_file["data"][demo_name]
            n = demo["actions"].shape[0]
            min_start = obs_horizon - 1
            max_start = n - pred_horizon
            if max_start < min_start:
                continue
            for idx in range(min_start, max_start + 1):
                self.indices.append((demo_name, idx))

        print(
            f"[DiffusionPolicyDataset:{filter_key}] {len(demo_names)} demos, "
            f"{len(self.indices)} samples (T_o={obs_horizon}, T_a={pred_horizon}, "
            f"use_qvel={self.use_qvel})"
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        demo_name, start = self.indices[idx]
        demo = self.hdf5_file["data"][demo_name]

        obs_start = start - self.obs_horizon + 1
        obs_end = start + 1

        qpos = np.asarray(demo["obs"]["qpos"][obs_start:obs_end], dtype=np.float32)
        if self.use_qvel:
            qvel = np.asarray(demo["obs"]["qvel"][obs_start:obs_end], dtype=np.float32)
            obs_lowdim = np.concatenate([qpos, qvel], axis=-1)
        else:
            obs_lowdim = qpos

        front = np.asarray(demo["obs"]["front"][obs_start:obs_end], dtype=np.float32) / 255.0
        wrist = np.asarray(demo["obs"]["wrist"][obs_start:obs_end], dtype=np.float32) / 255.0
        front = front.transpose(0, 3, 1, 2)  # (T_o, 3, H, W)
        wrist = wrist.transpose(0, 3, 1, 2)
        obs_images = np.stack([front, wrist], axis=1)  # (T_o, 2, 3, H, W)

        actions = np.asarray(
            demo["actions"][start:start + self.pred_horizon], dtype=np.float32,
        )

        return {
            "obs_images": torch.from_numpy(obs_images),
            "obs_lowdim": torch.from_numpy(obs_lowdim),
            "actions": torch.from_numpy(actions),
        }

    def close(self):
        try:
            self.hdf5_file.close()
        except Exception:
            pass
