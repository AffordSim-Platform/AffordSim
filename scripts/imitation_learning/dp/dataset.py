"""
Diffusion Policy Dataset ( diffusion_policy/dataset.py ).

 robomimic HDF5 :
  data/demo_X/actions        (N, D_act)   min-max  [-1, 1]
  data/demo_X/obs/qpos       (N, 7)       [-1, 1]
  data/demo_X/obs/qvel       (N, 7)       [-1, 1]
  data/demo_X/obs/front      (N, H, W, 3) uint8
  data/demo_X/obs/wrist      (N, H, W, 3) uint8
  mask/train, mask/valid     demo 

:
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

        #  hdf5  demo  obs  qvel.
        # use_qvel=None  to  ; True/False  to   ( True  qvel ).
        first_demo = next(iter(self.hdf5_file["data"].values()))
        src_has_qvel = "qvel" in first_demo["obs"]
        if use_qvel is None:
            use_qvel = src_has_qvel
        elif use_qvel and not src_has_qvel:
            raise ValueError(f"use_qvel=True  hdf5  obs/qvel: {hdf5_path}")
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
        #  SequenceSampler : allow pad_before=obs_horizon-1, pad_after=pred_horizon-1 
        #  padding (/). ,  dataset.py .
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
