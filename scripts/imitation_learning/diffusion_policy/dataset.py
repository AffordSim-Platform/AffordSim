"""
Diffusion Policy Dataset.

Loads data from robomimic-format HDF5 files, supporting action chunking (predicting
future T_a action steps) and observation history (using past T_o observation steps).

HDF5 format:
  data/demo_X/actions       (N, 7)    ee_delta[-1,1] + gripper[-1,+1]
  data/demo_X/obs/qpos      (N, 7)    normalized to [-1, 1]
  data/demo_X/obs/qvel      (N, 7)    normalized to [-1, 1]
  data/demo_X/obs/front     (N, 256, 256, 3) uint8
  data/demo_X/obs/wrist     (N, 256, 256, 3) uint8
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class DiffusionPolicyDataset(Dataset):
    """Dataset supporting action chunking.

    Each sample returns:
      - obs_images:  (T_o, 2, 3, H, W)  dual-view RGB images from the past T_o frames
      - obs_lowdim:  (T_o, D_low)        low-dimensional observations from the past T_o frames (qpos + qvel)
      - actions:     (T_a, D_act)        future T_a action steps (action chunk)
    """

    def __init__(
        self,
        hdf5_path: str,
        pred_horizon: int = 16,     # T_a: predicted action sequence length
        obs_horizon: int = 2,       # T_o: observation history length
        img_size: int = 256,
        filter_key: str = "train",  # "train" or "valid"
    ):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.img_size = img_size

        # Open HDF5 file
        self.hdf5_file = h5py.File(hdf5_path, "r")

        # Get demo list based on filter_key
        if filter_key is not None and "mask" in self.hdf5_file:
            demo_names = [
                name.decode() if isinstance(name, bytes) else name
                for name in self.hdf5_file["mask"][filter_key][:]
            ]
        else:
            demo_names = sorted(self.hdf5_file["data"].keys())

        # Build index table of (demo_name, start_idx)
        # Start index for each sample: from obs_horizon-1 to len - pred_horizon
        self.indices = []
        self.demo_lengths = {}
        for demo_name in demo_names:
            demo = self.hdf5_file["data"][demo_name]
            n_steps = demo["actions"].shape[0]
            self.demo_lengths[demo_name] = n_steps
            # Start position: need at least obs_horizon frames of history and pred_horizon frames of future
            min_start = obs_horizon - 1
            max_start = n_steps - pred_horizon
            if max_start < min_start:
                # Trajectory too short, skip
                continue
            for idx in range(min_start, max_start + 1):
                self.indices.append((demo_name, idx))

        print(f"[DiffusionPolicyDataset] Loaded {len(demo_names)} trajectories, "
              f"{len(self.indices)} samples total (T_o={obs_horizon}, T_a={pred_horizon})")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        demo_name, start = self.indices[idx]
        demo = self.hdf5_file["data"][demo_name]

        # --- Load observations (past T_o frames) ---
        obs_start = start - self.obs_horizon + 1
        obs_end = start + 1  # [obs_start, obs_end)

        # Low-dimensional observations: qpos + qvel
        qpos = np.array(demo["obs"]["qpos"][obs_start:obs_end], dtype=np.float32)  # (T_o, 7)
        qvel = np.array(demo["obs"]["qvel"][obs_start:obs_end], dtype=np.float32)  # (T_o, 7)
        obs_lowdim = np.concatenate([qpos, qvel], axis=-1)  # (T_o, 14)

        # Image observations: front + wrist
        front = np.array(demo["obs"]["front"][obs_start:obs_end], dtype=np.float32)  # (T_o, H, W, 3)
        wrist = np.array(demo["obs"]["wrist"][obs_start:obs_end], dtype=np.float32)  # (T_o, H, W, 3)

        # Normalize to [0, 1] and convert to CHW
        front = front / 255.0
        wrist = wrist / 255.0
        front = front.transpose(0, 3, 1, 2)  # (T_o, 3, H, W)
        wrist = wrist.transpose(0, 3, 1, 2)  # (T_o, 3, H, W)

        # Stack dual views: (T_o, 2, 3, H, W)
        obs_images = np.stack([front, wrist], axis=1)

        # --- Load actions (future T_a frames) ---
        act_start = start
        act_end = start + self.pred_horizon
        actions = np.array(demo["actions"][act_start:act_end], dtype=np.float32)  # (T_a, 7)

        return {
            "obs_images": torch.from_numpy(obs_images),    # (T_o, 2, 3, H, W)
            "obs_lowdim": torch.from_numpy(obs_lowdim),    # (T_o, 14)
            "actions": torch.from_numpy(actions),           # (T_a, 7)
        }

    def close(self):
        self.hdf5_file.close()
