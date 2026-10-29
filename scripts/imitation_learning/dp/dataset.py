"""
Diffusion Policy Dataset ( diffusion_policy/dataset.py ).

 robomimic HDF5 :
  data/demo_X/actions        (N, D_act)   min-max  [-1, 1]
  data/demo_X/obs/qpos       (N, J)       [-1, 1]
  data/demo_X/obs/qvel       (N, J)       [-1, 1]
  data/demo_X/obs/front      (N, H, W, 3) uint8
  data/demo_X/obs/wrist      (N, H, W, 3) uint8
  mask/train, mask/valid     demo 

:
  obs_images: (T_o, 2, 3, H, W) [0,1]
  obs_lowdim: (T_o, J or 2J)    [-1,1]
  actions:    (T_a, D_act)      [-1,1]
"""

from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def infer_demo_dimensions(data_group, demo_names, use_qvel):
    """Validate selected demos before any training batch is constructed."""
    if not demo_names:
        raise ValueError("HDF5 dataset contains no demonstration from which to infer dimensions")
    automatic_qvel = use_qvel is None
    first_obs = data_group[demo_names[0]]["obs"]
    has_qvel = "qvel" in first_obs
    use_qvel = has_qvel if automatic_qvel else bool(use_qvel)
    expected = None
    for name in demo_names:
        demo = data_group[name]
        obs = demo["obs"]
        if "qpos" not in obs:
            raise ValueError(f"{name} lacks obs/qpos")
        qpos = obs["qpos"]
        actions = demo["actions"]
        if len(qpos.shape) != 2 or qpos.shape[1] == 0 or len(actions.shape) != 2:
            raise ValueError(f"{name} needs 2-D qpos and actions with nonzero joint dimension")
        if qpos.shape[0] != actions.shape[0]:
            raise ValueError(f"{name} qpos/action timestep counts do not match")
        if automatic_qvel and ("qvel" in obs) != has_qvel:
            raise ValueError(f"{name} has inconsistent qvel availability; select use_qvel explicitly")
        if use_qvel:
            if "qvel" not in obs:
                raise ValueError(f"{name} lacks obs/qvel but use_qvel=True")
            if obs["qvel"].shape != qpos.shape:
                raise ValueError(f"{name} qvel shape {obs['qvel'].shape} differs from qpos {qpos.shape}")
        dimensions = (qpos.shape[1], actions.shape[1])
        if expected is not None and dimensions != expected:
            raise ValueError(f"Inconsistent qpos/action dimensions in {name}: {dimensions}; expected {expected}. "
                             "Use one compatible embodiment and action representation per dataset.")
        expected = dimensions
    return dict(qpos_dim=expected[0], action_dim=expected[1], use_qvel=use_qvel,
                lowdim_dim=expected[0] * (2 if use_qvel else 1))


def validate_dataset_compatibility(train_ds, valid_ds):
    """Do not silently combine different train/validation state dimensions."""
    if not train_ds.num_demos:
        raise ValueError("Training split contains no demonstrations")
    if not valid_ds.num_demos:
        return
    fields = ("qpos_dim", "action_dim", "use_qvel", "lowdim_dim")
    train = tuple(getattr(train_ds, field) for field in fields)
    valid = tuple(getattr(valid_ds, field) for field in fields)
    if train != valid:
        raise ValueError(f"Train and validation dataset dimensions differ: {train} versus {valid}")


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

        if filter_key is not None and "mask" in self.hdf5_file:
            demo_names = [
                name.decode() if isinstance(name, bytes) else name
                for name in self.hdf5_file["mask"][filter_key][:]
            ]
        else:
            demo_names = sorted(self.hdf5_file["data"].keys())

        if load_count is not None:
            demo_names = demo_names[:load_count]
        self.num_demos = len(demo_names)

        # An empty validation split can still use the training dataset's schema;
        # nonempty splits are validated over exactly the demos they load.
        dimension_names = demo_names or sorted(self.hdf5_file["data"].keys())[:1]
        try:
            dimensions = infer_demo_dimensions(self.hdf5_file["data"], dimension_names, use_qvel)
        except (KeyError, ValueError):
            self.hdf5_file.close()
            raise
        self.qpos_dim = dimensions['qpos_dim']
        self.action_dim = dimensions['action_dim']
        self.lowdim_dim = dimensions['lowdim_dim']
        self.use_qvel = dimensions['use_qvel']

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
            f"use_qvel={self.use_qvel}, qpos_dim={self.qpos_dim}, lowdim_dim={self.lowdim_dim})"
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
