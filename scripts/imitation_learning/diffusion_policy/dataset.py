"""
Diffusion Policy Dataset.

从 robomimic 格式的 HDF5 文件加载数据, 支持 action chunking (预测未来 T_a 步动作)
和 observation history (使用过去 T_o 步观测).

HDF5 格式:
  data/demo_X/actions       (N, 7)    ee_delta[-1,1] + gripper[-1,+1]
  data/demo_X/obs/qpos      (N, 7)    归一化到 [-1, 1]
  data/demo_X/obs/qvel      (N, 7)    归一化到 [-1, 1]
  data/demo_X/obs/front     (N, 256, 256, 3) uint8
  data/demo_X/obs/wrist     (N, 256, 256, 3) uint8
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class DiffusionPolicyDataset(Dataset):
    """支持 action chunking 的数据集.

    每个样本返回:
      - obs_images:  (T_o, 2, 3, H, W)  过去 T_o 帧的双视角 RGB 图像
      - obs_lowdim:  (T_o, D_low)        过去 T_o 帧的低维观测 (qpos + qvel)
      - actions:     (T_a, D_act)        未来 T_a 步动作 (action chunk)
    """

    def __init__(
        self,
        hdf5_path: str,
        pred_horizon: int = 16,     # T_a: 预测的动作序列长度
        obs_horizon: int = 2,       # T_o: 观测历史长度
        img_size: int = 256,
        filter_key: str = "train",  # "train" 或 "valid"
    ):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.img_size = img_size

        # 打开 HDF5 文件
        self.hdf5_file = h5py.File(hdf5_path, "r")

        # 根据 filter_key 获取 demo 列表
        if filter_key is not None and "mask" in self.hdf5_file:
            demo_names = [
                name.decode() if isinstance(name, bytes) else name
                for name in self.hdf5_file["mask"][filter_key][:]
            ]
        else:
            demo_names = sorted(self.hdf5_file["data"].keys())

        # 构建 (demo_name, start_idx) 的索引表
        # 每个样本的起始索引: 从 obs_horizon-1 开始, 到 len - pred_horizon 结束
        self.indices = []
        self.demo_lengths = {}
        for demo_name in demo_names:
            demo = self.hdf5_file["data"][demo_name]
            n_steps = demo["actions"].shape[0]
            self.demo_lengths[demo_name] = n_steps
            # 起始位置: 需要至少 obs_horizon 帧历史, 以及 pred_horizon 帧未来
            min_start = obs_horizon - 1
            max_start = n_steps - pred_horizon
            if max_start < min_start:
                # 轨迹太短, 跳过
                continue
            for idx in range(min_start, max_start + 1):
                self.indices.append((demo_name, idx))

        print(f"[DiffusionPolicyDataset] 加载 {len(demo_names)} 条轨迹, "
              f"共 {len(self.indices)} 个样本 (T_o={obs_horizon}, T_a={pred_horizon})")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        demo_name, start = self.indices[idx]
        demo = self.hdf5_file["data"][demo_name]

        # --- 加载观测 (过去 T_o 帧) ---
        obs_start = start - self.obs_horizon + 1
        obs_end = start + 1  # [obs_start, obs_end)

        # 低维观测: qpos + qvel
        qpos = np.array(demo["obs"]["qpos"][obs_start:obs_end], dtype=np.float32)  # (T_o, 7)
        qvel = np.array(demo["obs"]["qvel"][obs_start:obs_end], dtype=np.float32)  # (T_o, 7)
        obs_lowdim = np.concatenate([qpos, qvel], axis=-1)  # (T_o, 14)

        # 图像观测: front + wrist
        front = np.array(demo["obs"]["front"][obs_start:obs_end], dtype=np.float32)  # (T_o, H, W, 3)
        wrist = np.array(demo["obs"]["wrist"][obs_start:obs_end], dtype=np.float32)  # (T_o, H, W, 3)

        # 归一化到 [0, 1] 并转为 CHW
        front = front / 255.0
        wrist = wrist / 255.0
        front = front.transpose(0, 3, 1, 2)  # (T_o, 3, H, W)
        wrist = wrist.transpose(0, 3, 1, 2)  # (T_o, 3, H, W)

        # 拼接双视角: (T_o, 2, 3, H, W)
        obs_images = np.stack([front, wrist], axis=1)

        # --- 加载动作 (未来 T_a 帧) ---
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
