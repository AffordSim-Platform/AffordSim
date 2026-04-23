"""
Diffusion Policy Dataset (ACT-aligned style).

数据流:
  - action arm 已由 upstream minmax 到 [-1, 1], dataset 这里再 × 100 放大 SNR
    (DDPM 的 noisy_action = sqrt(α̅)·action + sqrt(1-α̅)·noise, action 不放大时噪声
     盖过 0 附近的小值, 模型塌成预测 0)
  - gripper 从 ±1 二值 → [0, 1] 连续 (带 5 步 ramp), 训练目标更平滑
  - idle 过滤由 upstream 做过, 这里不重复
  - 图像 uint8 原样存内存, `__getitem__` 里 /255 → [0, 1]
  - 轨迹边界: pad_before = T_o - 1, pad_after = T_a - T_o (末尾用 arm zeros + last gripper)

与 ACT 的 `train_act.py._load_robomimic` 预处理完全一致, 只是额外加了 T_o/T_a 窗口和 pad.

每个样本返回:
  - obs_images:  (T_o, 2, 3, H, W)  双视角 RGB, [0, 1]
  - obs_lowdim:  (T_o, D_low)       qpos + qvel
  - actions:     (T_a, D_act)       arm ∈ [-100, 100], gripper ∈ [0, 1]
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


# ============================================================================
# Helpers (与 ACT 一致)
# ============================================================================

def convert_binary_gripper_to_continuous(gripper_seq):
    """二值 gripper (-1 open / +1 close) -> [0,1] 连续 (0=close, 1=open) + 5 步 ramp."""
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


class DiffusionPolicyDataset(Dataset):
    """按 ManiSkill 参考实现的 pad_before / pad_after 策略切片."""

    def __init__(
        self,
        hdf5_path: str,
        pred_horizon: int = 16,          # T_a
        obs_horizon: int = 2,            # T_o
        img_size: int = 256,
        load_count: int | None = None,
        filter_key: str | None = None,   # robomimic mask split, e.g. "train" / "valid"
    ):
        super().__init__()
        self.hdf5_path = hdf5_path
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.img_size = img_size

        with h5py.File(hdf5_path, "r") as data:
            top_keys = list(data.keys())
            if "data" not in top_keys:
                raise NotImplementedError(
                    "Dataset expects robomimic format (data/demo_X/...). "
                    "请先用 scripts/imitation_learning/robomimic/convert_hdf5_for_robomimic.py 转换."
                )
            print(f"[DPDataset] Loading robomimic HDF5: {hdf5_path}")
            self._load_robomimic(data, load_count, filter_key)

        # 构建采样 slices, 与 ManiSkill 参考 train.py:130-137 一致
        pad_before = obs_horizon - 1
        pad_after = pred_horizon - obs_horizon
        self.slices = []
        for traj_idx, L in enumerate(self.traj_lengths):
            # start 范围: [-pad_before, L - pred_horizon + pad_after]  (含两端)
            start_lo = -pad_before
            start_hi = L - pred_horizon + pad_after
            for start in range(start_lo, start_hi + 1):
                self.slices.append((traj_idx, start))

        # arm 零向量模板, 末尾 pad_after 时需要 (arm zeros + last gripper)
        act_dim = self.actions[0].shape[1]
        self.arm_zero = np.zeros(act_dim - 1, dtype=np.float32)

        total_frames = sum(self.traj_lengths)
        print(f"[DPDataset] trajectories={len(self.traj_lengths)}, total_frames={total_frames}, "
              f"samples (after pad)={len(self.slices)}")
        print(f"  state dim:  {self.states[0].shape[1]}")
        print(f"  action dim: {act_dim}")
        print(f"  T_o={obs_horizon}, T_a={pred_horizon}, "
              f"pad_before={pad_before}, pad_after={pad_after}")

    # --------------------------------------------------------------------
    def _load_robomimic(self, data, load_count, filter_key):
        data_grp = data["data"]
        if filter_key is not None and "mask" in data:
            demo_keys = [
                name.decode() if isinstance(name, bytes) else name
                for name in data["mask"][filter_key][:]
            ]
            demo_keys = sorted(demo_keys, key=lambda x: int(x.split("_")[1]))
        else:
            demo_keys = sorted(
                [k for k in data_grp.keys() if k.startswith("demo_")],
                key=lambda x: int(x.split("_")[1]),
            )
        if load_count is not None:
            demo_keys = demo_keys[:load_count]

        # 保持 per-demo 列表 (不 concat), 方便边界处单独 pad.
        self.front_images = []   # list of (L_i, H, W, 3) uint8
        self.wrist_images = []   # list of (L_i, H, W, 3) uint8
        self.states = []         # list of (L_i, D_low) float32  (qpos+qvel, 已归一化)
        self.actions = []        # list of (L_i, D_act) float32  (arm [-1,1] + gripper ±1)
        self.traj_lengths = []

        for demo_key in tqdm(demo_keys, desc="Loading demos"):
            demo = data_grp[demo_key]
            action_raw = np.array(demo["actions"]).astype(np.float32)   # arm upstream 已到 [-1,1], gripper 仍 ±1
            qpos = np.array(demo["obs/qpos"]).astype(np.float32)        # upstream 已归一化
            qvel = np.array(demo["obs/qvel"]).astype(np.float32)        # upstream 已归一化
            front = np.array(demo["obs/front"], dtype=np.uint8)
            wrist = np.array(demo["obs/wrist"], dtype=np.uint8)

            # ACT 风格二次处理:
            #   arm 列 × 100 -> [-100, 100] 放 SNR
            #   gripper 二值 ±1 -> 连续 [0, 1] (close=0, open=1)
            arm_scaled = (action_raw[:, :-1] * 100.0).astype(np.float32)
            gripper_cont = convert_binary_gripper_to_continuous(action_raw[:, -1])
            action = np.concatenate([arm_scaled, gripper_cont[:, None]], axis=-1)

            state = np.concatenate([qpos, qvel], axis=-1)

            self.front_images.append(np.ascontiguousarray(front))
            self.wrist_images.append(np.ascontiguousarray(wrist))
            self.states.append(state)
            self.actions.append(action)
            self.traj_lengths.append(len(action))

    # --------------------------------------------------------------------
    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        traj_idx, start = self.slices[idx]
        T_o = self.obs_horizon
        T_a = self.pred_horizon
        L = self.traj_lengths[traj_idx]
        end = start + T_a

        # --- 观测 (T_o 帧, 起始在 start) ---
        # slice [max(0, start) : start + T_o] 后, 若 start<0 则在前面重复第一帧
        obs_lo = max(0, start)
        obs_hi = start + T_o   # 由 slices 构造保证 <= L

        state_slice = self.states[traj_idx][obs_lo:obs_hi]
        front_slice = self.front_images[traj_idx][obs_lo:obs_hi]
        wrist_slice = self.wrist_images[traj_idx][obs_lo:obs_hi]

        # --- 动作 (T_a 帧, 起始在 start, 结束在 end) ---
        act_lo = max(0, start)
        act_hi = min(end, L)
        act_slice = self.actions[traj_idx][act_lo:act_hi]

        # --- Pad before (起点在 traj 之前): 重复第一帧 ---
        if start < 0:
            pad = -start
            state_slice = np.concatenate([np.repeat(state_slice[:1], pad, axis=0), state_slice], axis=0)
            front_slice = np.concatenate([np.repeat(front_slice[:1], pad, axis=0), front_slice], axis=0)
            wrist_slice = np.concatenate([np.repeat(wrist_slice[:1], pad, axis=0), wrist_slice], axis=0)
            act_slice   = np.concatenate([np.repeat(act_slice[:1],   pad, axis=0), act_slice],   axis=0)

        # --- Pad after (end 越过 traj 末尾): 只发生在 action 侧, arm zeros + last gripper ---
        if end > L:
            pad = end - L
            last_gripper = act_slice[-1:, -1:]                 # (1, 1)
            pad_action = np.concatenate(
                [np.broadcast_to(self.arm_zero, (1, self.arm_zero.shape[0])), last_gripper], axis=-1,
            )                                                   # (1, D_act)
            act_slice = np.concatenate([act_slice, np.repeat(pad_action, pad, axis=0)], axis=0)

        assert state_slice.shape[0] == T_o, (state_slice.shape, T_o, start, L)
        assert act_slice.shape[0] == T_a, (act_slice.shape, T_a, start, L)

        # uint8 (T_o, H, W, 3) -> float32 [0,1] CHW, /255 推迟到这里做 (省 4× 内存)
        front_f = front_slice.astype(np.float32) / 255.0
        wrist_f = wrist_slice.astype(np.float32) / 255.0
        front_f = front_f.transpose(0, 3, 1, 2)                # (T_o, 3, H, W)
        wrist_f = wrist_f.transpose(0, 3, 1, 2)
        obs_images = np.stack([front_f, wrist_f], axis=1)      # (T_o, 2, 3, H, W)

        return {
            "obs_images": torch.from_numpy(np.ascontiguousarray(obs_images)).float(),
            "obs_lowdim": torch.from_numpy(np.ascontiguousarray(state_slice)).float(),
            "actions": torch.from_numpy(np.ascontiguousarray(act_slice)).float(),
        }

    def close(self):
        # 现在不持有 hdf5 handle, 保留以兼容旧调用方.
        pass
