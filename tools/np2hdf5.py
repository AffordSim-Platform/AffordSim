"""将指定文件夹下的 episode_XXXX.npy 文件转换为 HDF5 格式。

HDF5 的结构参照 reference.hdf5:
  action:              (T, D_act) float32   ee_delta(7) 或 joint_delta(8)
  observations/
    images/
      front:           (T, 256, 256, 3) uint8
      wrist:           (T, 256, 256, 3) uint8
    qpos:              (T, 7)  float32
    qvel:              (T, 7)  float32

Usage:
    python tools/npy2hdf5.py --npy_folder <folder> [--action_type ee_delta|joint_delta] [--out <output_folder>]
"""

import argparse
import os
import re
import glob

import cv2
import h5py
import numpy as np


# ──────────────── 默认路径 ────────────────
DEFAULT_NEW = "/home/limingyang/project/Real2Sim2Real-Pipeline-affordsim_clean/data/r2s2r_data/pour_redcup422/20260422_231824/20260422_231847"
DEFAULT_REF = "/home/limingyang/project/Real2Sim2Real-Pipeline-affordsim_clean/dataset/rdt_dataset/reference.hdf5"
DEFAULT_OUT = "/home/limingyang/project/Real2Sim2Real-Pipeline-affordsim_clean/dataset/r2s2r_data/task_gen/pour_with_affordance/pour_redcup422/hdf5"

# NPY 中图像的原始尺寸 — Isaac Lab 输出是 CHW 格式 (C, H, W)
NPY_IMG_SHAPE = (3, 480, 640)
# HDF5 中图像的目标尺寸
TARGET_IMG_SIZE = (256, 256)


def collect_episode_files(folder: str) -> list[str]:
    """收集文件夹下所有 episode_XXXX.npy，按编号排序。"""
    pattern = os.path.join(folder, "episode_*.npy")
    files = glob.glob(pattern)
    episode_re = re.compile(r"episode_(\d+)\.npy$")
    matched = []
    for f in files:
        m = episode_re.search(f)
        if m:
            matched.append((int(m.group(1)), f))
    matched.sort(key=lambda x: x[0])
    return [f for _, f in matched]


def load_ref_structure(ref_path: str) -> dict:
    """从 reference.hdf5 读取数据结构信息（键名、shape、dtype）。"""
    info = {}
    with h5py.File(ref_path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                info[name] = {"shape": obj.shape, "dtype": obj.dtype}
        f.visititems(visit)
    return info


def reshape_and_resize_images(flat: np.ndarray, src_shape: tuple, target_size: tuple) -> np.ndarray:
    """将 (T, H*W*C) float64 图像数据 reshape + resize 为 (T, tH, tW, C) uint8。"""
    T = flat.shape[0]
    imgs = flat.reshape(T, *src_shape)
    # CHW -> HWC
    if imgs.ndim == 4 and imgs.shape[1] == 3:
        imgs = imgs.transpose(0, 2, 3, 1)
    # 如果是 float 且范围在 [0,1]，转换到 [0,255]
    if imgs.dtype in (np.float32, np.float64):
        if imgs.max() <= 1.0:
            imgs = (imgs * 255).clip(0, 255).astype(np.uint8)
        else:
            imgs = imgs.clip(0, 255).astype(np.uint8)

    tH, tW = target_size
    H, W = imgs.shape[1], imgs.shape[2]
    if H != tH or W != tW:
        # Center crop to square (take shorter side), then resize
        crop_size = min(H, W)
        y0 = (H - crop_size) // 2
        x0 = (W - crop_size) // 2
        imgs = imgs[:, y0:y0+crop_size, x0:x0+crop_size, :]
        # Resize cropped square to target size
        if crop_size != tH or crop_size != tW:
            resized = np.empty((T, tH, tW, 3), dtype=np.uint8)
            for i in range(T):
                resized[i] = cv2.resize(imgs[i], (tW, tH), interpolation=cv2.INTER_AREA)
            return resized
    return imgs


def convert_episode(npy_path: str, action_type: str = "ee_delta") -> dict:
    """将单个 episode npy 转换为 HDF5 所需的字典结构。

    Args:
        npy_path: episode npy 文件路径
        action_type: "ee_delta" -> obs_policy_actions (T,7)
                     "joint_delta" -> obs_policy_joint_delta_actions (T,8)
    """
    data = np.load(npy_path, allow_pickle=True).item()

    def _pick_array(keys: list[str], field_name: str) -> np.ndarray:
        for k in keys:
            if k in data:
                return data[k]
        raise KeyError(f"{field_name} (expected one of: {keys})")

    # action: 根据 action_type 选取不同的 action 数据
    if action_type == "joint_delta":
        action = _pick_array(
            ["obs_policy_joint_delta_actions"], "joint_delta_actions"
        ).astype(np.float32)
    else:
        action = _pick_array(
            ["actions", "obs_policy_actions", "obs_policy_lastactions"], "actions"
        ).astype(np.float32)

    # qpos / qvel（兼容不同字段命名）
    qpos = _pick_array(["obs_policy_qpos", "qpos"], "qpos").astype(np.float32)
    qvel = _pick_array(["obs_policy_qvel", "qvel"], "qvel").astype(np.float32)

    # images — front (side camera) and wrist
    front_flat = _pick_array(["obs_policy_front", "front", "images_front"], "front image")
    wrist_flat = _pick_array(["obs_policy_wrist", "wrist", "images_wrist"], "wrist image")
    front = reshape_and_resize_images(front_flat, NPY_IMG_SHAPE, TARGET_IMG_SIZE)
    wrist = reshape_and_resize_images(wrist_flat, NPY_IMG_SHAPE, TARGET_IMG_SIZE)

    # 确保时间步一致（取最小值）
    T = min(action.shape[0], qpos.shape[0], qvel.shape[0], front.shape[0], wrist.shape[0])
    return {
        "action": action[:T],
        "observations": {
            "images": {
                "front": front[:T],
                "wrist": wrist[:T],
            },
            "qpos": qpos[:T],
            "qvel": qvel[:T],
        },
    }


def write_hdf5(episode_data: dict, out_path: str):
    """将单个 episode 数据写入一个 HDF5 文件。"""

    def _write_recursive(group: h5py.Group, d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                g = group.require_group(k)
                _write_recursive(g, v)
            else:
                group.create_dataset(k, data=v, compression="gzip")

    with h5py.File(out_path, "w") as f:
        _write_recursive(f, episode_data)
    print(f"  -> 已保存: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="将 episode npy 文件转换为 HDF5 格式")
    parser.add_argument("--npy_folder", default=DEFAULT_NEW, help="包含 episode_XXXX.npy 的文件夹")
    parser.add_argument("--ref", default=DEFAULT_REF, help="参考 HDF5 文件路径")
    parser.add_argument("--out", default=DEFAULT_OUT, help="输出 HDF5 文件夹")
    parser.add_argument("--action_type", default="ee_delta", choices=["ee_delta", "joint_delta"],
                        help="Action type: ee_delta (7D) or joint_delta (8D)")
    args = parser.parse_args()

    # 读取参考结构
    ref_info = load_ref_structure(args.ref)
    print("参考 HDF5 结构:")
    for k, v in ref_info.items():
        print(f"  {k}: shape={v['shape']}, dtype={v['dtype']}")

    # 收集 episode 文件
    episode_files = collect_episode_files(args.npy_folder)
    if not episode_files:
        print(f"未找到 episode 文件: {args.npy_folder}")
        return
    print(f"\n找到 {len(episode_files)} 个 episode 文件")
    print(f"Action type: {args.action_type}")

    # 创建输出目录
    os.makedirs(args.out, exist_ok=True)

    # 逐个转换
    for i, ep_file in enumerate(episode_files):
        out_path = os.path.join(args.out, f"episode_{i}.hdf5")
        print(f"\n处理: {ep_file} -> 输出为: episode_{i}.hdf5")
        try:
            ep_data = convert_episode(ep_file, action_type=args.action_type)
            # 打印数据形状
            print(f"  action: {ep_data['action'].shape}")
            print(f"  front:  {ep_data['observations']['images']['front'].shape}")
            print(f"  wrist:  {ep_data['observations']['images']['wrist'].shape}")
            print(f"  qpos:   {ep_data['observations']['qpos'].shape}")
            print(f"  qvel:   {ep_data['observations']['qvel'].shape}")
            write_hdf5(ep_data, out_path)
        except Exception as e:
            print(f"  !! 转换失败: {e}")


if __name__ == "__main__":
    main()

