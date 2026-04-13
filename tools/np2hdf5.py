import argparse
import os
import re
import glob

import cv2
import h5py
import numpy as np

DEFAULT_NEW = "data/r2s2r_data/pick_banana_and_place_in_plate_franka_fr3/20260331_015023/20260331_015128"
DEFAULT_REF = "data/reference.hdf5"
DEFAULT_OUT = "data/r2s2r_data/task_gen/pick_place/pick_banana_and_place_on_plate/hdf5"


NPY_IMG_SHAPE = (3, 480, 640)

TARGET_IMG_SIZE = (256, 256)


def collect_episode_files(folder: str) -> list[str]:
    
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
  
    info = {}
    with h5py.File(ref_path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                info[name] = {"shape": obj.shape, "dtype": obj.dtype}
        f.visititems(visit)
    return info


def reshape_and_resize_images(flat: np.ndarray, src_shape: tuple, target_size: tuple) -> np.ndarray:
    
    T = flat.shape[0]
    imgs = flat.reshape(T, *src_shape)

    if imgs.ndim == 4 and imgs.shape[1] == 3:
        imgs = imgs.transpose(0, 2, 3, 1)

    if imgs.dtype in (np.float32, np.float64):
        if imgs.max() <= 1.0:
            imgs = (imgs * 255).clip(0, 255).astype(np.uint8)
        else:
            imgs = imgs.clip(0, 255).astype(np.uint8)

    tH, tW = target_size
    H, W = imgs.shape[1], imgs.shape[2]
    if H != tH or W != tW:
  
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


def convert_episode(npy_path: str) -> dict:

    data = np.load(npy_path, allow_pickle=True).item()

    def _pick_array(keys: list[str], field_name: str) -> np.ndarray:
        for k in keys:
            if k in data:
                return data[k]
        raise KeyError(f"{field_name} (expected one of: {keys})")


    action = _pick_array(["actions", "obs_policy_actions", "obs_policy_lastactions"], "actions").astype(np.float32)

  
    qpos = _pick_array(["obs_policy_q_pos", "qpos"], "qpos").astype(np.float32)
    qvel = _pick_array(["obs_policy_q_vel", "qvel"], "qvel").astype(np.float32)


    front_flat = _pick_array(["obs_policy_external_cam_side", "front", "images_front"], "front image")
    wrist_flat = _pick_array(["obs_policy_wrist_cam", "wrist", "images_wrist"], "wrist image")
    front = reshape_and_resize_images(front_flat, NPY_IMG_SHAPE, TARGET_IMG_SIZE)
    wrist = reshape_and_resize_images(wrist_flat, NPY_IMG_SHAPE, TARGET_IMG_SIZE)

   
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


    def _write_recursive(group: h5py.Group, d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                g = group.require_group(k)
                _write_recursive(g, v)
            else:
                group.create_dataset(k, data=v, compression="gzip")

    with h5py.File(out_path, "w") as f:
        _write_recursive(f, episode_data)
    print(f"  -: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--npy_folder", default=DEFAULT_NEW, help="")
    parser.add_argument("--ref", default=DEFAULT_REF, help="")
    parser.add_argument("--out", default=DEFAULT_OUT, help="")
    args = parser.parse_args()

 
    ref_info = load_ref_structure(args.ref)
  
    for k, v in ref_info.items():
        print(f"  {k}: shape={v['shape']}, dtype={v['dtype']}")

   
    episode_files = collect_episode_files(args.npy_folder)
   
 
    os.makedirs(args.out, exist_ok=True)


    for i, ep_file in enumerate(episode_files):
        # i = i+142  
        out_path = os.path.join(args.out, f"episode_{i}.hdf5")
        print(f"\n: {ep_file} -> : episode_{i}.hdf5")
        try:
            ep_data = convert_episode(ep_file)
   
            print(f"  action: {ep_data['action'].shape}")
            print(f"  front:  {ep_data['observations']['images']['front'].shape}")
            print(f"  wrist:  {ep_data['observations']['images']['wrist'].shape}")
            print(f"  qpos:   {ep_data['observations']['qpos'].shape}")
            print(f"  qvel:   {ep_data['observations']['qvel'].shape}")
            write_hdf5(ep_data, out_path)
        except Exception as e:
            print(f"  !! : {e}")


if __name__ == "__main__":
    main()
