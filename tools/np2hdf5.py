import argparse
import os
import re
import glob
from functools import partial
from multiprocessing import Pool

import cv2
import h5py
import numpy as np

DEFAULT_NEW = "data/r2s2r_data/hang_mugcup_franka_fr3_default/20260618_170303/20260618_170311"
DEFAULT_REF = "data/reference.hdf5"
DEFAULT_OUT = "data/r2s2r_data/hang_mugcup_franka_fr3_default/20260618_170303/hdf5"


NPY_IMG_SHAPE = (3, 480, 640)

TARGET_IMG_SIZE = (480, 640)


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
        resized = np.empty((T, tH, tW, 3), dtype=np.uint8)
        for i in range(T):
            resized[i] = cv2.resize(imgs[i], (tW, tH), interpolation=cv2.INTER_AREA)
        return resized
    return imgs


def convert_episode(npy_path: str) -> dict:

    data = np.load(npy_path, allow_pickle=True).item()

    action = data["obs_policy_actions"].astype(np.float32)

    qpos = data["obs_policy_q_pos"].astype(np.float32)
    qvel = data["obs_policy_q_vel"].astype(np.float32)

    side_flat = data["obs_policy_external_cam_side"]
    front_flat = data["obs_policy_external_cam_front"]
    wrist_flat = data["obs_policy_wrist_cam"]
    side = reshape_and_resize_images(side_flat, NPY_IMG_SHAPE, TARGET_IMG_SIZE)
    front = reshape_and_resize_images(front_flat, NPY_IMG_SHAPE, TARGET_IMG_SIZE)
    wrist = reshape_and_resize_images(wrist_flat, NPY_IMG_SHAPE, TARGET_IMG_SIZE)


    T = min(action.shape[0], qpos.shape[0], qvel.shape[0], side.shape[0], front.shape[0], wrist.shape[0])
    return {
        "action": action[:T],
        "observations": {
            "images": {
                "side": side[:T],
                "front": front[:T],
                "wrist": wrist[:T],
            },
            "qpos": qpos[:T],
            "qvel": qvel[:T],
        },
    }


def _compression_kwargs(name: str) -> dict:
    """Map a compression name to h5py create_dataset kwargs."""
    if name == "none":
        return {}
    if name == "lzf":
        return {"compression": "lzf"}
    if name == "gzip-fast":
        return {"compression": "gzip", "compression_opts": 1}
    return {"compression": "gzip"}  # default gzip (level 4): smallest, slowest


def write_hdf5(episode_data: dict, out_path: str, comp_kwargs: dict = None):
    if comp_kwargs is None:
        comp_kwargs = {"compression": "gzip"}

    def _write_recursive(group: h5py.Group, d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                g = group.require_group(k)
                _write_recursive(g, v)
            else:
                group.create_dataset(k, data=v, **comp_kwargs)

    with h5py.File(out_path, "w") as f:
        _write_recursive(f, episode_data)


def _process_episode(task: tuple, out_dir: str, comp_kwargs: dict) -> tuple:
    """Convert one npy -> hdf5. Runs in a worker process. Returns (idx, ok, msg)."""
    idx, ep_file = task
    out_path = os.path.join(out_dir, f"episode_{idx}.hdf5")
    try:
        ep_data = convert_episode(ep_file)
        write_hdf5(ep_data, out_path, comp_kwargs)
        return (idx, True, os.path.basename(ep_file))
    except Exception as e:  # noqa: BLE001
        return (idx, False, f"{os.path.basename(ep_file)}: {e}")


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--npy_folder", default=DEFAULT_NEW, help="")
    parser.add_argument("--ref", default=DEFAULT_REF, help="")
    parser.add_argument("--out", default=DEFAULT_OUT, help="")
    parser.add_argument("--num_episodes", type=int, default=500,
                        help="Only convert the first N episodes. If fewer exist, convert all. Default: all.")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                        help="Number of parallel worker processes. 1 = serial.")
    parser.add_argument("--compression", default="gzip",
                        choices=["gzip", "gzip-fast", "lzf", "none"],
                        help="HDF5 compression: gzip (smallest), gzip-fast/lzf (faster), none (largest, fastest).")
    args = parser.parse_args()

    episode_files = collect_episode_files(args.npy_folder)

    # Keep only the first N episodes when requested; if fewer exist, keep all.
    if args.num_episodes is not None:
        if len(episode_files) < args.num_episodes:
            print(f"[WARN] requested {args.num_episodes} episodes but only {len(episode_files)} found; converting all.")
        episode_files = episode_files[:args.num_episodes]

    os.makedirs(args.out, exist_ok=True)
    comp_kwargs = _compression_kwargs(args.compression)

    print(f"Converting {len(episode_files)} episodes from {args.npy_folder} "
          f"(workers={args.workers}, compression={args.compression})")

    tasks = list(enumerate(episode_files))
    worker = partial(_process_episode, out_dir=args.out, comp_kwargs=comp_kwargs)

    n_ok = 0
    if args.workers <= 1:
        results = (worker(t) for t in tasks)
    else:
        pool = Pool(args.workers)
        # imap (not imap_unordered) keeps results in submission order for tidy logs;
        # throughput is effectively the same.
        results = pool.imap(worker, tasks)

    for idx, ok, msg in results:
        if ok:
            n_ok += 1
            print(f"  [{n_ok}/{len(tasks)}] episode_{idx}.hdf5  <- {msg}")
        else:
            print(f"  !! episode_{idx} FAILED: {msg}")

    if args.workers > 1:
        pool.close()
        pool.join()

    print(f"Done: {n_ok}/{len(tasks)} episodes -> {args.out}")


if __name__ == "__main__":
    main()
