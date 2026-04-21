"""
Debug: 保存训练数据中的图像 和 play时preprocess后的图像, 以便对比.
用法:
  python scripts/imitation_learning/BC/debug_compare_images.py \
    --demo_path <path_to_hdf5> \
    --output_dir /tmp/bc_debug_images
"""
import argparse
import os

import cv2
import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="/tmp/bc_debug_images")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data = h5py.File(args.demo_path, "r")
    data_grp = data["data"]
    demo = data_grp["demo_0"]

    front = np.array(demo["obs/front"])  # (N, 256, 256, 3) uint8
    wrist = np.array(demo["obs/wrist"])

    print(f"front shape={front.shape}, dtype={front.dtype}, min={front.min()}, max={front.max()}")
    print(f"wrist shape={wrist.shape}, dtype={wrist.dtype}, min={wrist.min()}, max={wrist.max()}")

    actions = np.array(demo["actions"])
    gripper_raw = actions[:, -1]
    switch_indices = np.where(np.abs(np.diff(gripper_raw)) > 0.5)[0]
    print(f"gripper switch at: {switch_indices}")

    # Save sample frames
    indices = [0, 10, 20]
    if len(switch_indices) > 0:
        si = switch_indices[0]
        indices.extend([si - 2, si - 1, si, si + 1, si + 2])
    indices.extend([len(front) - 1])

    for idx in indices:
        if idx < 0 or idx >= len(front):
            continue
        g = gripper_raw[idx]
        tag = "open" if g > 0 else "close"

        # front image (as stored in hdf5 - this is what training sees after /255.0)
        f_img = front[idx]  # (256, 256, 3) uint8
        w_img = wrist[idx]

        cv2.imwrite(os.path.join(args.output_dir, f"train_front_{idx:04d}_{tag}.png"),
                     cv2.cvtColor(f_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(args.output_dir, f"train_wrist_{idx:04d}_{tag}.png"),
                     cv2.cvtColor(w_img, cv2.COLOR_RGB2BGR))

    data.close()
    print(f"\nSaved images to {args.output_dir}")
    print("Now run play_bc.py and compare with the images saved during play.")
    print("Add cv2.imwrite in play_bc.py preprocess_image to save play-time images.")


if __name__ == "__main__":
    main()
