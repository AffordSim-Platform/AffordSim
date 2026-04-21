"""
Debug: 从 HDF5 取训练数据, 直接过模型, 检查 gripper 输出.
用法:
  python scripts/imitation_learning/BC/debug_gripper.py \
    --checkpoint <path_to_ckpt.pt> \
    --demo_path <path_to_hdf5>
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_bc import BCPolicy, minmax_normalize

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--demo_path", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    norm_params = ckpt.get("norm_params", None)

    # Infer state_dim
    ee_head_in = ckpt["policy"]["ee_head.0.weight"].shape[1]
    state_dim = ee_head_in - 512
    print(f"state_dim={state_dim}")

    policy = BCPolicy(state_dim=state_dim, ee_action_dim=6)
    policy.load_state_dict(ckpt["policy"])
    policy.to(device)
    policy.eval()

    # Load a few frames from HDF5
    data = h5py.File(args.demo_path, "r")
    data_grp = data["data"]
    demo_keys = sorted([k for k in data_grp.keys() if k.startswith("demo_")],
                       key=lambda x: int(x.split("_")[1]))

    print(f"\nTotal demos: {len(demo_keys)}")

    # Check a few demos with different gripper states
    for demo_key in demo_keys[:5]:
        demo = data_grp[demo_key]
        actions = np.array(demo["actions"])
        qpos = np.array(demo["obs/qpos"])
        qvel = np.array(demo["obs/qvel"])
        front = np.array(demo["obs/front"])
        wrist = np.array(demo["obs/wrist"])

        gripper_raw = actions[:, -1]  # -1 or +1
        switch_indices = np.where(np.abs(np.diff(gripper_raw)) > 0.5)[0]
        print(f"\n{'='*60}")
        print(f"{demo_key}: {len(actions)} steps, gripper switches at: {switch_indices}")
        print(f"  gripper values: first={gripper_raw[0]}, last={gripper_raw[-1]}")

        # Test frames: first frame, around switch point, last frame
        test_indices = [0, len(actions)//2, len(actions)-1]
        if len(switch_indices) > 0:
            si = switch_indices[0]
            test_indices.extend([max(0, si-1), si, min(len(actions)-1, si+1)])
        test_indices = sorted(set(test_indices))

        for idx in test_indices:
            # Prepare input exactly like training
            front_img = front[idx].astype(np.float32) / 255.0  # (256,256,3)
            wrist_img = wrist[idx].astype(np.float32) / 255.0

            state = np.concatenate([qpos[idx], qvel[idx]], axis=-1).astype(np.float32)

            front_t = torch.from_numpy(front_img).unsqueeze(0).to(device)  # (1,256,256,3)
            wrist_t = torch.from_numpy(wrist_img).unsqueeze(0).to(device)
            state_t = torch.from_numpy(state).unsqueeze(0).to(device)

            with torch.no_grad():
                ee_delta, gripper_logit = policy(front_t, wrist_t, state_t)
                prob = torch.sigmoid(gripper_logit).item()
                logit = gripper_logit.item()
                ee_np = ee_delta.cpu().numpy().squeeze(0)

            gt_gripper = gripper_raw[idx]
            gt_label = (gt_gripper + 1.0) / 2.0  # -1→0, +1→1
            gt_ee = actions[idx, :6]
            pred_class = 1.0 if prob > 0.5 else 0.0

            correct = "✓" if pred_class == gt_label else "✗"
            print(f"  [{idx:4d}] gt_gripper={gt_gripper:+.0f} (label={gt_label:.0f}) | "
                  f"logit={logit:+8.4f} prob={prob:.4f} pred={pred_class:.0f} {correct}")
            print(f"         pred_ee={np.round(ee_np, 4)}")
            print(f"         gt_ee  ={np.round(gt_ee, 4)}")

    data.close()

if __name__ == "__main__":
    main()
