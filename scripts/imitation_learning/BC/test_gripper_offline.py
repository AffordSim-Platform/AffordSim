"""
Offline test: run a single demo from training data through the model and check gripper logit/prob output.
Does not require an Isaac Lab environment.

Usage:
  python scripts/imitation_learning/BC/test_gripper_offline.py \
    --checkpoint runs/bc_hdf5merged_trajectories_1.hdf5_1775004902/checkpoints/iter_19600.pt \
    --demo_path dataset/r2s2r_data/task_gen/pick_place/pick_banana_and_place_on_plate_new/hdf5merged_trajectories_1.hdf5 \
    --demo_id 0
"""

import argparse
import sys
import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F

# Add the directory containing train_bc.py to path so we can import BCPolicy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_bc import BCPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--demo_path", type=str, required=True)
    parser.add_argument("--demo_id", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load model ----
    ckpt = torch.load(args.checkpoint, map_location=device)
    # Auto-detect encoder type
    has_resnet = any("front_encoder.conv1.weight" in k for k in ckpt["policy"].keys())
    encoder_type = "resnet18" if has_resnet else "plain"
    policy = BCPolicy(state_dim=14, ee_action_dim=6, visual_feat_dim=256, encoder_type=encoder_type).to(device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    print(f"Loaded checkpoint: {args.checkpoint} (encoder={encoder_type})")

    # ---- Load one demo ----
    f = h5py.File(args.demo_path, "r")
    demo_key = f"demo_{args.demo_id}"
    demo = f["data"][demo_key]
    actions = np.array(demo["actions"])       # (N, 7)
    qpos = np.array(demo["obs/qpos"])         # (N, 7)
    qvel = np.array(demo["obs/qvel"])         # (N, 7)
    front = np.array(demo["obs/front"])       # (N, 256, 256, 3) uint8
    wrist = np.array(demo["obs/wrist"])       # (N, 256, 256, 3) uint8
    f.close()

    N = len(actions)
    gripper_raw = actions[:, -1]              # +1=open, -1=close
    gripper_label = ((gripper_raw + 1.0) / 2.0)  # 1=open, 0=close (same as training)

    print(f"\nDemo {demo_key}: {N} steps")
    print(f"Gripper switch points: {np.where(np.abs(np.diff(gripper_raw)) > 0.5)[0]}")
    print(f"Gripper raw unique: {np.unique(gripper_raw)}")
    print()

    # ---- Run inference on every step ----
    print(f"{'Step':>5} {'GT_raw':>7} {'GT_label':>9} {'Logit':>10} {'Prob':>8} {'Pred':>6} {'Match':>6}")
    print("-" * 60)

    n_correct = 0
    n_close_correct = 0
    n_close_total = 0
    n_open_correct = 0
    n_open_total = 0

    with torch.no_grad():
        for i in range(N):
            # Prepare inputs (same preprocessing as training)
            front_i = torch.from_numpy(front[i].astype(np.float32) / 255.0).unsqueeze(0).to(device)  # (1,256,256,3)
            wrist_i = torch.from_numpy(wrist[i].astype(np.float32) / 255.0).unsqueeze(0).to(device)
            state_i = torch.from_numpy(
                np.concatenate([qpos[i], qvel[i]]).astype(np.float32)
            ).unsqueeze(0).to(device)  # (1, 14)

            ee_pred, gripper_logit = policy(front_i, wrist_i, state_i)
            prob = torch.sigmoid(gripper_logit).item()
            logit = gripper_logit.item()
            pred_label = 1.0 if prob > 0.5 else 0.0
            gt_label = gripper_label[i]
            match = "OK" if pred_label == gt_label else "MISS"

            if gt_label > 0.5:  # open
                n_open_total += 1
                if pred_label == gt_label:
                    n_open_correct += 1
            else:  # close
                n_close_total += 1
                if pred_label == gt_label:
                    n_close_correct += 1

            if pred_label == gt_label:
                n_correct += 1

            # Print around switch points or every 10 steps
            switch_pts = np.where(np.abs(np.diff(gripper_raw)) > 0.5)[0]
            near_switch = any(abs(i - sp) <= 5 for sp in switch_pts)
            if near_switch or i % 10 == 0 or match == "MISS":
                print(f"{i:5d} {gripper_raw[i]:>7.1f} {gt_label:>9.1f} {logit:>10.4f} {prob:>8.4f} {pred_label:>6.1f} {match:>6}")

    print(f"\n{'='*60}")
    print(f"Overall accuracy: {n_correct}/{N} = {n_correct/N*100:.1f}%")
    print(f"Open accuracy:    {n_open_correct}/{n_open_total} = {n_open_correct/max(n_open_total,1)*100:.1f}%")
    print(f"Close accuracy:   {n_close_correct}/{n_close_total} = {n_close_correct/max(n_close_total,1)*100:.1f}%")


if __name__ == "__main__":
    main()
