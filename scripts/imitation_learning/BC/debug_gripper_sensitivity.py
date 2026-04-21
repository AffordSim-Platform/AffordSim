"""
测试 gripper head 对输入的敏感度:
1. 用训练数据 step 0 的输入, logit 应该是正的 (open)
2. 用训练数据 step 45 的输入, logit 应该是负的 (close)
3. 用 step 0 的图像 + step 45 的 state → 看 gripper 跟谁走
4. 用 step 45 的图像 + step 0 的 state → 看 gripper 跟谁走
"""
import os, sys, json
import numpy as np
import h5py
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_bc import BCPolicy

ckpt_path = sys.argv[1]
demo_path = sys.argv[2]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
ee_head_in = ckpt["policy"]["ee_head.0.weight"].shape[1]
state_dim = ee_head_in - 512
has_resnet = any("front_encoder.conv1.weight" in k for k in ckpt["policy"].keys())
encoder_type = "resnet18" if has_resnet else "plain"
print(f"Encoder: {encoder_type}")
policy = BCPolicy(state_dim=state_dim, ee_action_dim=6, encoder_type=encoder_type)
policy.load_state_dict(ckpt["policy"])
policy.to(device).eval()

# Load data
data = h5py.File(demo_path, "r")
demo = data["data/demo_0"]
front = np.array(demo["obs/front"])
wrist = np.array(demo["obs/wrist"])
qpos = np.array(demo["obs/qpos"])
qvel = np.array(demo["obs/qvel"])
actions = np.array(demo["actions"])
data.close()

# Prepare tensors for two frames: open (step 0) and close (step 45)
def make_input(f_idx, w_idx, s_idx):
    f = torch.from_numpy(front[f_idx].astype(np.float32) / 255.0).unsqueeze(0).to(device)
    w = torch.from_numpy(wrist[w_idx].astype(np.float32) / 255.0).unsqueeze(0).to(device)
    s = np.concatenate([qpos[s_idx], qvel[s_idx]]).astype(np.float32)
    s = torch.from_numpy(s).unsqueeze(0).to(device)
    return f, w, s

tests = [
    ("all_open (step 0)",           0,  0,  0),
    ("all_close (step 50)",         50, 50, 50),
    ("open_img + close_state",      0,  0,  50),
    ("close_img + open_state",      50, 50, 0),
    ("open_front + close_wrist + open_state", 0, 50, 0),
    ("close_front + open_wrist + close_state", 50, 0, 50),
    ("zeros_img + open_state",     -1, -1,  0),
    ("zeros_img + close_state",    -1, -1, 50),
]

for name, fi, wi, si in tests:
    if fi == -1:
        f = torch.zeros(1, 256, 256, 3).to(device)
        w = torch.zeros(1, 256, 256, 3).to(device)
        s = np.concatenate([qpos[si], qvel[si]]).astype(np.float32)
        s = torch.from_numpy(s).unsqueeze(0).to(device)
    else:
        f, w, s = make_input(fi, wi, si)

    with torch.no_grad():
        ee, logit = policy(f, w, s)
    prob = torch.sigmoid(logit).item()
    pred = "OPEN" if prob > 0.5 else "CLOSE"
    print(f"{name:45s} | logit={logit.item():+8.4f} prob={prob:.4f} → {pred}")
