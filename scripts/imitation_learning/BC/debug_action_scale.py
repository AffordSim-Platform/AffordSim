"""检查原始action scale和累计位移"""
import json
import numpy as np
import h5py

def minmax_denormalize(data, dmin, dmax):
    drange = dmax - dmin
    return (data + 1.0) / 2.0 * drange + dmin

demo_path = "/home/lmy/code/Real2Sim2Real-Pipeline-auto_task/dataset/r2s2r_data/task_gen/pick_place/pick_banana_and_place_on_plate/hdf5/merged_trajectories_robomimic_guass.hdf5"
stat_path = "/home/lmy/code/Real2Sim2Real-Pipeline-auto_task/dataset/r2s2r_data/task_gen/pick_place/pick_banana_and_place_on_plate/hdf5/dataset_stat_guass.json"

with open(stat_path) as f:
    stat = json.load(f)
minmax = stat["_minmax_params"]["action"]
amin = np.array(minmax["min"][:6], dtype=np.float32)
amax = np.array(minmax["max"][:6], dtype=np.float32)

data = h5py.File(demo_path, "r")
demo = data["data/demo_0"]
actions_norm = np.array(demo["actions"])  # (N, 7), normalized to [-1,1]

print(f"Trajectory length: {len(actions_norm)}")
print(f"Action min/max (real): {amin} / {amax}")
print(f"Action range: {amax - amin}")
print()

# Denormalize
ee_norm = actions_norm[:, :6]
ee_real = minmax_denormalize(ee_norm, amin, amax)

print("First 5 steps (normalized):")
for i in range(5):
    print(f"  [{i}] {np.round(ee_norm[i], 4)}")

print("\nFirst 5 steps (real/denormalized):")
for i in range(5):
    print(f"  [{i}] {np.round(ee_real[i], 6)}")

print(f"\nCumulative displacement (real):")
cum = np.cumsum(ee_real, axis=0)
print(f"  After 10 steps:  {np.round(cum[9], 4)}")
print(f"  After 44 steps:  {np.round(cum[43], 4)}")
print(f"  After all steps: {np.round(cum[-1], 4)}")

print(f"\nPer-step stats (real):")
print(f"  mean abs: {np.round(np.abs(ee_real).mean(axis=0), 6)}")
print(f"  max abs:  {np.round(np.abs(ee_real).max(axis=0), 6)}")

data.close()
