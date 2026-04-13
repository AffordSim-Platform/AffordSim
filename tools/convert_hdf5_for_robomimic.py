

import argparse
import h5py
import json
import numpy as np
import os


def get_active_mask(src, traj_key):
 
    action = np.array(src[traj_key]["action"])
    return np.abs(action[:, :6]).sum(axis=1) > 1e-8


def compute_minmax(src, traj_keys, key_path):
  
    all_data = []
    for traj_key in traj_keys:
        data = np.array(src[traj_key][key_path])
        mask = get_active_mask(src, traj_key)
        data = data[mask]
        if len(data) > 0:
            all_data.append(data)
    all_data = np.concatenate(all_data, axis=0)
    return all_data.min(axis=0), all_data.max(axis=0)


def minmax_normalize(data, dmin, dmax):
 
    drange = dmax - dmin
    drange = np.where(drange < 1e-8, 1.0, drange)  
    return 2.0 * (data - dmin) / drange - 1.0


def convert(src_path: str, dst_path: str, stat_path: str, remove_idle: bool = True, gripper_gaussian_weight: bool = True):
    print(f"[INFO] : {src_path}")
    src = h5py.File(src_path, "r")

    traj_keys = sorted(
        [k for k in src.keys() if k.startswith("traj_")],
        key=lambda x: int(x.split("_")[1]),
    )
    print(f"[INFO]  {len(traj_keys)} ")

    
    print("[INFO]  min/max .")
    action_min, action_max = compute_minmax(src, traj_keys, "action")
    qpos_min, qpos_max = compute_minmax(src, traj_keys, "observations/qpos")
    qvel_min, qvel_max = compute_minmax(src, traj_keys, "observations/qvel")

    print(f"  action  min: {action_min}")
    print(f"  action  max: {action_max}")
    print(f"  qpos    min: {qpos_min}")
    print(f"  qpos    max: {qpos_max}")
    print(f"  qvel    min: {qvel_min}")
    print(f"  qvel    max: {qvel_max}")

  
    dst = h5py.File(dst_path, "w")
    data_grp = dst.create_group("data")
    total_samples = 0

   
    all_norm_actions = []
    all_norm_qpos = []
    all_norm_qvel = []

    num_skipped = 0
    demo_idx = 0
    for i, traj_key in enumerate(traj_keys):
        traj = src[traj_key]

       
        action = np.array(traj["action"])

    
        if remove_idle:
            ee_action_norm = np.abs(action[:, :6]).sum(axis=1)
            active_mask = ee_action_norm > 1e-8
            num_idle = int((~active_mask).sum())
            if num_idle > 0:
                print(f"  [{traj_key}] Removing {num_idle}/{len(action)} idle steps")
                action = action[active_mask]

           
            if len(action) == 0:
                print(f"  [{traj_key}] All steps idle, skipping trajectory")
                num_skipped += 1
                continue
        else:
            active_mask = np.ones(len(action), dtype=bool)
            num_idle = 0

        demo_name = f"demo_{demo_idx}"
        demo_idx += 1
        demo_grp = data_grp.create_group(demo_name)
        norm_action = action.copy().astype(np.float32)
        norm_action[:, :-1] = minmax_normalize(
            action[:, :-1], action_min[:-1], action_max[:-1]
        ).astype(np.float32)
 
        demo_grp.create_dataset("actions", data=norm_action)
        all_norm_actions.append(norm_action)

      
        if gripper_gaussian_weight:
            gripper = action[:, -1]
            gripper_diff = np.abs(np.diff(gripper))
            switch_indices = np.where(gripper_diff > 0.5)[0]  
            gripper_weights = np.ones(len(action), dtype=np.float32)
            if len(switch_indices) > 0:
                sigma = 5.0 
                peak_weight = 10.0  #
                for si in switch_indices:
                    dists = np.arange(len(action)) - si
                    gaussian = np.exp(-0.5 * (dists / sigma) ** 2)
                    gripper_weights = np.maximum(gripper_weights, 1.0 + (peak_weight - 1.0) * gaussian)
        else:
            gripper_weights = np.ones(len(action), dtype=np.float32)
        demo_grp.create_dataset("gripper_weights", data=gripper_weights)

        # obs
        obs_grp = demo_grp.create_group("obs")

        qpos = np.array(traj["observations/qpos"])
        if num_idle > 0:
            qpos = qpos[active_mask]
        norm_qpos = minmax_normalize(qpos, qpos_min, qpos_max).astype(np.float32)
        obs_grp.create_dataset("qpos", data=norm_qpos)
        all_norm_qpos.append(norm_qpos)

        qvel = np.array(traj["observations/qvel"])
        if num_idle > 0:
            qvel = qvel[active_mask]
        norm_qvel = minmax_normalize(qvel, qvel_min, qvel_max).astype(np.float32)
        obs_grp.create_dataset("qvel", data=norm_qvel)
        all_norm_qvel.append(norm_qvel)

      
        front = np.array(traj["observations/images/front"])
        wrist = np.array(traj["observations/images/wrist"])
        if num_idle > 0:
            front = front[active_mask]
            wrist = wrist[active_mask]
        obs_grp.create_dataset("front", data=front, compression="gzip")
        obs_grp.create_dataset("wrist", data=wrist, compression="gzip")

        n = len(action)
        demo_grp.attrs["num_samples"] = n
        total_samples += n

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(traj_keys)}] {traj_key} -> {demo_name}, steps={n}")

    if num_skipped > 0:
        print(f"[INFO] Skipped {num_skipped} empty trajectories after idle removal")

    #  train/valid mask (80/20)
    np.random.seed(42)
    n_demos = demo_idx
    n_train = int(n_demos * 0.95)
    indices = np.random.permutation(n_demos)
    train_demos = [f"demo_{indices[j]}" for j in range(n_train)]
    valid_demos = [f"demo_{indices[j]}" for j in range(n_train, n_demos)]

    mask_grp = dst.create_group("mask")
    mask_grp.create_dataset("train", data=[s.encode() for s in train_demos])
    mask_grp.create_dataset("valid", data=[s.encode() for s in valid_demos])

    #  data attrs
    data_grp.attrs["total"] = total_samples
    env_args = {
        "env_name": "Pick_Place_banana-v0",
        "type": 2,
        "env_kwargs": {},
    }
    data_grp.attrs["env_args"] = json.dumps(env_args)

    src.close()
    dst.close()
    print(f"[INFO] HDF5 : {dst_path}")
    print(f"[INFO] total: {total_samples}, train: {len(train_demos)}, valid: {len(valid_demos)}")

    # ========== mean/std → dataset_stat.json ==========
    all_norm_actions = np.concatenate(all_norm_actions, axis=0)
    all_norm_qpos = np.concatenate(all_norm_qpos, axis=0)
    all_norm_qvel = np.concatenate(all_norm_qvel, axis=0)

    dataset_stat = {
        "action": {
            "mean": all_norm_actions.mean(axis=0).tolist(),
            "std": all_norm_actions.std(axis=0).tolist(),
            "min": all_norm_actions.min(axis=0).tolist(),
            "max": all_norm_actions.max(axis=0).tolist(),
        },
        "qpos": {
            "mean": all_norm_qpos.mean(axis=0).tolist(),
            "std": all_norm_qpos.std(axis=0).tolist(),
            "min": all_norm_qpos.min(axis=0).tolist(),
            "max": all_norm_qpos.max(axis=0).tolist(),
        },
        "qvel": {
            "mean": all_norm_qvel.mean(axis=0).tolist(),
            "std": all_norm_qvel.std(axis=0).tolist(),
            "min": all_norm_qvel.min(axis=0).tolist(),
            "max": all_norm_qvel.max(axis=0).tolist(),
        },
  
        "_minmax_params": {
            "action": {"min": action_min.tolist(), "max": action_max.tolist()},
            "qpos": {"min": qpos_min.tolist(), "max": qpos_max.tolist()},
            "qvel": {"min": qvel_min.tolist(), "max": qvel_max.tolist()},
        },
    }

    with open(stat_path, "w") as f:
        json.dump(dataset_stat, f, indent=2)
    print(f"[INFO] dataset_stat.json 保存: {stat_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help=" HDF5 path of the input dataset")
    parser.add_argument("--dst", type=str, default=None, help="output HDF5 path")
    parser.add_argument("--stat", type=str, default=None, help="dataset_stat.json output path")
    parser.add_argument("--no_remove_idle", action="store_true", help="do not remove idle steps where EE delta is 0")
    parser.add_argument("--no_gripper_gaussian_weight", action="store_true", help="do not use Gaussian weighting for gripper switch points")
    args = parser.parse_args()

    if args.dst is None:
        base, ext = os.path.splitext(args.src)
        args.dst = base + "_robomimic" + ext

    if args.stat is None:
        args.stat = os.path.join(os.path.dirname(args.dst), "dataset_stat.json")

    convert(
        args.src, args.dst, args.stat,
        remove_idle=not args.no_remove_idle,
        gripper_gaussian_weight=not args.no_gripper_gaussian_weight,
    )
