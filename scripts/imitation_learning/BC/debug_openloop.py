"""
Open-loop replay: 从HDF5取action直接在环境里回放, 不经过模型.
验证反归一化后的action在环境里是否能正常执行.

用法:
  python scripts/imitation_learning/BC/debug_openloop.py \
    --task FrankaFr3-OurGripper-IKRel-VisPick-v0 \
    --task_config generated_task_config.yaml \
    --demo_path <hdf5> \
    --dataset_stat <stat.json>
"""
import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--task_config", type=str, default=None)
parser.add_argument("--demo_path", type=str, required=True)
parser.add_argument("--dataset_stat", type=str, required=True)
parser.add_argument("--demo_id", type=int, default=0)
parser.add_argument("--disable_fabric", action="store_true", default=False)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import gymnasium as gym
import h5py
import numpy as np
import torch

import isaaclab_tasks.real2sim.tasks.task_gen  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def minmax_denormalize(data, dmin, dmax):
    drange = np.array(dmax) - np.array(dmin)
    return (data + 1.0) / 2.0 * drange + np.array(dmin)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset stat
    with open(args_cli.dataset_stat) as f:
        stat = json.load(f)
    minmax = stat["_minmax_params"]

    # Load demo actions
    data = h5py.File(args_cli.demo_path, "r")
    demo_key = f"demo_{args_cli.demo_id}"
    actions_norm = np.array(data[f"data/{demo_key}/actions"])  # (N, 7) normalized
    data.close()

    print(f"Loaded {demo_key}: {len(actions_norm)} steps")

    # Denormalize actions
    ee_real = minmax_denormalize(
        actions_norm[:, :6],
        minmax["action"]["min"][:6],
        minmax["action"]["max"][:6],
    )
    gripper = actions_norm[:, -1]  # already -1/+1
    actions_real = np.concatenate([ee_real, gripper[:, None]], axis=-1).astype(np.float32)

    print(f"Action range after denorm: ee min={ee_real.min(0).round(4)}, max={ee_real.max(0).round(4)}")
    print(f"First 3 actions: {actions_real[:3].round(5)}")

    # Create environment
    if args_cli.task_config is not None:
        from isaaclab_tasks.real2sim.tasks.task_gen.gen_r2s_env_cfg import create_gen_env_cfg_from_yaml
        from isaaclab_tasks.real2sim.tasks.task_gen.config.sceneconfig.franka_fr3.ourgripper.ik_rel_vis_env_cfg import (
            FrankaFr3OurGripperIKRelVisGenEnvCfg,
        )
        env_cfg = create_gen_env_cfg_from_yaml(
            yaml_path=args_cli.task_config,
            env_cfg_class=FrankaFr3OurGripperIKRelVisGenEnvCfg,
            num_envs=1,
            device=args_cli.device,
        )
    else:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1,
                                use_fabric=not args_cli.disable_fabric)

    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.terminations.time_out = None
    if hasattr(env_cfg.terminations, "object_dropping"):
        env_cfg.terminations.object_dropping = None
    env_cfg.recorders = None
    success_term = getattr(env_cfg.terminations, "success", None)
    if success_term is not None:
        env_cfg.terminations.success = None

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    obs_dict, _ = env.reset()

    # Replay actions
    for i in range(len(actions_real)):
        action = torch.from_numpy(actions_real[i]).to(device).view(1, -1)

        if i <= 5 or i % 1 == 0:
            print(f"[Step {i}] action={actions_real[i].round(5)}")

        obs_dict, _, terminated, truncated, _ = env.step(action)

        if success_term is not None and bool(success_term.func(env, **success_term.params)[0]):
            print(f"SUCCESS at step {i}!")
            break
        if terminated or truncated:
            print(f"Episode ended at step {i}")
            break
    else:
        print(f"Finished {len(actions_real)} steps without success")

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
