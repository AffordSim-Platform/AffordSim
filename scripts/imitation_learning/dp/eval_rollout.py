"""
训练期评估子进程 (由 dp_rss/train.py 在每次 save ckpt 后拉起).

- 启动 Isaac Lab, 加载 ckpt (优先 EMA 权重), 跑 N 次 rollout
- task yaml 的 primitives 作为评估阶段, 每个 episode 结束按阶段独立判成功
- 结果写 JSON 给 train.py 读回写 tensorboard

用法 (一般由 train.py 自动调用):
    python scripts/imitation_learning/dp_rss/eval_rollout.py \
        --checkpoint /path/to/model_epoch_100.pth \
        --stat /path/to/dataset_stat.json \
        --task_config /path/to/task.yaml \
        --task Pick_Place_banana-v0 \
        --num_rollouts 5 --horizon 300 \
        --output_json /tmp/eval_epoch_100.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="dp_rss Eval Rollout (subprocess)")
parser.add_argument("--task", type=str, default="Pick_Place_banana-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--stat", type=str, default=None)
parser.add_argument("--task_config", type=str, required=True)
parser.add_argument("--horizon", type=int, default=300)
parser.add_argument("--num_rollouts", type=int, default=5)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output_json", type=str, required=True)
parser.add_argument("--env_cfg_class", type=str, default=None,
                    help="'pkg.mod:ClassName', 不传则从 gym registry 解析")
parser.add_argument("--pick_lift_threshold", type=float, default=0.05)
parser.add_argument(
    "--warmup_action", type=str, default=None, choices=["zero", "hold"],
    help="reset 后预跑 10 步的动作模式. 不传则按 action_type 推断: "
         "ee_delta→zero (delta=0 不动), joint_delta→hold (绝对模式下 zero 会飞向零位).",
)
parser.add_argument(
    "--joint_delta_mode", type=str, default="cumulative", choices=["current", "cumulative"],
    help=(
        "joint_delta -> 绝对位置的展开方式. "
        "'current'=arm_target(k)=cur_qpos(k)+delta(k); "
        "'cumulative'=arm_target(k)=arm_target(k-1)+delta(k), 初值=首帧 arm qpos."
    ),
)
parser.add_argument("--use_main_weights", action="store_true",
                    help="默认加载 EMA 权重; 传此 flag 用非 EMA 的 main 权重")
parser.add_argument(
    "--action_horizon", type=int, default=None,
    help="开环执行步数; 不传则用 ckpt 里训练时的值. 调小可减少累积偏差但增加推理频率.",
)
parser.add_argument(
    "--print_action_every", type=int, default=0,
    help="每 N 步打印一次下发给 env 的 action; 0 关闭 (评测默认关).",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- 以下在 Isaac Sim 启动后 import ---
import gymnasium as gym
import importlib
import numpy as np
import random
import torch
import yaml

import isaaclab_tasks.real2sim.tasks  # noqa: F401

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from rollout_helpers import (
    load_norm_params,
    load_background_images,
    load_eval_stages,
    resolve_pick_target,
    evaluate_stages_at_episode_end,
    frames_to_gif,
    build_warmup_action,
    rollout_until_horizon,
    load_policy_from_ckpt,
)

from isaaclab.motion_collect.success.scene_helpers import attach_scene_success_meta
from isaaclab.motion_collect.success.success import capture_object_initial_z


def _import_env_cfg_class(spec: str):
    if ":" not in spec:
        raise ValueError(f"--env_cfg_class 必须是 'pkg.mod:ClassName', got: {spec}")
    mod_path, cls_name = spec.split(":", 1)
    return getattr(importlib.import_module(mod_path), cls_name)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args_cli.output_json = os.path.abspath(args_cli.output_json)
    print(f"[EVAL] device={device}, ckpt={args_cli.checkpoint}")
    print(f"[EVAL] output_json={args_cli.output_json}")

    # --- 模型 ---
    policy, model_args, _ = load_policy_from_ckpt(
        args_cli.checkpoint, device, prefer_ema=(not args_cli.use_main_weights),
    )
    ckpt_action_horizon = model_args.get("action_horizon", 8)
    action_horizon = args_cli.action_horizon if args_cli.action_horizon is not None else ckpt_action_horizon
    obs_horizon = model_args.get("obs_horizon", 2)
    action_type = model_args.get("action_type", "ee_delta")
    # use_qvel: 必须和训练时一致, 直接从 ckpt 读. 缺该字段的旧 ckpt 默认 True (老行为).
    use_qvel = bool(model_args.get("use_qvel", True))
    print(f"[EVAL] use_qvel={use_qvel}, lowdim_dim={model_args.get('lowdim_dim', '?')}")

    # warmup_action 必须配 action_type, 否则 joint_delta 用 zero 会把机械臂命令到全零关节位形.
    _WARMUP_BY_TYPE = {"ee_delta": "zero", "joint_delta": "hold"}
    if args_cli.warmup_action is None:
        args_cli.warmup_action = _WARMUP_BY_TYPE.get(action_type, "zero")
        print(f"[EVAL] warmup_action 自动推断为 '{args_cli.warmup_action}' (action_type={action_type})")
    elif action_type == "joint_delta" and args_cli.warmup_action == "zero":
        print(f"[EVAL] WARN: action_type={action_type} 但 warmup_action=zero, 起步会飞向零位关节; 推荐 'hold'")

    if args_cli.action_horizon is not None and args_cli.action_horizon != ckpt_action_horizon:
        print(f"[EVAL] action_horizon override: ckpt={ckpt_action_horizon} -> use {action_horizon}")
    print(f"[EVAL] obs_horizon={obs_horizon}, action_horizon={action_horizon}, action_type={action_type}")

    # --- env ---
    if args_cli.env_cfg_class:
        env_cfg_cls = _import_env_cfg_class(args_cli.env_cfg_class)
    else:
        entry = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]
        env_cfg_cls = _import_env_cfg_class(entry)
        print(f"[EVAL] task={args_cli.task} -> env_cfg_class={env_cfg_cls.__name__}")

    from isaaclab_tasks.real2sim.tasks.pick_r2s_env_cfg import create_pick_env_cfg_from_yaml
    env_cfg = create_pick_env_cfg_from_yaml(
        yaml_path=args_cli.task_config,
        env_cfg_class=env_cfg_cls,
        num_envs=1,
        device=args_cli.device,
    )
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.terminations.time_out = None
    if hasattr(env_cfg.terminations, "object_dropping"):
        env_cfg.terminations.object_dropping = None
    env_cfg.recorders = None
    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    with open(args_cli.task_config) as f:
        task_yaml = yaml.safe_load(f) or {}
    attach_scene_success_meta(env.scene, task_yaml)

    stages = load_eval_stages(args_cli.task_config)
    pick_target = resolve_pick_target(stages)
    pour_target = next((s.get("pour_target") for s in stages if s["type"] == "pour"), None)
    pull_target = next((s.get("pull_target") for s in stages if s["type"] == "pull"), None)
    print(
        f"[EVAL] stages={[s['type'] for s in stages]}, "
        f"pick={pick_target}, pour={pour_target}, pull={pull_target}"
    )

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)

    norm_params = load_norm_params(args_cli.stat) if args_cli.stat else None
    bg_images = load_background_images(args_cli.task_config)

    # --- rollouts ---
    per_rollout: list = []
    gif_dir = os.path.join(os.path.dirname(args_cli.output_json), "gifs")
    os.makedirs(gif_dir, exist_ok=True)
    gif_paths: list = []

    for trial in range(args_cli.num_rollouts):
        print(f"[EVAL] trial {trial + 1}/{args_cli.num_rollouts}")
        initial_obs_dict, _ = env.reset()
        for _ in range(10):
            warm = build_warmup_action(args_cli.warmup_action, env, initial_obs_dict, device)
            initial_obs_dict, _, _, _, _ = env.step(warm)
        pick_initial_z = capture_object_initial_z(env.scene, pick_target, env_idx=0) if pick_target else None

        res = rollout_until_horizon(
            policy=policy, env=env,
            horizon=args_cli.horizon,
            action_horizon=action_horizon,
            obs_horizon=obs_horizon,
            device=device,
            norm_params=norm_params,
            bg_images=bg_images,
            initial_obs_dict=initial_obs_dict,
            pick_target=pick_target,
            pour_target=pour_target,
            pull_target=pull_target,
            collect_front_images=True,
            stages=stages,
            pick_initial_z=pick_initial_z,
            pick_lift_threshold=args_cli.pick_lift_threshold,
            warmup_action=args_cli.warmup_action,
            action_type=action_type,
            joint_delta_mode=args_cli.joint_delta_mode,
            print_action_every=args_cli.print_action_every,
            print_action_tag=f"trial{trial}",
            use_qvel=use_qvel,
        )

        gif_path = os.path.join(gif_dir, f"trial_{trial}.gif")
        saved = frames_to_gif(res["front_frames"], gif_path, target_frames=30, target_size=256)
        if saved:
            gif_paths.append(saved)

        stage_results = evaluate_stages_at_episode_end(
            scene=env.scene, stages=stages,
            pick_initial_z=pick_initial_z,
            pick_lift_threshold=args_cli.pick_lift_threshold,
            pick_peak_z=res["pick_peak_z"],
            min_pour_tilt_z=res["min_pour_tilt_z"],
            pull_peak_displacement=res["pull_peak_displacement"],
        )
        per_rollout.append(stage_results)
        summary = ", ".join(
            f"{k}={'OK' if v.get('success') else 'FAIL'}" for k, v in stage_results.items()
        )
        print(f"[EVAL] trial {trial}: {summary}")

    # --- aggregate ---
    stage_keys: list = []
    for r in per_rollout:
        for k in r.keys():
            if k not in stage_keys:
                stage_keys.append(k)

    stage_rate: dict = {}
    for key in stage_keys:
        succ = sum(1 for r in per_rollout if r.get(key, {}).get("success"))
        stage_rate[key] = succ / max(len(per_rollout), 1)

    overall_cnt = sum(
        1 for r in per_rollout
        if stage_keys and all(r.get(k, {}).get("success") for k in stage_keys)
    )
    overall_rate = overall_cnt / max(len(per_rollout), 1)

    out = {
        "checkpoint": args_cli.checkpoint,
        "num_rollouts": args_cli.num_rollouts,
        "stage_success_rate": stage_rate,
        "overall_success_rate": overall_rate,
        "per_rollout": per_rollout,
        "gif_paths": gif_paths,
    }
    Path(os.path.dirname(args_cli.output_json)).mkdir(parents=True, exist_ok=True)
    with open(args_cli.output_json, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[EVAL] wrote {args_cli.output_json}")
    print(f"[EVAL] stage={stage_rate}, overall={overall_rate:.2%}")
    env.close()


if __name__ == "__main__":
    import traceback
    exit_code = 0
    try:
        main()
    except SystemExit as e:
        exit_code = int(e.code) if isinstance(e.code, int) else 1
    except BaseException:
        exit_code = 1
        print("[EVAL] ===== UNCAUGHT EXCEPTION =====")
        traceback.print_exc()
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass
    sys.exit(exit_code)
