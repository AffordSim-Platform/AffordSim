"""
Rollout helpers shared by dp_rss/play.py  dp_rss/eval_rollout.py.

 isaaclab.* / isaaclab_tasks.*  import  lazy import,
 AppLauncher  import ().
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import torch


# ===========================================================================
# dataset_stat  ( dataset )
# ===========================================================================

def load_norm_params(stat_path: str) -> dict:
    with open(stat_path) as f:
        stat = json.load(f)
    params = stat["_minmax_params"]
    return {
        key: {
            "min": np.asarray(params[key]["min"], dtype=np.float32),
            "max": np.asarray(params[key]["max"], dtype=np.float32),
        }
        for key in params
    }


def minmax_normalize(data, dmin, dmax):
    drange = dmax - dmin
    drange = np.where(drange < 1e-8, 1.0, drange)
    return 2.0 * (data - dmin) / drange - 1.0


def minmax_denormalize(data, dmin, dmax):
    drange = dmax - dmin
    return (data + 1.0) / 2.0 * drange + dmin


# ===========================================================================
#  + 
# ===========================================================================

def load_background_images(task_config_path: str) -> dict:
    import cv2
    import yaml

    with open(task_config_path) as f:
        cfg = yaml.safe_load(f)
    runtime = cfg.get("runtime", {}) or {}
    config_dir = os.path.dirname(os.path.abspath(task_config_path))

    bg_images: dict = {}
    for key, bg_key in [("front", "cam_side_background"), ("wrist", "cam_wrist_background")]:
        bg_path = runtime.get(bg_key)
        if not bg_path:
            continue
        if os.path.isabs(bg_path):
            full = bg_path
        else:
            full = os.path.abspath(bg_path)
            if not os.path.exists(full):
                full = os.path.normpath(os.path.join(config_dir, bg_path))
        img = cv2.imread(full)
        if img is not None:
            bg_images[key] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return bg_images


def process_image(image_tensor: torch.Tensor, seg_tensor=None, bg_image=None, target_size: int = 256):
    """RGB (C,H,W)  to    to  center crop to square  to  resize to target_size.

     (tools/np2hdf5.py) :
        center crop  to  uint8  to  cv2.resize(INTER_AREA)  to  /255 float.
     (3, target_size, target_size) in [0,1].
    """
    import cv2
    from isaaclab_tasks.real2sim.utils.EpisodeBuffer import replace_background_with_segmentation

    image = image_tensor.clone().float()
    src_device = image_tensor.device
    C, H, W = image.shape
    if bg_image is not None and seg_tensor is not None:
        seg = seg_tensor.cpu().numpy().reshape(4, H, W).transpose(1, 2, 0)
        rgb_np = image.cpu().numpy().transpose(1, 2, 0)
        composite, _ = replace_background_with_segmentation(rgb_np, seg, bg_image)
        image = torch.from_numpy(composite.transpose(2, 0, 1)).float()
    crop = min(H, W)
    y0 = (H - crop) // 2
    x0 = (W - crop) // 2
    image = image[:, y0:y0 + crop, x0:x0 + crop]
    if crop != target_size:
        #  np2hdf5:  uint8 HWC  to  cv2.INTER_AREA  to   float [0,1] CHW.
        #  INTER_AREA ()  bilinear ,  byte-identical.
        img_hwc = image.cpu().numpy().transpose(1, 2, 0)  # float [0,1] HWC
        img_u8 = (img_hwc * 255.0).round().clip(0, 255).astype(np.uint8)
        resized_u8 = cv2.resize(img_u8, (target_size, target_size), interpolation=cv2.INTER_AREA)
        resized_f = resized_u8.astype(np.float32) / 255.0
        image = torch.from_numpy(resized_f.transpose(2, 0, 1).copy()).to(src_device)
    return image


# ===========================================================================
# task yaml  to  
# ===========================================================================

def load_eval_stages(task_config_path: str) -> list[dict]:
    import yaml
    with open(task_config_path) as f:
        cfg = yaml.safe_load(f) or {}
    primitives = cfg.get("primitives") or []
    stages: list[dict] = []
    for entry in primitives:
        if not isinstance(entry, dict):
            continue
        ptype = str(entry.get("type", "")).lower()
        if ptype not in ("pick", "place", "pour", "hang", "pull", "open", "close", "push"):
            continue
        params = entry.get("params") or {}
        stages.append({
            "type": ptype,
            "target": params.get("target"),
            "place_target": params.get("target") if ptype == "place" else None,
            "place_mode": str(entry.get("mode", "")).lower() if ptype == "place" else None,
            "place_position": params.get("position") if ptype == "place" else None,
            "place_orientation": params.get("orientation") if ptype == "place" else None,
            "pour_target": params.get("target") if ptype == "pour" else None,
            "hang_container": params.get("target") if ptype == "hang" else None,
            "hang_container_mount_npz": params.get("container_mount_affordance_npz_path") if ptype == "hang" else None,
            "pull_target": params.get("target") if ptype == "pull" else None,
            "open_target": params.get("target") if ptype == "open" else None,
            "open_joint_name": params.get("joint_name") if ptype == "open" else None,
            "close_target": params.get("target") if ptype == "close" else None,
            "close_joint_name": params.get("joint_name") if ptype == "close" else None,
            "push_target_label": params.get("target") if ptype == "push" else None,
            "push_target_position": params.get("push_target") if ptype == "push" else None,
        })
    return stages


def resolve_pick_target(stages: list[dict]) -> Optional[str]:
    for s in stages:
        if s["type"] == "pick" and s.get("target"):
            return s["target"]
    return None


def evaluate_stages_at_episode_end(
    scene,
    stages: list[dict],
    pick_initial_z: Optional[float],
    pick_lift_threshold: float,
    pick_peak_z: Optional[float] = None,
    min_pour_tilt_z: Optional[float] = None,
    pull_peak_displacement: Optional[float] = None,
) -> dict:
    """ env  (env_idx=0). Lazy-import  isaaclab ."""
    from isaaclab.motion_collect.success.evaluators import (
        _evaluate_pick_success_with_metrics,
        _evaluate_place_pose_success_with_metrics,
    )
    try:
        from isaaclab.motion_collect.success.evaluators import _evaluate_place_on_success_with_metrics
        _place_on_fn = _evaluate_place_on_success_with_metrics
    except ImportError:
        from isaaclab.motion_collect.success.evaluators import _evaluate_place_success_with_metrics as _place_on_fn
    from isaaclab.motion_collect.success.evaluators import (
        _evaluate_pour_success_with_metrics,
        _evaluate_hang_success_with_metrics,
        _evaluate_pull_success_with_metrics,
        _evaluate_open_success_with_metrics,
        _evaluate_close_success_with_metrics,
        _evaluate_push_success_with_metrics,
    )

    pick_target = resolve_pick_target(stages)
    results: dict = {}

    for idx, stage in enumerate(stages):
        ptype = stage["type"]
        key = f"{idx:02d}_{ptype}"

        if ptype == "pick":
            if pick_initial_z is None or not pick_target:
                results[key] = {"success": False, "reason": "no_initial_z_or_target"}
                continue
            ok, metrics = _evaluate_pick_success_with_metrics(
                scene=scene, pick_target=pick_target,
                initial_z_world=pick_initial_z, env_idx=0,
                lift_threshold=pick_lift_threshold, peak_z_world=pick_peak_z,
            )
            results[key] = {"success": bool(ok), "metrics": metrics}

        elif ptype == "place":
            place_mode = stage.get("place_mode") or "on"
            if place_mode == "pose":
                target_pos = stage.get("place_position")
                target_quat = stage.get("place_orientation")
                if target_pos is None:
                    results[key] = {"success": False, "reason": "no_place_position"}
                    continue
                ok, metrics = _evaluate_place_pose_success_with_metrics(
                    scene=scene, pick_target=pick_target,
                    target_position_env=np.asarray(target_pos, dtype=float),
                    target_quat_wxyz=np.asarray(target_quat, dtype=float) if target_quat is not None else None,
                    env_idx=0,
                )
            else:
                place_target = stage.get("place_target")
                if not place_target or not pick_target:
                    results[key] = {"success": False, "reason": "no_place_or_pick_target"}
                    continue
                ok, metrics = _place_on_fn(
                    scene=scene, pick_target=pick_target, place_target=place_target, env_idx=0,
                )
            results[key] = {"success": bool(ok), "metrics": metrics}

        elif ptype == "pour":
            pour_target = stage.get("pour_target")
            if not pour_target or not pick_target:
                results[key] = {"success": False, "reason": "no_pour_or_pick_target"}
                continue
            ok, metrics = _evaluate_pour_success_with_metrics(
                scene=scene, pick_target=pick_target, pour_target=pour_target, env_idx=0,
                min_z_component_observed=min_pour_tilt_z,
            )
            results[key] = {"success": bool(ok), "metrics": metrics}

        elif ptype == "hang":
            hang_container = stage.get("hang_container")
            if not hang_container or not pick_target:
                results[key] = {"success": False, "reason": "no_hang_container_or_pick_target"}
                continue
            mount_npz = stage.get("hang_container_mount_npz")
            try:
                if not mount_npz:
                    import importlib
                    afford_mod = importlib.import_module("isaaclab.motion_collect.tools.affordance_resolver")
                    mount_npz = afford_mod.resolve_affordance_path(hang_container, "hang")
                if not mount_npz:
                    results[key] = {"success": False, "reason": "no_container_mount_affordance"}
                    continue
                npz = np.load(mount_npz, allow_pickle=True)
                if "centroids" in npz:
                    container_mount_local = np.asarray(npz["centroids"], dtype=np.float64)[0]
                elif "grasp_center" in npz:
                    container_mount_local = np.asarray(npz["grasp_center"], dtype=np.float64)
                else:
                    results[key] = {"success": False, "reason": "invalid_mount_npz_keys"}
                    continue
            except Exception as e:
                results[key] = {"success": False, "reason": f"mount_load_failed:{e}"}
                continue
            ok, metrics = _evaluate_hang_success_with_metrics(
                scene=scene, held_object_label=pick_target, container_label=hang_container,
                container_mount_local=container_mount_local, env_idx=0,
            )
            results[key] = {"success": bool(ok), "metrics": metrics}

        elif ptype == "pull":
            pull_target = stage.get("pull_target")
            if not pull_target:
                results[key] = {"success": False, "reason": "no_pull_target"}
                continue
            ok, metrics = _evaluate_pull_success_with_metrics(
                scene=scene, target_label=pull_target, env_idx=0,
                peak_displacement=pull_peak_displacement,
            )
            results[key] = {"success": bool(ok), "metrics": metrics}

        elif ptype == "open":
            open_target = stage.get("open_target")
            if not open_target:
                results[key] = {"success": False, "reason": "no_open_target"}
                continue
            ok, metrics = _evaluate_open_success_with_metrics(
                scene=scene, target_label=open_target,
                joint_name=stage.get("open_joint_name"), env_idx=0,
            )
            results[key] = {"success": bool(ok), "metrics": metrics}

        elif ptype == "close":
            close_target = stage.get("close_target")
            if not close_target:
                results[key] = {"success": False, "reason": "no_close_target"}
                continue
            ok, metrics = _evaluate_close_success_with_metrics(
                scene=scene, target_label=close_target,
                joint_name=stage.get("close_joint_name"), env_idx=0,
            )
            results[key] = {"success": bool(ok), "metrics": metrics}

        elif ptype == "push":
            push_label = stage.get("push_target_label")
            push_pos = stage.get("push_target_position")
            if not push_label or push_pos is None:
                results[key] = {"success": False, "reason": "no_push_target_or_position"}
                continue
            ok, metrics = _evaluate_push_success_with_metrics(
                scene=scene, push_target_label=push_label,
                push_target_position_env=np.asarray(push_pos, dtype=float), env_idx=0,
            )
            results[key] = {"success": bool(ok), "metrics": metrics}

    return results


# ===========================================================================
# GIF
# ===========================================================================

def frames_to_gif(frames: list, save_path: str, target_frames: int = 30, target_size: int = 256):
    if not frames:
        return None
    from PIL import Image

    n = len(frames)
    idxs = np.linspace(0, n - 1, target_frames, dtype=int) if n > target_frames else np.arange(n)
    pil = [Image.fromarray(frames[i]).resize((target_size, target_size), Image.BILINEAR) for i in idxs]
    duration = max(5000 // len(pil), 50)
    pil[0].save(save_path, save_all=True, append_images=pil[1:], duration=duration, loop=0)
    return save_path


# ===========================================================================
# Warmup action (delta  zero,  hold)
# ===========================================================================

def build_warmup_action(mode: str, env, obs_dict, device):
    D = env.action_space.shape[1]
    if mode == "zero":
        return torch.zeros(1, D, device=device)
    if mode == "hold":
        qpos = obs_dict["policy"]["qpos"].squeeze().detach().to(device=device, dtype=torch.float32)
        arm = qpos[:D - 1]
        gripper = torch.tensor([-1.0], device=device, dtype=torch.float32)
        return torch.cat([arm, gripper]).view(1, -1)
    raise ValueError(f"unknown warmup_action mode: {mode}")


# ===========================================================================
# Pick peak / pour tilt / pull displacement 
# ===========================================================================

def _track_peaks_and_pour(
    scene,
    pick_target: Optional[str],
    pour_target: Optional[str],
    pull_target: Optional[str],
    state: dict,
):
    """ state['pick_peak_z', 'min_pour_tilt_z', 'pull_peak_displacement']."""
    from isaaclab.motion_collect.success.scene_helpers import (
        _get_articulation_joint_pos,
        _get_object_pose_world_from_scene,
    )
    from isaaclab.motion_collect.tools.pose_utils import _quat_wxyz_to_matrix

    if pull_target:
        from isaaclab.motion_collect.success.constants import PULL_SUCCESS_JOINT_NAME
        jp = _get_articulation_joint_pos(scene, pull_target, PULL_SUCCESS_JOINT_NAME, 0)
        if jp is not None:
            disp = abs(float(jp))
            if state.get("pull_peak_displacement") is None or disp > state["pull_peak_displacement"]:
                state["pull_peak_displacement"] = disp

    if not pick_target:
        return

    pos_w, quat_w = _get_object_pose_world_from_scene(scene, pick_target, 0)
    if pos_w is None:
        return
    z = float(pos_w[2])
    if state.get("pick_peak_z") is None or z > state["pick_peak_z"]:
        state["pick_peak_z"] = z

    if pour_target and quat_w is not None:
        pour_pos_w, _ = _get_object_pose_world_from_scene(scene, pour_target, 0)
        if pour_pos_w is not None:
            xy = float(np.linalg.norm(pos_w[:2] - pour_pos_w[:2]))
            if xy <= 0.20:
                rot = _quat_wxyz_to_matrix(np.asarray(quat_w, dtype=float))
                tilt_z = float(rot[2, 2])
                if state.get("min_pour_tilt_z") is None or tilt_z < state["min_pour_tilt_z"]:
                    state["min_pour_tilt_z"] = tilt_z


# ===========================================================================
# Rollout  ( policy.predict_action)
# ===========================================================================

def _build_obs_window(
    obs: dict,
    obs_img_buffer: list,
    obs_low_buffer: list,
    obs_horizon: int,
    norm_params: Optional[dict],
    bg_images: Optional[dict],
    rgb_keys,
    seg_keys,
    env_cam_hw,
    train_img_size: int,
    device,
    use_qvel: bool = True,
):
    env_cam_h, env_cam_w = env_cam_hw
    images_processed = []
    for name in rgb_keys:
        if name in obs:
            img = obs[name].squeeze().float().reshape(3, env_cam_h, env_cam_w)
            seg = obs.get(seg_keys[name], None)
            if seg is not None:
                seg = seg.squeeze()
            bg = bg_images.get(name) if bg_images else None
            img = process_image(img, seg, bg, train_img_size)
            images_processed.append(img)
    obs_img = torch.stack(images_processed, dim=0).to(device)

    qpos = obs["qpos"].squeeze().cpu().numpy()
    if norm_params is not None:
        qpos = minmax_normalize(qpos, norm_params["qpos"]["min"], norm_params["qpos"]["max"])
    if use_qvel:
        qvel = obs["qvel"].squeeze().cpu().numpy()
        if norm_params is not None and "qvel" in norm_params:
            qvel = minmax_normalize(qvel, norm_params["qvel"]["min"], norm_params["qvel"]["max"])
        obs_low = torch.from_numpy(np.concatenate([qpos, qvel])).float().to(device)
    else:
        obs_low = torch.from_numpy(qpos).float().to(device)

    obs_img_buffer.append(obs_img)
    obs_low_buffer.append(obs_low)
    if len(obs_img_buffer) > obs_horizon:
        obs_img_buffer.pop(0)
        obs_low_buffer.pop(0)
    while len(obs_img_buffer) < obs_horizon:
        obs_img_buffer.insert(0, obs_img_buffer[0])
        obs_low_buffer.insert(0, obs_low_buffer[0])

    obs_images = torch.stack(obs_img_buffer, dim=0).unsqueeze(0)  # (1, T_o, V, 3, H, W)
    obs_lowdim = torch.stack(obs_low_buffer, dim=0).unsqueeze(0)
    return obs_images, obs_lowdim, images_processed


@torch.no_grad()
def rollout_until_horizon(
    policy,
    env,
    horizon: int,
    action_horizon: int,
    obs_horizon: int,
    device,
    norm_params: Optional[dict] = None,
    bg_images: Optional[dict] = None,
    initial_obs_dict=None,
    pick_target: Optional[str] = None,
    pour_target: Optional[str] = None,
    pull_target: Optional[str] = None,
    collect_front_images: bool = False,
    stages: Optional[list] = None,
    pick_initial_z: Optional[float] = None,
    pick_lift_threshold: float = 0.05,
    early_stop_check_every: int = 10,
    warmup_action: str = "zero",
    action_type: str = "ee_delta",
    joint_delta_mode: str = "cumulative",
    success_term=None,
    train_img_size: int = 256,
    env_cam_hw=(480, 640),
    print_action_every: int = 0,
    print_action_tag: str = "",
    use_qvel: bool = True,
):
    """ horizon  env terminate.  env success_term ().

    print_action_every: >0 ,  N  env  action (debug ); 0 .
    print_action_tag:    ( "trial0"),  trial .
    use_qvel:            qvel  obs_lowdim, .
    """
    policy.eval()

    if initial_obs_dict is None:
        obs_dict, _ = env.reset()
    else:
        obs_dict = initial_obs_dict

    obs_img_buffer, obs_low_buffer = [], []
    seg_keys = {"front": "semantic_segmentation_side", "wrist": "semantic_segmentation_wrist"}
    rgb_keys = ["front", "wrist"]

    state = {"pick_peak_z": None, "min_pour_tilt_z": None, "pull_peak_displacement": None}
    front_frames: list = []
    # joint_delta:  arm qpos,  delta  ( qpos),
    #  arm_target(k) = arm_target(k-1) + delta(k).
    accumulated_arm = None

    def _collect_front(obs_dict_):
        if not collect_front_images:
            return
        if "front" not in obs_dict_["policy"]:
            return
        H, W = env_cam_hw
        _fr = obs_dict_["policy"]["front"].squeeze().float().reshape(3, H, W)
        _seg = obs_dict_["policy"].get(seg_keys["front"], None)
        if _seg is not None:
            _seg = _seg.squeeze()
        _bg = bg_images.get("front") if bg_images else None
        _p = process_image(_fr, _seg, _bg, train_img_size)
        front_frames.append((_p.cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8))

    def _check_early_success():
        if stages is None:
            return False
        res = evaluate_stages_at_episode_end(
            scene=env.scene, stages=stages,
            pick_initial_z=pick_initial_z,
            pick_lift_threshold=pick_lift_threshold,
            pick_peak_z=state["pick_peak_z"],
            min_pour_tilt_z=state["min_pour_tilt_z"],
            pull_peak_displacement=state["pull_peak_displacement"],
        )
        return bool(res) and all(r.get("success") for r in res.values())

    step = 0
    early_success = False
    term_success = False

    while step < horizon and not early_success:
        obs = obs_dict["policy"]

        # warmup at step 0 ()
        if step == 0:
            warm = build_warmup_action(warmup_action, env, obs_dict, device)
            obs_dict, _, _, _, _ = env.step(warm)
            step += 1
            _collect_front(obs_dict)
            continue

        obs_images, obs_lowdim, _ = _build_obs_window(
            obs, obs_img_buffer, obs_low_buffer,
            obs_horizon, norm_params, bg_images,
            rgb_keys, seg_keys, env_cam_hw, train_img_size, device,
            use_qvel=use_qvel,
        )

        # --- : diffusion policy  predict_action ---
        action_chunk = policy.predict_action(obs_images, obs_lowdim).squeeze(0).cpu().numpy()

        terminated = False
        truncated = False
        for j in range(min(action_horizon, horizon - step)):
            action = action_chunk[j].astype(np.float32)

            if norm_params is not None and "action" in norm_params:
                action[:-1] = minmax_denormalize(
                    action[:-1],
                    norm_params["action"]["min"][:-1],
                    norm_params["action"]["max"][:-1],
                )
            action[-1] = 1.0 if action[-1] > 0.0 else -1.0

            # joint_delta: env  JointPositionActionCfg .  (--joint_delta_mode):
            #   "current"   : arm_target(k) = cur_qpos(k) + delta(k), 
            #   "cumulative": arm_target(k) = arm_target(k-1) + delta(k), = arm qpos
            if action_type == "joint_delta":
                if joint_delta_mode == "current":
                    cur_qpos = obs_dict["policy"]["qpos"].squeeze().detach().cpu().numpy().astype(action.dtype)
                    arm_abs = cur_qpos[:action.shape[0] - 1] + action[:-1]
                    action_out = np.concatenate([arm_abs, action[-1:]], axis=0)
                else:  # "cumulative"
                    if accumulated_arm is None:
                        cur_qpos = obs_dict["policy"]["qpos"].squeeze().detach().cpu().numpy().astype(action.dtype)
                        accumulated_arm = cur_qpos[:action.shape[0] - 1].copy()
                    accumulated_arm = accumulated_arm + action[:-1]
                    action_out = np.concatenate([accumulated_arm, action[-1:]], axis=0)
            else:
                action_out = action

            action_tensor = torch.from_numpy(action_out).float().to(device).view(1, -1)
            if print_action_every > 0 and step % print_action_every == 0:
                tag = f"[{print_action_tag}] " if print_action_tag else ""
                print(
                    f"{tag}step={step:04d} act_type={action_type} "
                    f"raw={np.round(action, 3).tolist()} "
                    f"out={np.round(action_out, 3).tolist()}"
                )
            obs_dict, _, terminated, truncated, _ = env.step(action_tensor)
            step += 1
            _track_peaks_and_pour(env.scene, pick_target, pour_target, pull_target, state)
            _collect_front(obs_dict)

            if success_term is not None and bool(success_term.func(env, **success_term.params)[0]):
                term_success = True
                break
            if terminated or truncated:
                break

            if step % early_stop_check_every == 0 and _check_early_success():
                early_success = True
                break

            if j < action_horizon - 1:
                obs = obs_dict["policy"]
                _build_obs_window(
                    obs, obs_img_buffer, obs_low_buffer,
                    obs_horizon, norm_params, bg_images,
                    rgb_keys, seg_keys, env_cam_hw, train_img_size, device,
                    use_qvel=use_qvel,
                )

        if terminated or truncated or early_success or term_success:
            break

    return {
        "obs_dict": obs_dict,
        "pick_peak_z": state["pick_peak_z"],
        "min_pour_tilt_z": state["min_pour_tilt_z"],
        "pull_peak_displacement": state["pull_peak_displacement"],
        "front_frames": front_frames,
        "term_success": term_success,
        "early_success": early_success,
        "step": step,
    }


# ===========================================================================
# Checkpoint :  EMA 
# ===========================================================================

def load_policy_from_ckpt(ckpt_path: str, device, prefer_ema: bool = True):
    """ ckpt  DiffusionPolicy  (EMA )."""
    import sys
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from policy import DiffusionPolicy

    # weights_only=False: PyTorch 2.6+  True  ckpt  numpy scalars (epoch );
    #  ckpt, .
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})

    policy = DiffusionPolicy(
        action_dim=args.get("action_dim", 7),
        obs_horizon=args.get("obs_horizon", 2),
        pred_horizon=args.get("pred_horizon", 16),
        lowdim_dim=args.get("lowdim_dim", 14),
        img_feature_dim=args.get("img_feature_dim", 128),
        lowdim_feature_dim=args.get("lowdim_feature_dim", 64),
        resize_shape=tuple(args.get("resize_shape", [240, 240])),
        crop_shape=tuple(args.get("crop_shape", [216, 216])),
        num_views=args.get("num_views", 2),
        pretrained_resnet=False,  #  ckpt  ImageNet 
        use_group_norm=True,
        image_aug=args.get("image_aug", True),
        down_dims=tuple(args.get("down_dims", [256, 512, 1024])),
        kernel_size=args.get("kernel_size", 5),
        diffusion_step_embed_dim=args.get("diffusion_step_embed_dim", 256),
        cond_predict_scale=True,
        num_train_timesteps=args.get("num_diffusion_steps", 100),
        beta_schedule=args.get("beta_schedule", "squaredcos_cap_v2"),
        prediction_type="epsilon",
        clip_sample=True,
        num_inference_steps=args.get("num_inference_steps", 16),
        inference_scheduler=args.get("inference_scheduler", "ddim"),
    ).to(device)

    state = None
    if prefer_ema and ckpt.get("ema_model") is not None:
        state = ckpt["ema_model"]
        print(f"[load_ckpt] using EMA weights (optimization_step={ckpt.get('ema_step', '?')})")
    elif "model" in ckpt:
        state = ckpt["model"]
        print("[load_ckpt] using main weights (no EMA found in ckpt)")
    else:
        raise ValueError(f"ckpt {ckpt_path} has neither 'ema_model' nor 'model'")

    policy.load_state_dict(state)
    policy.eval()
    return policy, args, ckpt
