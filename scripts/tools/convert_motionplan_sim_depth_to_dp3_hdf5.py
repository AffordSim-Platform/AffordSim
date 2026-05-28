#!/usr/bin/env python3
"""Convert AffordSim motion-plan simulator depth episodes to a DP3 HDF5.

The output schema matches the existing DP3 point-cloud datasets:

    obs/point_cloud: [T, N, 6] xyzrgb, rgb in [0, 1]
    obs/robot0_eye_in_hand_image: [T, H, W, 3]
    obs/robot0_eef_pos, obs/robot0_eef_quat, obs/robot0_gripper_qpos
    actions: absolute EE pose as xyz + rotvec + gripper command
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Optional

import cv2
import h5py
import numpy as np
from tqdm import tqdm


DEFAULT_INTRINSIC = (
    609.7103881835938,
    609.10546875,
    317.0648498535156,
    247.10394287109375,
)
DEFAULT_CAMERA_POS = (0.8370140563091792, -0.053834917559291456, 0.6196097154959959)
DEFAULT_CAMERA_QUAT_WXYZ = (
    0.183989617715562,
    -0.7135613706579924,
    -0.6642694460586073,
    0.125395749172607,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a DP3 robomimic-style HDF5 from AffordSim RGB + simulator depth episodes."
    )
    parser.add_argument("--input", required=True, type=Path, help="Directory containing episode_*.npy files.")
    parser.add_argument("--output", required=True, type=Path, help="Output DP3 HDF5 path.")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--skip-errors", action="store_true")
    parser.add_argument("--min-steps", type=int, default=10)

    parser.add_argument("--camera", default="external_cam_side")
    parser.add_argument("--rgb-key", default="obs_policy_external_cam_side")
    parser.add_argument("--depth-key", default="obs_policy_external_cam_side_distance_to_image_plane")
    parser.add_argument("--eef-pose-key", default="obs_policy_arm_ee_pose")
    parser.add_argument("--gripper-key", default="obs_policy_gripper_pos")

    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=84)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--min-depth", type=float, default=0.05)
    parser.add_argument("--max-depth", type=float, default=2.0)
    parser.add_argument("--z-min", type=float, default=0.02)
    parser.add_argument("--pointcloud-frame", choices=("world", "camera"), default="world")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compression", choices=("gzip", "none"), default="gzip")

    parser.add_argument("--fx", type=float, default=DEFAULT_INTRINSIC[0])
    parser.add_argument("--fy", type=float, default=DEFAULT_INTRINSIC[1])
    parser.add_argument("--cx", type=float, default=DEFAULT_INTRINSIC[2])
    parser.add_argument("--cy", type=float, default=DEFAULT_INTRINSIC[3])
    parser.add_argument("--camera-pos", type=float, nargs=3, default=DEFAULT_CAMERA_POS)
    parser.add_argument("--camera-quat-wxyz", type=float, nargs=4, default=DEFAULT_CAMERA_QUAT_WXYZ)
    return parser.parse_args()


def episode_id(path: Path) -> int:
    match = re.search(r"episode_(\d+)\.npy$", path.name)
    if not match:
        raise ValueError(f"Cannot parse episode id from {path}")
    return int(match.group(1))


def list_episode_files(root: Path, max_episodes: Optional[int]) -> list[Path]:
    files = sorted(root.glob("episode_*.npy"), key=episode_id)
    if max_episodes is not None:
        files = files[:max_episodes]
    if not files:
        raise FileNotFoundError(f"No episode_*.npy files under {root}")
    return files


def quat_wxyz_to_matrix(quat: Iterable[float]) -> np.ndarray:
    w, x, y, z = np.asarray(tuple(quat), dtype=np.float64)
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0:
        raise ValueError("Zero-norm quaternion.")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def quat_wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).reshape(4)
    return np.asarray([quat[1], quat[2], quat[3], quat[0]], dtype=np.float32)


def quat_wxyz_to_rotvec(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0:
        return np.zeros(3, dtype=np.float32)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    w = np.clip(w, -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    scale = np.sqrt(max(1.0 - w * w, 0.0))
    if scale < 1e-8:
        return np.zeros(3, dtype=np.float32)
    axis = np.asarray([x, y, z], dtype=np.float64) / scale
    return (axis * angle).astype(np.float32)


def image_from_flat(flat: np.ndarray) -> np.ndarray:
    flat = np.asarray(flat)
    if flat.ndim != 1:
        raise ValueError(f"Expected flat RGB image, got {flat.shape}")
    if flat.size != 640 * 480 * 3:
        side = int(round(np.sqrt(flat.size / 3.0)))
        if side * side * 3 != flat.size:
            raise ValueError(f"Cannot infer image shape from flat size {flat.size}")
        h, w = side, side
    else:
        h, w = 480, 640
    image = flat.reshape(3, h, w).transpose(1, 2, 0)
    if image.dtype.kind == "f" and image.max(initial=0) <= 1.0:
        image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def depth_from_flat(flat: np.ndarray) -> np.ndarray:
    flat = np.asarray(flat)
    if flat.ndim == 2:
        return flat.astype(np.float32)
    if flat.ndim != 1:
        raise ValueError(f"Expected flat depth image, got {flat.shape}")
    if flat.size == 640 * 480:
        return flat.reshape(480, 640).astype(np.float32)
    side = int(round(np.sqrt(flat.size)))
    if side * side != flat.size:
        raise ValueError(f"Cannot infer depth shape from flat size {flat.size}")
    return flat.reshape(side, side).astype(np.float32)


def load_rgb_sequence(raw_dir: Path, npy_data: dict, ep_id: int, camera: str, rgb_key: str) -> np.ndarray:
    image_dir = raw_dir / "original_images" / camera / f"env0_ep{ep_id}"
    if image_dir.is_dir():
        frames = sorted(image_dir.glob("step_*.png"))
        if frames:
            images = []
            for frame in frames:
                bgr = cv2.imread(str(frame), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError(f"Could not read image {frame}")
                images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            return np.stack(images, axis=0).astype(np.uint8)

    if rgb_key not in npy_data:
        raise KeyError(f"No {rgb_key} in npy and no original_images for episode {ep_id}")
    return np.stack([image_from_flat(frame) for frame in npy_data[rgb_key]], axis=0)


def depth_dir_candidates(raw_dir: Path, camera: str, depth_key: str, ep_id: int) -> list[Path]:
    short_key = depth_key.removeprefix("obs_policy_")
    return [
        raw_dir / "depth_images" / depth_key / f"env0_ep{ep_id}",
        raw_dir / "depth_images" / short_key / f"env0_ep{ep_id}",
        raw_dir / "depth_images" / f"{camera}_distance_to_image_plane" / f"env0_ep{ep_id}",
        raw_dir / "depth_images" / camera / f"env0_ep{ep_id}",
    ]


def load_depth_sequence(raw_dir: Path, npy_data: dict, ep_id: int, camera: str, depth_key: str) -> np.ndarray:
    for image_dir in depth_dir_candidates(raw_dir, camera, depth_key, ep_id):
        if not image_dir.is_dir():
            continue
        frames = sorted(image_dir.glob("step_*.png"))
        if not frames:
            continue
        depth_frames = []
        for frame in frames:
            depth_mm = cv2.imread(str(frame), cv2.IMREAD_UNCHANGED)
            if depth_mm is None:
                raise ValueError(f"Could not read depth image {frame}")
            depth_frames.append(depth_mm.astype(np.float32) / 1000.0)
        return np.stack(depth_frames, axis=0).astype(np.float32)

    if depth_key not in npy_data:
        raise KeyError(f"No {depth_key} in npy and no depth_images for episode {ep_id}")
    return np.stack([depth_from_flat(frame) for frame in npy_data[depth_key]], axis=0)


def depth_to_point_cloud(
    depth_m: np.ndarray,
    rgb: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    stride: int,
    frame: str,
    camera_rot: np.ndarray,
    camera_pos: np.ndarray,
    min_depth: float,
    max_depth: float,
    z_min: float,
) -> tuple[np.ndarray, np.ndarray]:
    depth = depth_m[::stride, ::stride]
    color = rgb[::stride, ::stride]
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    u *= stride
    v *= stride

    valid = np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
    z = depth[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    points = np.stack([x, y, z], axis=1).astype(np.float32)
    colors = color[valid].astype(np.uint8)

    if frame == "world":
        points = points @ camera_rot.T + camera_pos[None, :]
        keep = points[:, 2] >= z_min
        points = points[keep]
        colors = colors[keep]
    return points, colors


def sample_point_cloud(points: np.ndarray, colors: np.ndarray, num_points: int, rng: np.random.Generator) -> np.ndarray:
    if len(points) == 0:
        raise ValueError("Point cloud is empty.")
    idx = rng.choice(len(points), size=num_points, replace=len(points) < num_points)
    xyz = points[idx].astype(np.float32)
    rgb = colors[idx].astype(np.float32) / 255.0
    return np.concatenate([xyz, rgb], axis=1).astype(np.float32)


def build_episode(
    path: Path,
    raw_dir: Path,
    args: argparse.Namespace,
    camera_rot: np.ndarray,
    camera_pos: np.ndarray,
    demo_idx: int,
) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True).item()
    ep_id = episode_id(path)
    images = load_rgb_sequence(raw_dir, data, ep_id, args.camera, args.rgb_key)
    depths = load_depth_sequence(raw_dir, data, ep_id, args.camera, args.depth_key)

    eef_pose = np.asarray(data[args.eef_pose_key], dtype=np.float32)
    gripper = np.asarray(data[args.gripper_key], dtype=np.float32)
    steps = min(len(images), len(depths), len(eef_pose), len(gripper)) - 1
    if steps <= 0:
        raise ValueError("Episode has fewer than two aligned frames.")
    if steps < args.min_steps:
        raise ValueError(f"Episode has only {steps} aligned steps, below --min-steps={args.min_steps}.")

    images = images[:steps]
    depths = depths[:steps]
    curr_pose = eef_pose[:steps]
    next_pose = eef_pose[1 : steps + 1]
    gripper = gripper[:steps]

    rng = np.random.default_rng(args.seed + demo_idx * 100000)
    point_clouds = []
    for depth_m, rgb in zip(depths, images):
        points, colors = depth_to_point_cloud(
            depth_m=depth_m,
            rgb=rgb,
            fx=args.fx,
            fy=args.fy,
            cx=args.cx,
            cy=args.cy,
            stride=max(1, args.stride),
            frame=args.pointcloud_frame,
            camera_rot=camera_rot,
            camera_pos=camera_pos,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            z_min=args.z_min,
        )
        point_clouds.append(sample_point_cloud(points, colors, args.num_points, rng))

    actions = []
    for pose, width in zip(next_pose, gripper):
        gripper_cmd = -1.0 if float(np.mean(width)) < 0.025 else 1.0
        actions.append(np.concatenate([pose[:3], quat_wxyz_to_rotvec(pose[3:7]), [gripper_cmd]]))

    eye_images = np.stack(
        [cv2.resize(image, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA) for image in images],
        axis=0,
    ).astype(np.uint8)

    return {
        "actions": np.stack(actions, axis=0).astype(np.float32),
        "robot0_eef_pos": curr_pose[:, :3].astype(np.float32),
        "robot0_eef_quat": np.stack([quat_wxyz_to_xyzw(q) for q in curr_pose[:, 3:7]], axis=0),
        "robot0_gripper_qpos": gripper.astype(np.float32),
        "point_cloud": np.stack(point_clouds, axis=0).astype(np.float32),
        "robot0_eye_in_hand_image": eye_images,
        "source_episode_id": np.asarray([ep_id], dtype=np.int32),
    }


def write_demo(group: h5py.Group, demo_idx: int, source: Path, episode: dict[str, np.ndarray], compression) -> int:
    demo = group.create_group(f"demo_{demo_idx}")
    demo.attrs["source_npy"] = str(source)
    demo.attrs["source_episode_id"] = int(episode["source_episode_id"][0])
    demo.attrs["num_samples"] = int(episode["actions"].shape[0])
    demo.create_dataset("actions", data=episode["actions"], compression=compression)

    obs = demo.create_group("obs")
    for key in (
        "point_cloud",
        "robot0_eye_in_hand_image",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    ):
        obs.create_dataset(key, data=episode[key], compression=compression)
    return int(episode["actions"].shape[0])


def main() -> int:
    args = parse_args()
    raw_dir = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    files = list_episode_files(raw_dir, args.max_episodes)
    compression = None if args.compression == "none" else args.compression
    camera_rot = quat_wxyz_to_matrix(args.camera_quat_wxyz)
    camera_pos = np.asarray(args.camera_pos, dtype=np.float32)

    print(f"Input episodes: {len(files)} from {raw_dir}")
    print(f"Output: {output}")
    print(f"Point cloud: frame={args.pointcloud_frame}, points={args.num_points}, depth=[{args.min_depth}, {args.max_depth}]")

    total_samples = 0
    written = 0
    with h5py.File(output, "w") as h5:
        data_group = h5.create_group("data")
        h5.attrs["format"] = "dp3_sim_depth_point_cloud"
        h5.attrs["num_points"] = int(args.num_points)
        h5.attrs["pointcloud_camera"] = args.camera
        h5.attrs["pointcloud_frame"] = args.pointcloud_frame
        h5.attrs["depth_source"] = "sim_distance_to_image_plane"
        h5.attrs["min_depth"] = float(args.min_depth)
        h5.attrs["max_depth"] = float(args.max_depth)
        h5.attrs["z_min"] = float(args.z_min)
        h5.attrs["eye_image_stored_color_order"] = "rgb"
        h5.attrs["has_voxels"] = False
        h5.attrs["uses_sim_depth"] = True
        h5.attrs["uses_background_replacement"] = False

        iterator = tqdm(files, desc="Converting episodes")
        for path in iterator:
            try:
                episode = build_episode(path, raw_dir, args, camera_rot, camera_pos, written)
                total_samples += write_demo(data_group, written, path, episode, compression)
                print(f"[Write] demo_{written}: {path.name}, steps={episode['actions'].shape[0]}")
                written += 1
            except Exception as exc:
                if not args.skip_errors:
                    raise
                print(f"[Skip] {path.name}: {exc}")

        data_group.attrs["total"] = int(total_samples)
        data_group.attrs["num_demos"] = int(written)

    print(f"Done. Wrote {written}/{len(files)} demos, total samples={total_samples}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
