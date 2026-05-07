import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from gsnet import AnyGrasp


SCRIPT_DIR = Path(__file__).resolve().parent
ANYGRASP_HOME = Path(os.environ.get("ANYGRASP_HOME", SCRIPT_DIR.parent)).expanduser().resolve()
DEFAULT_CHECKPOINT = ANYGRASP_HOME / "log" / "checkpoint_detection.tar"
DEFAULT_DATA_DIR = Path(os.environ.get("ANYGRASP_DATA_DIR", Path.cwd())).expanduser()


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", default=str(DEFAULT_CHECKPOINT), help="Model checkpoint path")
parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR), help="Directory containing rgb_*.npy, depth_*.npy and bbox_*.json")
parser.add_argument("--max_gripper_width", type=float, default=0.1, help="Maximum gripper width (<=0.1m)")
parser.add_argument("--gripper_height", type=float, default=0.03, help="Gripper height")
parser.add_argument("--top_down_grasp", action="store_true", help="Output top-down grasps")
parser.add_argument("--debug", action="store_true", help="Enable debug visualization")
parser.add_argument("--output_dir", type=str, default="", help="Deprecated; results are written into data_dir")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to process")

cfgs = parser.parse_args()
cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))


def process_envs(data_dir: Path) -> None:
    """
    Batch process AnyGrasp inputs from AffordSim.

    Reads:
        rgb_{i}.npy, depth_{i}.npy, bbox_{i}.json
    Writes:
        best_grasp_pose_{i}.json
    """
    data_dir = data_dir.expanduser().resolve()
    checkpoint = Path(cfgs.checkpoint_path).expanduser().resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"AnyGrasp data directory does not exist: {data_dir}")
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"AnyGrasp checkpoint does not exist: {checkpoint}\n"
            "Set ANYGRASP_HOME to the SDK root, or pass --checkpoint_path explicitly."
        )

    print(f"Loading AnyGrasp checkpoint: {checkpoint}")
    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()
    print("Model loaded.\n")

    fx, fy = 927.17, 927.37
    cx, cy = 651.32, 349.62
    scale = 1000.0

    total_start_time = time.time()
    success_count = 0

    for env_id in range(cfgs.num_envs):
        print(f"--- Processing Env ID: {env_id} ---")
        rgb_path = data_dir / f"rgb_{env_id}.npy"
        depth_path = data_dir / f"depth_{env_id}.npy"
        bbox_path = data_dir / f"bbox_{env_id}.json"
        output_path = data_dir / f"best_grasp_pose_{env_id}.json"

        if not rgb_path.exists() or not depth_path.exists():
            print(f"[WARN] Missing input for Env {env_id}: {rgb_path} or {depth_path}")
            continue

        try:
            colors = np.array(np.load(rgb_path), dtype=np.float32) / 255.0
            depths = np.squeeze(np.array(np.load(depth_path)))

            try:
                with open(bbox_path, "r", encoding="utf-8") as f:
                    bbox_data = json.load(f)
                xmin = bbox_data["world_xmin"]
                xmax = bbox_data["world_xmax"]
                ymin = bbox_data["world_ymin"]
                ymax = bbox_data["world_ymax"]
                zmin, zmax = 0.0, 1.0
            except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
                print(f"[WARN] Could not load {bbox_path} ({exc}). Using default workspace.")
                xmin, xmax = -0.30, 0.50
                ymin, ymax = -0.50, 0.50
                zmin, zmax = 0.0, 1.0

            lims = [xmin, xmax, ymin, ymax, zmin, zmax]

            xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
            xmap, ymap = np.meshgrid(xmap, ymap)
            points_z = depths / scale
            points_x = (xmap - cx) / fx * points_z
            points_y = (ymap - cy) / fy * points_z

            mask = (points_z > 0) & (points_z < 1)
            points = np.stack([points_x, points_y, points_z], axis=-1)
            points = points[mask].astype(np.float32)
            colors_filtered = colors[mask].astype(np.float32)

            if len(points) == 0:
                print(f"[ERROR] Env {env_id}: no valid point cloud points.")
                continue

            gg, cloud = anygrasp.get_grasp(
                points,
                colors_filtered,
                lims=lims,
                apply_object_mask=True,
                dense_grasp=False,
                collision_detection=True,
            )

            if len(gg) == 0:
                print(f"[INFO] Env {env_id}: no valid grasp found.")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump({"best_grasp": None, "status": "no_grasp_found"}, f, indent=4)
                continue

            gg = gg.nms().sort_by_score()
            best_grasp = gg[0]

            print(f"Env {env_id}: detected {len(gg)} grasps. Export score: {best_grasp.score:.4f}")
            print(f"  translation: {best_grasp.translation}")
            print(f"  width: {best_grasp.width:.4f}")

            best_grasp_data = {
                "env_id": env_id,
                "status": "success",
                "best_grasp": {
                    "translation": best_grasp.translation.tolist(),
                    "rotation_matrix": best_grasp.rotation_matrix.tolist(),
                    "width": float(best_grasp.width),
                    "depth": float(best_grasp.depth),
                    "score": float(best_grasp.score),
                },
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(best_grasp_data, f, indent=4)

            print(f"Saved result to: {output_path}")
            success_count += 1

            if cfgs.debug:
                trans_mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
                cloud.transform(trans_mat)
                gripper = best_grasp.to_open3d_geometry()
                gripper.transform(trans_mat)
                coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                o3d.visualization.draw_geometries([gripper, cloud, coord])

        except Exception as exc:
            print(f"[ERROR] Env {env_id} failed: {exc}")
            import traceback

            traceback.print_exc()

    total_time = time.time() - total_start_time
    print("\n========================================")
    print("Batch processing completed.")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Successful envs: {success_count} / {cfgs.num_envs}")
    print("========================================")


if __name__ == "__main__":
    process_envs(Path(cfgs.data_dir))
