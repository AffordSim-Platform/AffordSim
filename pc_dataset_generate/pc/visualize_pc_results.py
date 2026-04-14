import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

try:
    import torch
except ImportError:
    torch = None


GRIPPER_MAX_OPENING = 0.08
GRIPPER_FINGER_DEPTH = 0.10
GRIPPER_FINGER_WIDTH = 0.02
GRIPPER_FINGER_THICKNESS = 0.012

LABEL_FILE_CANDIDATES = (
    "label_result.pt",
    "labeled_result.pt",
    "label_result.npz",
    "labeled_result.npz",
)
RESULT_FILE_CANDIDATES = ("grasp_result.npz", "affordance_clusters.npz")
CLUSTER_COLORS = [
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 0.5, 0.0],
    [0.5, 0.0, 1.0],
    [0.0, 0.7, 1.0],
]


def parse_args():
    parser = argparse.ArgumentParser(description="依次可视化 pc_dataset_generate/pc 下的抓取和 affordance 结果。")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent, help="数据根目录")
    parser.add_argument("--contains", type=str, default=None, help="仅可视化目录名包含该子串的结果")
    parser.add_argument("--result", choices=("all", "grasp", "affordance"), default="all", help="结果类型筛选")
    parser.add_argument("--limit", type=int, default=None, help="最多可视化多少个结果")
    parser.add_argument("--start-at", type=str, default=None, help="从目录名包含该子串的位置开始")
    parser.add_argument("--list-only", action="store_true", help="仅列出将要可视化的结果，不打开窗口")
    return parser.parse_args()


def rotation_from_z_to_dir(direction):
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    d = np.array(direction, dtype=np.float64)
    d /= np.linalg.norm(d)
    v = np.cross(z, d)
    c = np.dot(z, d)
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def resolve_label_file(folder: Path) -> Path | None:
    for name in LABEL_FILE_CANDIDATES:
        candidate = folder / name
        if candidate.exists():
            return candidate
    return None


def load_points_and_labels(label_path: Path):
    if label_path.suffix == ".pt":
        if torch is None:
            raise ImportError("需要 torch 才能读取 .pt 文件")
        data = torch.load(label_path, map_location="cpu")
        points = data["points"]
        labels = data["labels"]
        points = points.numpy() if hasattr(points, "numpy") else np.asarray(points, dtype=np.float64)
        labels = labels.numpy() if hasattr(labels, "numpy") else np.asarray(labels)
    else:
        data = np.load(label_path)
        points = np.asarray(data["points"], dtype=np.float64)
        labels = np.asarray(data["labels"])

    coord_range = float(np.ptp(points, axis=0).max())
    if coord_range > 100.0:
        print(f"[INFO] {label_path.parent.name}: 检测到毫米单位，转换为米")
        points = points / 1000.0
    return points, labels


def create_colored_point_cloud(points, labels):
    colors = np.zeros((len(labels), 3), dtype=np.float64)
    colors[labels == 0] = [0.0, 1.0, 0.0]
    colors[labels == 1] = [1.0, 0.0, 0.0]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def create_grasp_visualization(result_path: Path):
    data = np.load(result_path)

    center = np.asarray(data["grasp_center"], dtype=np.float64)
    approach = np.asarray(data["approach_dir"], dtype=np.float64)
    grip = np.asarray(data["grip_axis"], dtype=np.float64)
    approach /= np.linalg.norm(approach)
    grip /= np.linalg.norm(grip)

    if "binormal" in data:
        binormal = np.asarray(data["binormal"], dtype=np.float64)
        binormal /= np.linalg.norm(binormal)
    else:
        binormal = np.cross(approach, grip)
        binormal /= np.linalg.norm(binormal)

    r_local = np.column_stack([binormal, grip, approach])
    geometries = []

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
    sphere.translate(center)
    sphere.paint_uniform_color([1.0, 1.0, 0.0])
    sphere.compute_vertex_normals()
    geometries.append(sphere)

    arrow_len = 0.07
    arrow_start = center - approach * arrow_len
    shaft = o3d.geometry.TriangleMesh.create_cylinder(radius=0.0015, height=arrow_len * 0.75)
    shaft.translate([0.0, 0.0, arrow_len * 0.75 / 2.0])
    r_arrow = rotation_from_z_to_dir(approach)
    shaft.rotate(r_arrow, center=[0.0, 0.0, 0.0])
    shaft.translate(arrow_start)
    shaft.paint_uniform_color([0.0, 0.3, 1.0])
    shaft.compute_vertex_normals()
    geometries.append(shaft)

    cone = o3d.geometry.TriangleMesh.create_cone(radius=0.004, height=arrow_len * 0.25)
    cone.translate([0.0, 0.0, arrow_len * 0.75])
    cone.rotate(r_arrow, center=[0.0, 0.0, 0.0])
    cone.translate(arrow_start)
    cone.paint_uniform_color([0.0, 0.3, 1.0])
    cone.compute_vertex_normals()
    geometries.append(cone)

    half_open = GRIPPER_MAX_OPENING / 2.0
    for sign in [+1, -1]:
        finger = o3d.geometry.TriangleMesh.create_box(
            width=GRIPPER_FINGER_WIDTH,
            height=GRIPPER_FINGER_THICKNESS,
            depth=GRIPPER_FINGER_DEPTH,
        )
        finger.translate(
            [
                -GRIPPER_FINGER_WIDTH / 2.0,
                -GRIPPER_FINGER_THICKNESS / 2.0 + sign * half_open,
                -GRIPPER_FINGER_DEPTH,
            ]
        )
        finger.rotate(r_local, center=[0.0, 0.0, 0.0])
        finger.translate(center)
        finger.paint_uniform_color([0.0, 0.8, 0.8])
        finger.compute_vertex_normals()
        geometries.append(finger)

    grip_line_len = GRIPPER_MAX_OPENING + 2 * GRIPPER_FINGER_THICKNESS + 0.01
    grip_cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=0.0008, height=grip_line_len)
    grip_cyl.translate([0.0, 0.0, -grip_line_len / 2.0])
    r_grip = rotation_from_z_to_dir(grip)
    grip_cyl.rotate(r_grip, center=[0.0, 0.0, 0.0])
    grip_cyl.translate(center)
    grip_cyl.paint_uniform_color([1.0, 0.0, 1.0])
    grip_cyl.compute_vertex_normals()
    geometries.append(grip_cyl)

    grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.025)
    grasp_frame.rotate(r_local, center=[0.0, 0.0, 0.0])
    grasp_frame.translate(center)
    geometries.append(grasp_frame)

    score = float(np.asarray(data["score"]).reshape(-1)[0]) if "score" in data else None
    return geometries, {
        "grasp_center": center,
        "approach_dir": approach,
        "grip_axis": grip,
        "score": score,
    }


def create_affordance_visualization(result_path: Path):
    data = np.load(result_path)
    geometries = []

    centroids = np.asarray(data["centroids"], dtype=np.float64)
    for i, centroid in enumerate(centroids):
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        sphere.translate(centroid)
        sphere.paint_uniform_color(color)
        sphere.compute_vertex_normals()
        geometries.append(sphere)

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
        frame.translate(centroid)
        geometries.append(frame)

        cluster_key = f"cluster_{i}_points"
        if cluster_key in data:
            cluster_pcd = o3d.geometry.PointCloud()
            cluster_pts = np.asarray(data[cluster_key], dtype=np.float64)
            cluster_pcd.points = o3d.utility.Vector3dVector(cluster_pts)
            cluster_pcd.paint_uniform_color(color)
            geometries.append(cluster_pcd)

    info = {
        "num_clusters": int(np.asarray(data["num_clusters"]).reshape(-1)[0]) if "num_clusters" in data else len(centroids),
        "centroids": centroids,
    }
    return geometries, info


def build_items(root: Path, result_filter: str):
    items = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        label_file = resolve_label_file(folder)
        if label_file is None:
            continue

        result_names = []
        for result_name in RESULT_FILE_CANDIDATES:
            result_path = folder / result_name
            if result_path.exists():
                result_names.append(result_name)

        if result_filter == "grasp":
            result_names = [n for n in result_names if n == "grasp_result.npz"]
        elif result_filter == "affordance":
            result_names = [n for n in result_names if n == "affordance_clusters.npz"]

        for result_name in result_names:
            items.append((folder, label_file, folder / result_name))
    return items


def print_grasp_info(folder: Path, info: dict):
    center = info["grasp_center"]
    approach = info["approach_dir"]
    grip = info["grip_axis"]
    elevation = np.degrees(np.arcsin(np.clip(-approach[2], -1.0, 1.0)))
    print(f"[GRASP] {folder.name}")
    print(f"  center:   [{center[0]:.5f}, {center[1]:.5f}, {center[2]:.5f}]")
    print(f"  approach: [{approach[0]:.4f}, {approach[1]:.4f}, {approach[2]:.4f}]")
    print(f"  grip:     [{grip[0]:.4f}, {grip[1]:.4f}, {grip[2]:.4f}]")
    print(f"  elevation:{elevation:.1f}°")
    if info["score"] is not None:
        print(f"  score:    {info['score']:.4f}")


def print_affordance_info(folder: Path, info: dict):
    print(f"[AFFORDANCE] {folder.name}")
    print(f"  num_clusters: {info['num_clusters']}")
    for i, centroid in enumerate(info["centroids"]):
        print(f"  centroid_{i}: [{centroid[0]:.5f}, {centroid[1]:.5f}, {centroid[2]:.5f}]")


def main():
    args = parse_args()
    root = args.root.resolve()
    items = build_items(root, args.result)

    if args.contains:
        items = [item for item in items if args.contains in item[0].name]

    if args.start_at:
        started = False
        filtered = []
        for item in items:
            if not started and args.start_at in item[0].name:
                started = True
            if started:
                filtered.append(item)
        items = filtered

    if args.limit is not None:
        items = items[: args.limit]

    if not items:
        print("[WARN] 没有找到可视化结果。")
        return

    print(f"[INFO] 共找到 {len(items)} 个结果，关闭当前窗口后会自动打开下一个。")

    if args.list_only:
        for idx, (folder, label_file, result_path) in enumerate(items, start=1):
            print(f"[{idx}/{len(items)}] {folder.name} -> {result_path.name} ({label_file.name})")
        return

    for idx, (folder, label_file, result_path) in enumerate(items, start=1):
        print(f"\n{'=' * 90}")
        print(f"[{idx}/{len(items)}] {folder.name} -> {result_path.name}")
        print(f"  label file: {label_file.name}")

        try:
            points, labels = load_points_and_labels(label_file)
        except Exception as exc:
            print(f"[WARN] 读取标注失败，跳过: {exc}")
            continue

        geometries = [
            create_colored_point_cloud(points, labels),
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0.0, 0.0, 0.0]),
        ]

        if result_path.name == "grasp_result.npz":
            result_geometries, info = create_grasp_visualization(result_path)
            geometries.extend(result_geometries)
            print_grasp_info(folder, info)
            window_name = f"{folder.name} | grasp_result"
        else:
            result_geometries, info = create_affordance_visualization(result_path)
            geometries.extend(result_geometries)
            print_affordance_info(folder, info)
            window_name = f"{folder.name} | affordance_clusters"

        o3d.visualization.draw_geometries(
            geometries,
            window_name=window_name,
            width=1280,
            height=720,
        )


if __name__ == "__main__":
    main()
