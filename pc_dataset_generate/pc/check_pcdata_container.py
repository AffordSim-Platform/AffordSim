import numpy as np
import open3d as o3d
import os

try:
    import torch
except ImportError:
    torch = None

from unit_threshold_helper import (
    DEFAULT_UNIT_THRESHOLD,
    attach_visualization,
    run_threshold_debug_loop,
)


PT_CANDIDATE_NAMES = ("labeled_result.pt",)
NPZ_CANDIDATE_NAMES = ("labeled_result.npz",)


# ============================================================================
# Step 1: Affordance 点聚类
# ============================================================================
def cluster_affordance_points(points, labels, affordance_label=1,
                              dbscan_eps=0.02, dbscan_min_samples=5):
    """提取 affordance 点并用 DBSCAN 聚类，返回所有连续簇的点和质心列表。

    Returns:
        clusters: list of dict, 每个 dict 包含 'points' 和 'centroid'，按点数从大到小排列。
    """
    aff_mask = (labels == affordance_label)
    aff_points = points[aff_mask]

    if len(aff_points) < 3:
        print(f"[WARN] affordance 点太少 ({len(aff_points)}), 使用全部")
        centroid = aff_points.mean(axis=0) if len(aff_points) > 0 else points.mean(axis=0)
        return [{'points': aff_points, 'centroid': centroid}]

    if len(aff_points) < 20:
        print(f"[INFO] affordance 点较少 ({len(aff_points)}), 跳过聚类")
        return [{'points': aff_points, 'centroid': aff_points.mean(axis=0)}]

    aff_pcd = o3d.geometry.PointCloud()
    aff_pcd.points = o3d.utility.Vector3dVector(aff_points)
    cluster_labels = np.array(aff_pcd.cluster_dbscan(
        eps=dbscan_eps, min_points=dbscan_min_samples, print_progress=False))

    if cluster_labels.max() < 0:
        print("[WARN] DBSCAN 未发现聚类, 使用全部 affordance 点")
        return [{'points': aff_points, 'centroid': aff_points.mean(axis=0)}]

    unique_labels, counts = np.unique(cluster_labels[cluster_labels >= 0], return_counts=True)
    # 按点数从大到小排列
    sorted_indices = np.argsort(-counts)

    clusters = []
    for idx in sorted_indices:
        label = unique_labels[idx]
        cluster_mask = (cluster_labels == label)
        cluster_points = aff_points[cluster_mask]
        centroid = cluster_points.mean(axis=0)
        clusters.append({'points': cluster_points, 'centroid': centroid})

    print(f"[INFO] DBSCAN: 发现 {len(clusters)} 个簇 (共 {len(aff_points)} 个 affordance 点)")
    for i, c in enumerate(clusters):
        ct = c['centroid']
        print(f"  簇 {i}: {len(c['points'])} 个点, 质心: [{ct[0]:.5f}, {ct[1]:.5f}, {ct[2]:.5f}]")

    return clusters


# ============================================================================
# Step 2: 可视化
# ============================================================================
CLUSTER_COLORS = [
    [1.0, 1.0, 0.0],  # 黄色
    [0.0, 1.0, 1.0],  # 青色
    [1.0, 0.0, 1.0],  # 品红
    [1.0, 0.5, 0.0],  # 橙色
    [0.5, 0.0, 1.0],  # 紫色
]


def visualize_affordance(points, labels, clusters):
    """可视化点云和所有 affordance 聚类质心。"""
    # 原始点云着色: label=0 绿色, label=1 红色
    colors = np.zeros((len(labels), 3))
    colors[labels == 0] = [0.0, 1.0, 0.0]
    colors[labels == 1] = [1.0, 0.0, 0.0]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    geometries = [pcd]

    # 坐标系
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    geometries.append(coord_frame)

    # 为每个簇添加质心球体和坐标系
    for i, cluster in enumerate(clusters):
        centroid = cluster['centroid']
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        sphere.translate(centroid)
        sphere.paint_uniform_color(color)
        sphere.compute_vertex_normals()
        geometries.append(sphere)

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
        frame.translate(centroid)
        geometries.append(frame)

    return geometries


# ============================================================================
# 数据加载与批处理
# ============================================================================
TARGET_SUFFIXES = ("_hang", "_place", "_pourin", "_pourout")


def find_target_dirs(script_dir):
    return sorted(
        os.path.join(script_dir, entry)
        for entry in os.listdir(script_dir)
        if entry.endswith(TARGET_SUFFIXES) and os.path.isdir(os.path.join(script_dir, entry))
    )


def load_points_and_labels(type_path):
    pt_path = None
    for name in PT_CANDIDATE_NAMES:
        candidate = os.path.join(type_path, name)
        if os.path.exists(candidate):
            pt_path = candidate
            break

    if pt_path is not None:
        if torch is None:
            raise ImportError("未安装 torch, 无法加载 .pt 文件")
        data = torch.load(pt_path, map_location="cpu")
        points = data['points'].numpy()
        labels = data['labels'].numpy()
        print(f"[INFO] 加载: {pt_path}")
        return points, labels

    npz_path = None
    for name in NPZ_CANDIDATE_NAMES:
        candidate = os.path.join(type_path, name)
        if os.path.exists(candidate):
            npz_path = candidate
            break

    if npz_path is None:
        raise FileNotFoundError(f"未找到标注文件: {type_path}")

    data = np.load(npz_path)
    points = data['points']
    labels = data['labels']
    print(f"[INFO] 加载: {npz_path}")
    return points, labels


def process_type_path(type_path, default_threshold=DEFAULT_UNIT_THRESHOLD):
    raw_points, labels = load_points_and_labels(type_path)
    save_path = os.path.join(type_path, "affordance_clusters.npz")

    def preview_fn(points, threshold, converted, coord_range):
        print(f"[INFO] 总点数: {len(points)}, label=0: {(labels==0).sum()}, label=1: {(labels==1).sum()}")

        # 聚类 (每轮阈值变化都要重算, 因为 dbscan_eps 是基于米的)
        clusters = cluster_affordance_points(points, labels)

        print(f"\n{'='*60}")
        print(f"  Affordance 聚类结果: 共 {len(clusters)} 个簇")
        for i, c in enumerate(clusters):
            ct = c['centroid']
            print(f"  簇 {i}: {len(c['points'])} 个点, 质心: [{ct[0]:.5f}, {ct[1]:.5f}, {ct[2]:.5f}]")
        print(f"{'='*60}")

        geometries = visualize_affordance(points, labels, clusters)
        return attach_visualization(
            {"clusters": clusters},
            geometries,
            window_name=f"Affordance 聚类可视化 ({os.path.basename(type_path)})",
            width=1280,
            height=720,
        )

    def accept_fn(payload, threshold, converted, coord_range):
        clusters = payload["clusters"]
        centroids = np.array([c['centroid'] for c in clusters])
        all_aff_points = np.concatenate([c['points'] for c in clusters], axis=0)
        save_dict = {
            'centroids': centroids,           # (N, 3) 所有簇的质心
            'num_clusters': len(clusters),
            'aff_points': all_aff_points,
            'unit_threshold': np.array(threshold),
            'unit_converted': np.array(int(converted)),
        }
        for i, c in enumerate(clusters):
            save_dict[f'cluster_{i}_points'] = c['points']
        np.savez(save_path, **save_dict)
        print(f"[INFO] 聚类结果已保存: {save_path}")

    return run_threshold_debug_loop(
        raw_points, os.path.basename(type_path), preview_fn, accept_fn, default_threshold
    )


# ============================================================================
# 主程序
# ============================================================================
if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    type_paths = find_target_dirs(script_dir)

    if not type_paths:
        raise FileNotFoundError(f"{script_dir} 下未找到任何 *_hang/*_place/*_pourin/*_pourout 文件夹")

    if len(type_paths) > 1:
        print(f"[INFO] 检测到 {len(type_paths)} 个目标文件夹, 将逐个处理。")
        print("[INFO] 键位: Y/Enter=保存, U/↑=阈值×10, D/↓=阈值÷10, Q/Esc/X=退出批处理")

    for index, type_path in enumerate(type_paths, start=1):
        folder_name = os.path.basename(type_path)
        print(f"\n[INFO] ({index}/{len(type_paths)}) 处理文件夹: {folder_name}")
        try:
            if not process_type_path(type_path):
                print("[INFO] 用户中止批处理。")
                break
        except FileNotFoundError as exc:
            print(f"[WARN] 跳过 {folder_name}: {exc}")
