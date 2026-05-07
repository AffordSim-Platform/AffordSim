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
# Step 1: Affordance 
# ============================================================================
def cluster_affordance_points(points, labels, affordance_label=1,
                              dbscan_eps=0.02, dbscan_min_samples=5):
    """ affordance  DBSCAN 

    Returns:
        clusters: list of dict,  dict  'points'  'centroid'
    """
    aff_mask = (labels == affordance_label)
    aff_points = points[aff_mask]

    if len(aff_points) < 3:
        print(f"[WARN] affordance  ({len(aff_points)}), ")
        centroid = aff_points.mean(axis=0) if len(aff_points) > 0 else points.mean(axis=0)
        return [{'points': aff_points, 'centroid': centroid}]

    if len(aff_points) < 20:
        print(f"[INFO] affordance  ({len(aff_points)}), ")
        return [{'points': aff_points, 'centroid': aff_points.mean(axis=0)}]

    aff_pcd = o3d.geometry.PointCloud()
    aff_pcd.points = o3d.utility.Vector3dVector(aff_points)
    cluster_labels = np.array(aff_pcd.cluster_dbscan(
        eps=dbscan_eps, min_points=dbscan_min_samples, print_progress=False))

    if cluster_labels.max() < 0:
        print("[WARN] DBSCAN ,  affordance ")
        return [{'points': aff_points, 'centroid': aff_points.mean(axis=0)}]

    unique_labels, counts = np.unique(cluster_labels[cluster_labels >= 0], return_counts=True)
    # 
    sorted_indices = np.argsort(-counts)

    clusters = []
    for idx in sorted_indices:
        label = unique_labels[idx]
        cluster_mask = (cluster_labels == label)
        cluster_points = aff_points[cluster_mask]
        centroid = cluster_points.mean(axis=0)
        clusters.append({'points': cluster_points, 'centroid': centroid})

    print(f"[INFO] DBSCAN:  {len(clusters)}  ( {len(aff_points)}  affordance )")
    for i, c in enumerate(clusters):
        ct = c['centroid']
        print(f"   {i}: {len(c['points'])} , : [{ct[0]:.5f}, {ct[1]:.5f}, {ct[2]:.5f}]")

    return clusters


# ============================================================================
# Step 2: 
# ============================================================================
CLUSTER_COLORS = [
    [1.0, 1.0, 0.0],  # 
    [0.0, 1.0, 1.0],  # 
    [1.0, 0.0, 1.0],  # 
    [1.0, 0.5, 0.0],  # 
    [0.5, 0.0, 1.0],  # 
]


def visualize_affordance(points, labels, clusters):
    """ affordance """
    # : label=0 , label=1 
    colors = np.zeros((len(labels), 3))
    colors[labels == 0] = [0.0, 1.0, 0.0]
    colors[labels == 1] = [1.0, 0.0, 0.0]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    geometries = [pcd]

    # 
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    geometries.append(coord_frame)

    # 
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
# 
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
            raise ImportError(" torch,  .pt ")
        data = torch.load(pt_path, map_location="cpu")
        points = data['points'].numpy()
        labels = data['labels'].numpy()
        print(f"[INFO] Loaded: {pt_path}")
        return points, labels

    npz_path = None
    for name in NPZ_CANDIDATE_NAMES:
        candidate = os.path.join(type_path, name)
        if os.path.exists(candidate):
            npz_path = candidate
            break

    if npz_path is None:
        raise FileNotFoundError(f": {type_path}")

    data = np.load(npz_path)
    points = data['points']
    labels = data['labels']
    print(f"[INFO] Saved result: {npz_path}")
    return points, labels


def process_type_path(type_path, default_threshold=DEFAULT_UNIT_THRESHOLD):
    raw_points, labels = load_points_and_labels(type_path)
    save_path = os.path.join(type_path, "affordance_clusters.npz")

    def preview_fn(points, threshold, converted, coord_range):
        print(f"[INFO] Points: {len(points)}, label=0: {(labels==0).sum()}, label=1: {(labels==1).sum()}")

        #  (,  dbscan_eps )
        clusters = cluster_affordance_points(points, labels)

        print(f"\n{'='*60}")
        print(f"  Affordance :  {len(clusters)} ")
        for i, c in enumerate(clusters):
            ct = c['centroid']
            print(f"   {i}: {len(c['points'])} , : [{ct[0]:.5f}, {ct[1]:.5f}, {ct[2]:.5f}]")
        print(f"{'='*60}")

        geometries = visualize_affordance(points, labels, clusters)
        return attach_visualization(
            {"clusters": clusters},
            geometries,
            window_name=f"Affordance  ({os.path.basename(type_path)})",
            width=1280,
            height=720,
        )

    def accept_fn(payload, threshold, converted, coord_range):
        clusters = payload["clusters"]
        centroids = np.array([c['centroid'] for c in clusters])
        all_aff_points = np.concatenate([c['points'] for c in clusters], axis=0)
        save_dict = {
            'centroids': centroids,           # (N, 3) 
            'num_clusters': len(clusters),
            'aff_points': all_aff_points,
            'unit_threshold': np.array(threshold),
            'unit_converted': np.array(int(converted)),
        }
        for i, c in enumerate(clusters):
            save_dict[f'cluster_{i}_points'] = c['points']
        np.savez(save_path, **save_dict)
        print(f"[INFO] Saved result: {save_path}")

    return run_threshold_debug_loop(
        raw_points, os.path.basename(type_path), preview_fn, accept_fn, default_threshold
    )


# ============================================================================
# 
# ============================================================================
if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    type_paths = find_target_dirs(script_dir)

    if not type_paths:
        raise FileNotFoundError(f"{script_dir}  *_hang/*_place/*_pourin/*_pourout ")

    if len(type_paths) > 1:
        print(f"[INFO]  {len(type_paths)} , ")
        print("[INFO] Controls: Y/Enter=save, U/Up=x10, D/Down=/10, Q/Esc/X=quit")

    for index, type_path in enumerate(type_paths, start=1):
        folder_name = os.path.basename(type_path)
        print(f"\n[INFO] ({index}/{len(type_paths)}) : {folder_name}")
        try:
            if not process_type_path(type_path):
                print("[INFO] ")
                break
        except FileNotFoundError as exc:
            print(f"[WARN]  {folder_name}: {exc}")
