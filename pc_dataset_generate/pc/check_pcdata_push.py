"""
 pull 

:
  1.  affordance , 
  2.  pull/push 
  3. approach  check_pcdata_handle_open.py:
      XY ,  75 deg 
  4. , 
"""

import numpy as np
import open3d as o3d
import os
import argparse
from unit_threshold_helper import DEFAULT_UNIT_THRESHOLD, attach_visualization, run_threshold_debug_loop

try:
    import torch
except ImportError:
    torch = None


# ============================================================================
#  ()
# ============================================================================
GRIPPER_MAX_OPENING = 0.08        # 80mm
GRIPPER_FINGER_DEPTH = 0.10       # 100mm
GRIPPER_FINGER_WIDTH = 0.02       # 20mm
GRIPPER_FINGER_THICKNESS = 0.012  # 12mm
EE_OFFSET_Z = 0.220               # fr3_hand  to  ee_frame 

TILT_ANGLE_DEG = 75.0             # 75 deg =  75 deg 

CLUSTER_COLORS = [
    [1.0, 1.0, 0.0],  # 
    [0.0, 1.0, 1.0],  # 
    [1.0, 0.0, 1.0],  # 
    [1.0, 0.5, 0.0],  # 
    [0.5, 0.0, 1.0],  # 
]

PT_CANDIDATE_NAMES = ("labeled_result.pt",)
NPZ_CANDIDATE_NAMES = ("labeled_result.npz",)


# ============================================================================
# Step 1: Affordance 
# ============================================================================
def cluster_affordance_points(points, labels, affordance_label=1,
                              dbscan_eps=0.02, dbscan_min_samples=5):
    """ affordance  DBSCAN """
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
# Step 2:  pull 
# ============================================================================
def compute_cluster_grasp(cluster, obj_center):
    """ pull/push """
    grasp_center = cluster['centroid']

    diff_xy = obj_center[:2] - grasp_center[:2]

    if abs(diff_xy[0]) < 1e-6 and abs(diff_xy[1]) < 1e-6:
        print("[WARN]  XY ,  +X")
        approach_horiz = np.array([1.0, 0.0])
    elif abs(diff_xy[0]) >= abs(diff_xy[1]):
        approach_horiz = np.array([np.sign(diff_xy[0]), 0.0])
    else:
        approach_horiz = np.array([0.0, np.sign(diff_xy[1])])

    print(f"[INFO] approach : [{approach_horiz[0]:.0f}, {approach_horiz[1]:.0f}]")

    tilt_rad = np.radians(TILT_ANGLE_DEG)
    approach_dir = np.array([
        approach_horiz[0] * np.cos(tilt_rad),
        approach_horiz[1] * np.cos(tilt_rad),
        -np.sin(tilt_rad)
    ])
    approach_dir /= np.linalg.norm(approach_dir)

    grip_axis = np.array([-approach_horiz[1], approach_horiz[0], 0.0])
    grip_axis /= np.linalg.norm(grip_axis)

    binormal = np.cross(approach_dir, grip_axis)
    binormal /= np.linalg.norm(binormal)

    grip_axis = np.cross(binormal, approach_dir)
    grip_axis /= np.linalg.norm(grip_axis)

    rotation_matrix = np.column_stack([binormal, grip_axis, approach_dir])

    return {
        'grasp_center': grasp_center,
        'approach_dir': approach_dir,
        'grip_axis': grip_axis,
        'binormal': binormal,
        'rotation_matrix': rotation_matrix,
        'aff_points': cluster['points'],
    }


def compute_all_cluster_grasps(points, labels):
    """ affordance  pull """
    clusters = cluster_affordance_points(points, labels)
    obj_center = points.mean(axis=0)

    print(f"[INFO] Object center: [{obj_center[0]:.5f}, {obj_center[1]:.5f}, {obj_center[2]:.5f}]")

    grasp_results = []
    for i, cluster in enumerate(clusters):
        grasp_result = compute_cluster_grasp(cluster, obj_center)
        grasp_results.append(grasp_result)

        center = grasp_result['grasp_center']
        approach = grasp_result['approach_dir']
        grip = grasp_result['grip_axis']

        print(f"\n[INFO]  {i} ")
        print(f"  Center:     [{center[0]:.5f}, {center[1]:.5f}, {center[2]:.5f}]")
        print(f"  Approach:   [{approach[0]:.4f}, {approach[1]:.4f}, {approach[2]:.4f}]")
        print(f"  Grip axis:  [{grip[0]:.4f}, {grip[1]:.4f}, {grip[2]:.4f}]")

    return grasp_results, clusters, obj_center


# ============================================================================
# 
# ============================================================================
def rotation_from_z_to_dir(direction):
    z = np.array([0.0, 0.0, 1.0])
    d = np.array(direction, dtype=np.float64)
    d /= np.linalg.norm(d)
    v = np.cross(z, d)
    c = np.dot(z, d)
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def create_grasp_visualization(grasp_results):
    geometries = []

    half_open = GRIPPER_MAX_OPENING / 2.0
    fd = GRIPPER_FINGER_DEPTH
    fw = GRIPPER_FINGER_WIDTH
    ft = GRIPPER_FINGER_THICKNESS

    for i, grasp_result in enumerate(grasp_results):
        center = grasp_result['grasp_center']
        approach = grasp_result['approach_dir']
        grip = grasp_result['grip_axis']
        r_local = grasp_result['rotation_matrix']
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
        sphere.translate(center)
        sphere.paint_uniform_color(color)
        sphere.compute_vertex_normals()
        geometries.append(sphere)

        arrow_len = 0.07
        arrow_start = center - approach * arrow_len
        shaft = o3d.geometry.TriangleMesh.create_cylinder(radius=0.0015, height=arrow_len * 0.75)
        shaft.translate([0, 0, arrow_len * 0.75 / 2])
        r_arrow = rotation_from_z_to_dir(approach)
        shaft.rotate(r_arrow, center=[0, 0, 0])
        shaft.translate(arrow_start)
        shaft.paint_uniform_color(color)
        shaft.compute_vertex_normals()
        geometries.append(shaft)

        cone = o3d.geometry.TriangleMesh.create_cone(radius=0.004, height=arrow_len * 0.25)
        cone.translate([0, 0, arrow_len * 0.75])
        cone.rotate(r_arrow, center=[0, 0, 0])
        cone.translate(arrow_start)
        cone.paint_uniform_color(color)
        cone.compute_vertex_normals()
        geometries.append(cone)

        for sign in [+1, -1]:
            finger = o3d.geometry.TriangleMesh.create_box(width=fw, height=ft, depth=fd)
            finger.translate([-fw / 2, -ft / 2 + sign * half_open, -fd])
            finger.rotate(r_local, center=[0, 0, 0])
            finger.translate(center)
            finger.paint_uniform_color([0.0, 0.8, 0.8])
            finger.compute_vertex_normals()
            geometries.append(finger)

        grip_line_len = GRIPPER_MAX_OPENING + 2 * ft + 0.01
        grip_cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=0.0008, height=grip_line_len)
        grip_cyl.translate([0, 0, -grip_line_len / 2])
        r_grip = rotation_from_z_to_dir(grip)
        grip_cyl.rotate(r_grip, center=[0, 0, 0])
        grip_cyl.translate(center)
        grip_cyl.paint_uniform_color(color)
        grip_cyl.compute_vertex_normals()
        geometries.append(grip_cyl)

        grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.025)
        grasp_frame.rotate(r_local, center=[0, 0, 0])
        grasp_frame.translate(center)
        geometries.append(grasp_frame)

    return geometries


def _to_numpy(array_like):
    if hasattr(array_like, "cpu"):
        array_like = array_like.cpu()
    if hasattr(array_like, "numpy"):
        return array_like.numpy()
    return np.asarray(array_like)


def _load_pt_data(pt_path):
    if torch is None:
        raise ImportError("torch  .pt ")

    data = torch.load(pt_path, map_location="cpu")
    return _to_numpy(data["points"]), _to_numpy(data["labels"])


def _load_npz_data(npz_path):
    with np.load(npz_path) as data:
        return data["points"], data["labels"]


def _iter_candidate_files(base_dir):
    for name in PT_CANDIDATE_NAMES:
        path = os.path.join(base_dir, name)
        if os.path.isfile(path):
            yield path

    for name in NPZ_CANDIDATE_NAMES:
        path = os.path.join(base_dir, name)
        if os.path.isfile(path):
            yield path


def resolve_input_path(script_dir, input_dir=None, input_file=None):
    if input_file is not None:
        input_path = input_file if os.path.isabs(input_file) else os.path.join(script_dir, input_file)
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f": {input_path}")
        return input_path

    candidate_dirs = []
    if input_dir is not None:
        resolved_dir = input_dir if os.path.isabs(input_dir) else os.path.join(script_dir, input_dir)
        if not os.path.isdir(resolved_dir):
            raise FileNotFoundError(f": {resolved_dir}")
        candidate_dirs.append(resolved_dir)
    else:
        candidate_dirs = sorted(
            os.path.join(script_dir, entry)
            for entry in os.listdir(script_dir)
            if entry.endswith("_push") and os.path.isdir(os.path.join(script_dir, entry))
        )

    searched_dirs = []
    for candidate_dir in candidate_dirs:
        searched_dirs.append(candidate_dir)
        for path in _iter_candidate_files(candidate_dir):
            return path

    if input_dir is not None:
        raise FileNotFoundError(f": {candidate_dirs[0]}")

    raise FileNotFoundError(
        " push "
        f" : {searched_dirs if searched_dirs else [script_dir]}"
    )


def load_points_and_labels(script_dir, input_dir=None, input_file=None):
    input_path = resolve_input_path(script_dir, input_dir=input_dir, input_file=input_file)
    input_dirname = os.path.dirname(input_path)

    if input_path.endswith(".pt"):
        points, labels = _load_pt_data(input_path)
    elif input_path.endswith(".npz"):
        points, labels = _load_npz_data(input_path)
    else:
        raise ValueError(f": {input_path}")

    return points, labels, input_path, input_dirname


def find_push_dirs(script_dir):
    return sorted(
        os.path.join(script_dir, entry)
        for entry in os.listdir(script_dir)
        if entry.endswith("_push") and os.path.isdir(os.path.join(script_dir, entry))
    )


def process_input_path(script_dir, input_dir=None, input_file=None, visualize=True, default_threshold=DEFAULT_UNIT_THRESHOLD):
    raw_points, labels, input_path, type_path = load_points_and_labels(
        script_dir,
        input_dir=input_dir,
        input_file=input_file,
    )
    print(f"[INFO] Loaded: {input_path}")
    save_path = os.path.join(type_path, "grasp_result.npz")

    def preview_fn(points, threshold, converted, coord_range):
        print(f"[INFO] Points: {len(points)}, label=0: {(labels==0).sum()}, label=1: {(labels==1).sum()}")

        colors = np.zeros((len(labels), 3))
        colors[labels == 0] = [0.0, 1.0, 0.0]
        colors[labels == 1] = [1.0, 0.0, 0.0]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
        grasp_results, clusters, obj_center = compute_all_cluster_grasps(points, labels)

        print(f"\n{'='*60}")
        print(f"  Push :  {len(grasp_results)} ")
        for i, grasp_result in enumerate(grasp_results):
            ee_pos = grasp_result['grasp_center'] - grasp_result['approach_dir'] * EE_OFFSET_Z
            print(f"   {i}:")
            print(f"    Center:   [{grasp_result['grasp_center'][0]:.5f}, "
                  f"{grasp_result['grasp_center'][1]:.5f}, {grasp_result['grasp_center'][2]:.5f}]")
            print(f"    EE:   [{ee_pos[0]:.5f}, {ee_pos[1]:.5f}, {ee_pos[2]:.5f}]")
        print(f"{'='*60}")

        if visualize:
            grasp_geometries = create_grasp_visualization(grasp_results)
            all_geometries = [pcd, coord_frame] + grasp_geometries
            return attach_visualization(
                {"grasp_results": grasp_results, "obj_center": obj_center},
                all_geometries,
                window_name=" Push ",
                width=1280,
                height=720,
            )
        return {"grasp_results": grasp_results, "obj_center": obj_center}

    def accept_fn(payload, threshold, converted, coord_range):
        grasp_results = payload["grasp_results"]
        obj_center = payload["obj_center"]
        save_dict = {
            'num_clusters': len(grasp_results),
            'tilt_angle': np.array(TILT_ANGLE_DEG),
            'object_center': obj_center,
            'unit_threshold': np.array(threshold),
            'unit_converted': np.array(int(converted)),
        }
        for i, grasp_result in enumerate(grasp_results):
            prefix = f'cluster_{i}_'
            ee_pos = grasp_result['grasp_center'] - grasp_result['approach_dir'] * EE_OFFSET_Z
            save_dict[prefix + 'grasp_center'] = grasp_result['grasp_center']
            save_dict[prefix + 'approach_dir'] = grasp_result['approach_dir']
            save_dict[prefix + 'grip_axis'] = grasp_result['grip_axis']
            save_dict[prefix + 'binormal'] = grasp_result['binormal']
            save_dict[prefix + 'rotation_matrix'] = grasp_result['rotation_matrix']
            save_dict[prefix + 'score'] = np.array(1.0)
            save_dict[prefix + 'ee_position'] = ee_pos
            save_dict[prefix + 'aff_points'] = grasp_result['aff_points']
        if len(grasp_results) > 0:
            ee_pos = grasp_results[0]['grasp_center'] - grasp_results[0]['approach_dir'] * EE_OFFSET_Z
            save_dict['grasp_center'] = grasp_results[0]['grasp_center']
            save_dict['approach_dir'] = grasp_results[0]['approach_dir']
            save_dict['grip_axis'] = grasp_results[0]['grip_axis']
            save_dict['binormal'] = grasp_results[0]['binormal']
            save_dict['rotation_matrix'] = grasp_results[0]['rotation_matrix']
            save_dict['score'] = np.array(1.0)
            save_dict['ee_position'] = ee_pos
        np.savez(save_path, **save_dict)
        print(f"[INFO] Push : {save_path}")

    return run_threshold_debug_loop(raw_points, os.path.basename(type_path), preview_fn, accept_fn, default_threshold)


# ============================================================================
# 
# ============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=" push ")
    parser.add_argument("--input-dir", type=str, default=None, help=" *_push ")
    parser.add_argument("--input-file", type=str, default=None, help=" .pt/.npz")
    parser.add_argument("--no-vis", action="store_true", help=" Open3D ")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    visualize = not args.no_vis

    if args.input_file is not None or args.input_dir is not None:
        process_input_path(
            script_dir,
            input_dir=args.input_dir,
            input_file=args.input_file,
            visualize=visualize,
        )
    else:
        push_dirs = find_push_dirs(script_dir)
        if not push_dirs:
            raise FileNotFoundError(f"{script_dir}  *_push ")

        if visualize and len(push_dirs) > 1:
            print(f"[INFO]  {len(push_dirs)}  *_push ,  Open3D  Y/Enter U/D Q/Esc/X ")

        for index, push_dir in enumerate(push_dirs, start=1):
            folder_name = os.path.basename(push_dir)
            print(f"\n[INFO] ({index}/{len(push_dirs)}) : {folder_name}")
            if not process_input_path(
                script_dir,
                input_dir=push_dir,
                input_file=None,
                visualize=visualize,
            ):
                print("[INFO] ")
                break
