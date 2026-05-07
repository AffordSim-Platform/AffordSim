"""
 -- 

:
  1.  affordance   to  
  2. : approach  XY ()
  3. approach  (0 deg ), 
  4. grip axis  Z   to  , 
  5. , 
"""

import numpy as np
import open3d as o3d
import os
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
EE_OFFSET_Z = 0.220              # fr3_hand  to  ee_frame 

TILT_ANGLE_DEG = 0.0             # approach  ()
                                   # 0 deg = , 


# ============================================================================
# Step 1: Affordance   to  
# ============================================================================
def cluster_affordance_points(points, labels, affordance_label=1,
                              dbscan_eps=0.008, dbscan_min_samples=5):
    """ affordance , ."""
    aff_mask = (labels == affordance_label)
    aff_points = points[aff_mask]

    if len(aff_points) < 3:
        print(f"[WARN] affordance  ({len(aff_points)}), ")
        centroid = aff_points.mean(axis=0) if len(aff_points) > 0 else points.mean(axis=0)
        return aff_points, centroid

    if len(aff_points) < 20:
        print(f"[INFO] affordance  ({len(aff_points)}), ")
        return aff_points, aff_points.mean(axis=0)

    aff_pcd = o3d.geometry.PointCloud()
    aff_pcd.points = o3d.utility.Vector3dVector(aff_points)
    cluster_labels = np.array(aff_pcd.cluster_dbscan(
        eps=dbscan_eps, min_points=dbscan_min_samples, print_progress=False))

    if cluster_labels.max() < 0:
        print("[WARN] DBSCAN ,  affordance ")
        return aff_points, aff_points.mean(axis=0)

    unique_labels, counts = np.unique(cluster_labels[cluster_labels >= 0], return_counts=True)
    largest_label = unique_labels[np.argmax(counts)]
    cluster_mask = (cluster_labels == largest_label)
    cluster_points = aff_points[cluster_mask]

    print(f"[INFO] DBSCAN: {len(unique_labels)} ,  {np.max(counts)}  "
          f"( {len(aff_points)}  affordance )")

    return cluster_points, cluster_points.mean(axis=0)


# ============================================================================
# Step 2: 
# ============================================================================
def compute_handle_grasp(points, labels):
    """
    .

    :
      -  affordance  (grasp_center)
      - approach : ,  XY  ()
      -  (, )
      - approach  (TILT_ANGLE_DEG=0)
      - grip axis:  Z   to  , 
    """
    aff_points, handle_center = cluster_affordance_points(points, labels)
    obj_center = points.mean(axis=0)

    # --- approach :  (, ) ---
    diff_xy = obj_center[:2] - handle_center[:2]

    if abs(diff_xy[0]) < 1e-6 and abs(diff_xy[1]) < 1e-6:
        print("[WARN]  XY ,  +X")
        approach_horiz = np.array([1.0, 0.0])
    elif abs(diff_xy[0]) >= abs(diff_xy[1]):
        # X   to   X 
        approach_horiz = np.array([np.sign(diff_xy[0]), 0.0])
    else:
        # Y   to   Y 
        approach_horiz = np.array([0.0, np.sign(diff_xy[1])])

    print(f"[INFO] approach : [{approach_horiz[0]:.0f}, {approach_horiz[1]:.0f}]")

    #  (TILT=0  to  )
    tilt_rad = np.radians(TILT_ANGLE_DEG)
    approach_dir = np.array([
        approach_horiz[0] * np.cos(tilt_rad),
        approach_horiz[1] * np.cos(tilt_rad),
        -np.sin(tilt_rad)
    ])
    approach_dir /= np.linalg.norm(approach_dir)

    # --- grip axis:  (Z ),  ---
    #  (, )
    grip_axis = np.array([0.0, 0.0, 1.0])

    # --- binormal = approach x grip_axis (, ) ---
    binormal = np.cross(approach_dir, grip_axis)
    binormal /= np.linalg.norm(binormal)

    #  grip_axis ( Z ,  approach )
    grip_axis = np.cross(binormal, approach_dir)
    grip_axis /= np.linalg.norm(grip_axis)

    rotation_matrix = np.column_stack([binormal, grip_axis, approach_dir])

    print(f"[INFO] Handle center: [{handle_center[0]:.5f}, {handle_center[1]:.5f}, {handle_center[2]:.5f}]")
    print(f"[INFO] Object center: [{obj_center[0]:.5f}, {obj_center[1]:.5f}, {obj_center[2]:.5f}]")
    print(f"[INFO] Approach (): [{approach_dir[0]:.4f}, {approach_dir[1]:.4f}, {approach_dir[2]:.4f}]")
    print(f"[INFO] Grip axis (): [{grip_axis[0]:.4f}, {grip_axis[1]:.4f}, {grip_axis[2]:.4f}]")
    print(f"[INFO] Binormal (): [{binormal[0]:.4f}, {binormal[1]:.4f}, {binormal[2]:.4f}]")
    print(f"[INFO] Tilt angle: {TILT_ANGLE_DEG} deg")

    return {
        'grasp_center': handle_center,
        'approach_dir': approach_dir,
        'grip_axis': grip_axis,
        'binormal': binormal,
        'rotation_matrix': rotation_matrix,
        'aff_points': aff_points,
    }


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


def create_grasp_visualization(grasp_result):
    center = grasp_result['grasp_center']
    approach = grasp_result['approach_dir']
    grip = grasp_result['grip_axis']
    R_local = grasp_result['rotation_matrix']
    geometries = []

    half_open = GRIPPER_MAX_OPENING / 2.0
    fd = GRIPPER_FINGER_DEPTH
    fw = GRIPPER_FINGER_WIDTH
    ft = GRIPPER_FINGER_THICKNESS

    # : 
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
    sphere.translate(center)
    sphere.paint_uniform_color([1.0, 1.0, 0.0])
    sphere.compute_vertex_normals()
    geometries.append(sphere)

    # Approach  ()
    arrow_len = 0.07
    arrow_start = center - approach * arrow_len
    shaft = o3d.geometry.TriangleMesh.create_cylinder(radius=0.0015, height=arrow_len * 0.75)
    shaft.translate([0, 0, arrow_len * 0.75 / 2])
    R_arrow = rotation_from_z_to_dir(approach)
    shaft.rotate(R_arrow, center=[0, 0, 0])
    shaft.translate(arrow_start)
    shaft.paint_uniform_color([0.0, 0.3, 1.0])
    shaft.compute_vertex_normals()
    geometries.append(shaft)

    cone = o3d.geometry.TriangleMesh.create_cone(radius=0.004, height=arrow_len * 0.25)
    cone.translate([0, 0, arrow_len * 0.75])
    cone.rotate(R_arrow, center=[0, 0, 0])
    cone.translate(arrow_start)
    cone.paint_uniform_color([0.0, 0.3, 1.0])
    cone.compute_vertex_normals()
    geometries.append(cone)

    #  ()
    for sign in [+1, -1]:
        finger = o3d.geometry.TriangleMesh.create_box(width=fw, height=ft, depth=fd)
        finger.translate([-fw / 2, -ft / 2 + sign * half_open, -fd])
        finger.rotate(R_local, center=[0, 0, 0])
        finger.translate(center)
        finger.paint_uniform_color([0.0, 0.8, 0.8])
        finger.compute_vertex_normals()
        geometries.append(finger)

    # Grip axis  ()
    grip_line_len = GRIPPER_MAX_OPENING + 2 * ft + 0.01
    grip_cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=0.0008, height=grip_line_len)
    grip_cyl.translate([0, 0, -grip_line_len / 2])
    R_grip = rotation_from_z_to_dir(grip)
    grip_cyl.rotate(R_grip, center=[0, 0, 0])
    grip_cyl.translate(center)
    grip_cyl.paint_uniform_color([1.0, 0.0, 1.0])
    grip_cyl.compute_vertex_normals()
    geometries.append(grip_cyl)

    # 
    grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.025)
    grasp_frame.rotate(R_local, center=[0, 0, 0])
    grasp_frame.translate(center)
    geometries.append(grasp_frame)

    return geometries


# ============================================================================
# 
# ============================================================================
def find_pt_path(type_path):
    pt_path = os.path.join(type_path, "labeled_result.pt")
    if not os.path.isfile(pt_path):
        raise FileNotFoundError(f"{type_path}  labeled_result.pt")
    return pt_path


def load_points_and_labels(pt_path):
    if torch is None:
        raise ImportError(" torch,  .pt ")

    data = torch.load(pt_path, map_location="cpu")
    points = data['points'].numpy()
    labels = data['labels'].numpy()
    print(f"[INFO] Loaded: {pt_path}")
    return points, labels


def process_type_path(type_path, visualize=False, default_threshold=DEFAULT_UNIT_THRESHOLD):
    pt_path = find_pt_path(type_path)
    raw_points, labels = load_points_and_labels(pt_path)
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
        grasp_result = compute_handle_grasp(points, labels)

        print(f"\n{'='*60}")
        print(f"   ( - )")
        print(f"  Center:     [{grasp_result['grasp_center'][0]:.5f}, "
              f"{grasp_result['grasp_center'][1]:.5f}, {grasp_result['grasp_center'][2]:.5f}]")
        print(f"  Approach:   [{grasp_result['approach_dir'][0]:.4f}, "
              f"{grasp_result['approach_dir'][1]:.4f}, {grasp_result['approach_dir'][2]:.4f}]")
        print(f"  Grip axis:  [{grasp_result['grip_axis'][0]:.4f}, "
              f"{grasp_result['grip_axis'][1]:.4f}, {grasp_result['grip_axis'][2]:.4f}]")

        ee_pos = grasp_result['grasp_center'] - grasp_result['approach_dir'] * EE_OFFSET_Z
        print(f"  EE:     [{ee_pos[0]:.5f}, {ee_pos[1]:.5f}, {ee_pos[2]:.5f}]")
        print(f"{'='*60}")

        if visualize:
            grasp_geometries = create_grasp_visualization(grasp_result)
            all_geometries = [pcd, coord_frame] + grasp_geometries
            return attach_visualization(
                {"grasp_result": grasp_result, "ee_pos": ee_pos},
                all_geometries,
                window_name=" ()",
                width=1280,
                height=720,
            )

        return {"grasp_result": grasp_result, "ee_pos": ee_pos}

    def accept_fn(payload, threshold, converted, coord_range):
        grasp_result = payload["grasp_result"]
        ee_pos = payload["ee_pos"]
        np.savez(save_path,
                 grasp_center=grasp_result['grasp_center'],
                 approach_dir=grasp_result['approach_dir'],
                 grip_axis=grasp_result['grip_axis'],
                 binormal=grasp_result['binormal'],
                 rotation_matrix=grasp_result['rotation_matrix'],
                 score=np.array(1.0),
                 ee_position=ee_pos,
                 tilt_angle=np.array(TILT_ANGLE_DEG),
                 unit_threshold=np.array(threshold),
                 unit_converted=np.array(int(converted)))
        print(f"[INFO] Saved result: {save_path}")

    return run_threshold_debug_loop(raw_points, os.path.basename(type_path), preview_fn, accept_fn, default_threshold)


def find_pull_dirs(script_dir):
    return sorted(
        os.path.join(script_dir, entry)
        for entry in os.listdir(script_dir)
        if entry.endswith("_pull") and os.path.isdir(os.path.join(script_dir, entry))
    )


# ============================================================================
# 
# ============================================================================
if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    type_paths = find_pull_dirs(script_dir)

    if not type_paths:
        raise FileNotFoundError(f"{script_dir}  *_pull ")

    visualize = True
    if len(type_paths) > 1:
        print(f"[INFO]  {len(type_paths)}  *_pull ,  Open3D  Y/Enter U/D Q/Esc/X ")

    for index, type_path in enumerate(type_paths, start=1):
        folder_name = os.path.basename(type_path)
        print(f"\n[INFO] ({index}/{len(type_paths)}) : {folder_name}")
        if not process_type_path(type_path, visualize=visualize):
            print("[INFO] ")
            break
