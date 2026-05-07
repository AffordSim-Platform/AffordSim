
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
GRIPPER_MAX_OPENING = 0.08        # 80mm (Y/)
GRIPPER_FINGER_DEPTH = 0.10       # 100mm /(Z/approach)
GRIPPER_FINGER_WIDTH = 0.02       # 20mm (X)
GRIPPER_FINGER_THICKNESS = 0.012  # 12mm (Y/)
EE_OFFSET_Z = 0.220              # fr3_hand  to  ee_frame 

# ============================================================================
# 
# ============================================================================
TILT_ANGLE_DEG = 90.0             # 90 deg = 


# ============================================================================
#  : 
# ============================================================================
def tilt_direction_upward(direction, tilt_deg=TILT_ANGLE_DEG):
    """ tilt_deg  (Z+).

    :  direction ,
    ,  tilt_deg.
     ( [0,0,-1]), .

    Args:
        direction: 3D ()
        tilt_deg:  (), =

    Returns:
        
    """
    d = np.array(direction, dtype=np.float64)
    d /= np.linalg.norm(d)

    # 
    d_horiz = np.array([d[0], d[1], 0.0])
    horiz_norm = np.linalg.norm(d_horiz)

    # , 
    if horiz_norm < 1e-6:
        return d

    d_horiz /= horiz_norm

    tilt_rad = np.radians(tilt_deg)
    #  =  * cos(tilt) +  * sin(tilt)
    # : approach "", 
    #  Z  (),  approach ,
    # , 
    # approach  = , Z""
    new_dir = d_horiz * np.cos(tilt_rad) + np.array([0.0, 0.0, -np.sin(tilt_rad)])

    norm = np.linalg.norm(new_dir)
    if norm < 1e-8:
        return d
    return new_dir / norm


# ============================================================================
# Step 1: Affordance 
# ============================================================================
def cluster_affordance_points(points, labels, affordance_label=1,
                              dbscan_eps=0.008, dbscan_min_samples=5):
    """ affordance """
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
def sample_grasp_centers(aff_points, n_samples=12, local_radius=0.003):
    """ affordance """
    aff_spread = aff_points.ptp(axis=0).max()
    centroid = aff_points.mean(axis=0)

    if aff_spread < GRIPPER_MAX_OPENING * 1.2 or len(aff_points) < 30:
        return [(centroid, aff_points, "")]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(aff_points)
    try:
        sampled_pcd = pcd.farthest_point_down_sample(n_samples)
    except (AttributeError, RuntimeError):
        idx = np.random.choice(len(aff_points), min(n_samples, len(aff_points)), replace=False)
        sampled_pcd = pcd.select_by_index(idx.tolist())
    sample_pts = np.asarray(sampled_pcd.points)

    centers = []

    kdtree = o3d.geometry.KDTreeFlann(pcd)
    for i, pt in enumerate(sample_pts):
        [k, idx, _] = kdtree.search_radius_vector_3d(pt, local_radius)
        if k < 5:
            [k, idx, _] = kdtree.search_knn_vector_3d(pt, 10)
        local_pts = aff_points[idx]
        local_center = local_pts.mean(axis=0)
        centers.append((local_center, local_pts, f"{i}"))

    print(f"[INFO]  {len(centers)}  (aff_spread={aff_spread*1000:.1f}mm, "
          f"local_r={local_radius*1000:.0f}mm)")
    return centers


# ============================================================================
# Step 3:  approach 
# ============================================================================
def compute_handle_direction(aff_points, all_points):
    """: affordance 

     (, Z=0), 
     None
    """
    if len(aff_points) < 3:
        return None
    aff_centroid = aff_points.mean(axis=0)
    obj_centroid = all_points.mean(axis=0)
    diff = aff_centroid - obj_centroid
    # 
    handle_dir = np.array([diff[0], diff[1], 0.0])
    norm = np.linalg.norm(handle_dir)
    if norm < 1e-6:
        return None
    return handle_dir / norm


def generate_approach_directions(local_aff_points, all_points, grasp_center):
    """"""
    del local_aff_points, all_points, grasp_center
    return [(np.array([0.0, 0.0, -1.0]), "")]


# ============================================================================
# Step 4:  ( : upright_score 45 deg)
# ============================================================================
def score_grasp(approach_dir, grip_axis, grasp_center,
                all_points, labels,
                gripper_opening=GRIPPER_MAX_OPENING,
                finger_depth=GRIPPER_FINGER_DEPTH,
                finger_width=GRIPPER_FINGER_WIDTH,
                finger_thickness=GRIPPER_FINGER_THICKNESS,
                handle_dir=None):
    """ (grasp_center, approach, grip_axis)  None 

    : X=binormal(), Y=grip_axis(), Z=approach()
    """
    binormal = np.cross(approach_dir, grip_axis)
    bn_norm = np.linalg.norm(binormal)
    if bn_norm < 1e-8:
        return None
    binormal /= bn_norm
    grip_axis = np.cross(binormal, approach_dir)
    grip_axis /= np.linalg.norm(grip_axis)

    R = np.column_stack([binormal, grip_axis, approach_dir])
    all_local = (all_points - grasp_center) @ R
    aff_mask = (labels == 1)
    non_aff_local = all_local[~aff_mask]
    aff_local = all_local[aff_mask]

    half_open = gripper_opening / 2.0
    half_width = finger_width / 2.0

    def in_box(pts, xr, yr, zr):
        return ((pts[:, 0] >= xr[0]) & (pts[:, 0] <= xr[1]) &
                (pts[:, 1] >= yr[0]) & (pts[:, 1] <= yr[1]) &
                (pts[:, 2] >= zr[0]) & (pts[:, 2] <= zr[1]))

    #  ( aff )  to  
    n_closing = int(in_box(non_aff_local,
                           [-half_width, half_width],
                           [-half_open, half_open],
                           [-finger_depth, 0]).sum())
    if n_closing > 3:
        return None

    # 
    n_left = int(in_box(non_aff_local,
                        [-half_width, half_width],
                        [half_open, half_open + finger_thickness],
                        [-finger_depth, 0]).sum())
    n_right = int(in_box(non_aff_local,
                         [-half_width, half_width],
                         [-half_open - finger_thickness, -half_open],
                         [-finger_depth, 0]).sum())

    # 
    n_palm = int(in_box(non_aff_local,
                        [-half_width, half_width],
                        [-half_open - finger_thickness, half_open + finger_thickness],
                        [-finger_depth - 0.02, -finger_depth]).sum())

    # 
    n_approach = int(in_box(all_local,
                            [-half_width, half_width],
                            [-half_open - finger_thickness, half_open + finger_thickness],
                            [-finger_depth - 0.06, -finger_depth - 0.02]).sum())

    n_finger_col = n_left + n_right
    collision_score = max(0.0, 1.0 - n_finger_col * 0.2 - n_closing * 0.3 - n_palm * 0.15)
    clearance_score = max(0.0, 1.0 - n_approach * 0.05)

    # 
    in_grip = in_box(aff_local,
                     [-half_width, half_width],
                     [-half_open, half_open],
                     [-finger_depth, 0])
    n_aff_in_grip = int(in_grip.sum())
    coverage_score = n_aff_in_grip / max(len(aff_local), 1)

    # 
    if n_aff_in_grip > 0:
        aff_in = aff_local[in_grip]
        aff_center_y = aff_in[:, 1].mean()
        aff_spread_y = aff_in[:, 1].ptp()
    else:
        aff_center_y = aff_local[:, 1].mean()
        aff_spread_y = aff_local[:, 1].ptp()

    centering = 1.0 - min(abs(aff_center_y) / (half_open + 1e-6), 1.0)
    narrowness = 1.0 - min(aff_spread_y / gripper_opening, 1.0)
    antipodal_score = centering * 0.5 + narrowness * 0.5

    if aff_spread_y > gripper_opening and n_aff_in_grip < 3:
        return None

    #  :  approach  
    target_tilt_rad = np.radians(TILT_ANGLE_DEG)
    target_z = -np.sin(target_tilt_rad)  #  approach  Z  (=)

    #  approach 
    actual_z = approach_dir[2]
    #  ()
    tilt_error = abs(actual_z - target_z)
    #  [0, 1], 01
    tilt_score = max(0.0, 1.0 - tilt_error * 2.0)

    # grip_axis  binormal  ()
    grip_z = abs(grip_axis[2])
    binormal_z = abs(binormal[2])
    orientation_penalty = 0.5 * (grip_z + binormal_z)
    upright_score = tilt_score * (1.0 - 0.3 * orientation_penalty)

    # 
    if n_aff_in_grip > 0:
        aff_in = aff_local[in_grip]
        dist_to_wall = half_open - np.abs(aff_in[:, 1])
        contact_score = np.mean(np.clip(1.0 - dist_to_wall / half_open, 0.0, 1.0))
    else:
        contact_score = 0.0

    #  : 
    #  approach  (Z0), 
    horiz_penalty = 0.0
    approach_elevation = np.degrees(np.arcsin(np.clip(-approach_dir[2], -1, 1)))
    if approach_elevation < 15.0:  # 15 deg, 
        horiz_penalty = 0.5 * (1.0 - approach_elevation / 15.0)

    #   
    # 1) approach 
    # 2) grip axis () , 
    handle_align_score = 0.5  #  ()
    grip_handle_score = 0.5   # grip axis 
    if handle_dir is not None:
        # approach 
        approach_horiz = np.array([approach_dir[0], approach_dir[1], 0.0])
        ah_norm = np.linalg.norm(approach_horiz)
        if ah_norm > 1e-6:
            approach_horiz /= ah_norm
            alignment = np.dot(approach_horiz, -handle_dir)
            handle_align_score = (alignment + 1.0) / 2.0
        else:
            handle_align_score = 0.3

        #  grip axis : grip axis 
        #  grip axis , , grip axis 
        #  grip axis   handle_dir  (dot ~= 0) 
        #  grip axis  handle_dir  (dot ~= +/-1) 
        # : ,  grip axis perp handle_dir
        grip_horiz = np.array([grip_axis[0], grip_axis[1], 0.0])
        gh_norm = np.linalg.norm(grip_horiz)
        if gh_norm > 1e-6:
            grip_horiz /= gh_norm
            # grip  handle_dir ,  0  ()
            grip_dot = abs(np.dot(grip_horiz, handle_dir))
            # grip_dot ~= 0  to    to   ( 1.0)
            # grip_dot ~= 1  to    to   ( 0.0)
            grip_handle_score = 1.0 - grip_dot
        else:
            grip_handle_score = 0.3

    total = (collision_score * 0.12 +
             clearance_score * 0.05 +
             coverage_score * 0.14 +
             antipodal_score * 0.08 +
             upright_score * 0.16 +
             contact_score * 0.08 +
             handle_align_score * 0.15 +    # approach 
             grip_handle_score * 0.17 -      #  grip axis  ()
             horiz_penalty * 0.05)
    total = max(0.0, total)

    return {
        'total_score': total,
        'collision_score': collision_score,
        'clearance_score': clearance_score,
        'coverage_score': coverage_score,
        'antipodal_score': antipodal_score,
        'upright_score': upright_score,
        'contact_score': contact_score,
        'handle_align_score': handle_align_score,
        'grip_handle_score': grip_handle_score,
        'grip_axis': grip_axis,
        'n_collisions': n_finger_col + n_closing + n_palm,
        'n_closing': n_closing,
        'aff_width': aff_spread_y,
        'n_aff_in_grip': n_aff_in_grip,
        'approach_elevation': approach_elevation,
    }


# ============================================================================
# Step 5: 
# ============================================================================
def compute_best_grasp(points, labels):
    """"""
    aff_points, global_centroid = cluster_affordance_points(points, labels)
    centers = sample_grasp_centers(aff_points)

    #   ()
    handle_dir = compute_handle_direction(aff_points, points)
    if handle_dir is not None:
        print(f"[INFO] Handle direction: [{handle_dir[0]:.3f}, {handle_dir[1]:.3f}, {handle_dir[2]:.3f}]")

    results = []
    fixed_grip_axis = np.array([1.0, 0.0, 0.0])

    for center, local_pts, center_name in centers:
        approach_dirs = generate_approach_directions(local_pts, points, center)

        for approach_dir, approach_name in approach_dirs:
            info = score_grasp(approach_dir, fixed_grip_axis, center, points, labels,
                               handle_dir=handle_dir)
            if info is None:
                continue

            info['approach_dir'] = approach_dir
            info['grasp_center'] = center
            info['source'] = f"{center_name}/{approach_name}"
            results.append(info)

    if not results:
        print("[WARN] No valid candidates found; using fallback")
        approach = np.array([0.0, 0.0, -1.0])
        grip = np.array([1.0, 0.0, 0.0])
        binormal = np.cross(approach, grip)
        binormal /= np.linalg.norm(binormal)
        return {
            'grasp_center': global_centroid,
            'approach_dir': approach,
            'grip_axis': grip,
            'binormal': binormal,
            'rotation_matrix': np.column_stack([binormal, grip, approach]),
            'score': 0.0,
            'source': 'fallback(vertical_down)',
            'all_candidates': [],
            'aff_points': aff_points,
        }

    results.sort(key=lambda x: x['total_score'], reverse=True)

    #  Top 20
    print(f"\n{'='*140}")
    print(f"  {'':<40} {'':>6} {'':>6} {'':>6} {'':>6} "
          f"{'':>6} {'':>6} {'':>6} {'':>6} {'':>6} {'':>5} {'':>5} {'aff':>5} {'mm':>6} {'':>6}")
    print(f"  {'-'*134}")
    for r in results[:20]:
        print(f"  {r['source']:<40} {r['total_score']:>6.3f} "
              f"{r['collision_score']:>6.3f} "
              f"{r['clearance_score']:>6.3f} "
              f"{r['coverage_score']:>6.3f} "
              f"{r['antipodal_score']:>6.3f} "
              f"{r['upright_score']:>6.3f} "
              f"{r['contact_score']:>6.3f} "
              f"{r.get('handle_align_score', 0):>6.3f} "
              f"{r.get('grip_handle_score', 0):>6.3f} "
              f"{r['n_collisions']:>5} "
              f"{r['n_closing']:>5} "
              f"{r['n_aff_in_grip']:>5} "
              f"{r['aff_width']*1000:>6.1f} "
              f"{r.get('approach_elevation', 0):>5.1f} deg")
    if len(results) > 20:
        print(f"  ...  {len(results)-20} ")
    print(f"{'='*120}")
    print(f"[INFO] Valid candidates: {len(results)}")

    best = results[0]
    approach_dir = best['approach_dir']
    grip_axis = best['grip_axis']
    binormal = np.cross(approach_dir, grip_axis)
    binormal /= np.linalg.norm(binormal)
    grip_axis = np.cross(binormal, approach_dir)
    grip_axis /= np.linalg.norm(grip_axis)

    #   approach 
    elevation = np.degrees(np.arcsin(np.clip(-approach_dir[2], -1, 1)))
    print(f"[INFO]  approach : {elevation:.1f} deg (target: {TILT_ANGLE_DEG} deg)")

    return {
        'grasp_center': best['grasp_center'],
        'approach_dir': approach_dir,
        'grip_axis': grip_axis,
        'binormal': binormal,
        'rotation_matrix': np.column_stack([binormal, grip_axis, approach_dir]),
        'score': best['total_score'],
        'source': best['source'],
        'all_candidates': results,
        'aff_points': aff_points,
    }


# ============================================================================
# Step 6: 
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
    geometries = []

    half_open = GRIPPER_MAX_OPENING / 2.0
    fd = GRIPPER_FINGER_DEPTH
    fw = GRIPPER_FINGER_WIDTH
    ft = GRIPPER_FINGER_THICKNESS

    binormal = grasp_result['binormal']
    R_local = np.column_stack([binormal, grip, approach])

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
    # grip_line_len = GRIPPER_MAX_OPENING + 2 * ft + 0.01
    # grip_cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=0.0008, height=grip_line_len)
    # grip_cyl.translate([0, 0, -grip_line_len / 2])
    # R_grip = rotation_from_z_to_dir(grip)
    # grip_cyl.rotate(R_grip, center=[0, 0, 0])
    # grip_cyl.translate(center)
    # grip_cyl.paint_uniform_color([1.0, 0.0, 1.0])
    # grip_cyl.compute_vertex_normals()
    # geometries.append(grip_cyl)

    # 
    # grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.025)
    # grasp_frame.rotate(R_local, center=[0, 0, 0])
    # grasp_frame.translate(center)
    # geometries.append(grasp_frame)

    #  :  ()
    elevation = np.degrees(np.arcsin(np.clip(-approach[2], -1, 1)))
    print(f"[VIS] approach : {elevation:.1f} deg")

    return geometries


# ============================================================================
# 
# ============================================================================
def find_pt_path(type_path):
    pt_path = os.path.join(type_path, "labeled_result_hang.pt")
    if not os.path.isfile(pt_path):
        raise FileNotFoundError(f"{type_path}  labeled_result_hang.pt")
    return pt_path


def load_points_and_labels(pt_path):
    if torch is None:
        raise ImportError(" torch,  .pt ")

    data = torch.load(pt_path, map_location="cpu")
    points = data['points'].numpy()
    labels = data['labels'].numpy()
    print(f"[INFO] Loaded: {pt_path}")
    return points, labels


def process_type_path(type_path, visualize=True, default_threshold=DEFAULT_UNIT_THRESHOLD):
    pt_path = find_pt_path(type_path)
    raw_points, labels = load_points_and_labels(pt_path)
    save_path = os.path.join(type_path, "grasp_result_hang.npz")

    def preview_fn(points, threshold, converted, coord_range):
        print(f"[INFO] Points: {len(points)}, label=0: {(labels==0).sum()}, label=1: {(labels==1).sum()}")
        print(f"[INFO] Mode: vertical grasp")

        colors = np.zeros((len(labels), 3))
        colors[labels == 0] = [0.0, 1.0, 0.0]
        colors[labels == 1] = [1.0, 0.0, 0.0]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
        grasp_result = compute_best_grasp(points, labels)

        print(f"\n{'='*60}")
        print("  Grasp result")
        print(f"  Center:     [{grasp_result['grasp_center'][0]:.5f}, "
              f"{grasp_result['grasp_center'][1]:.5f}, {grasp_result['grasp_center'][2]:.5f}]")
        print(f"  Approach:   [{grasp_result['approach_dir'][0]:.4f}, "
              f"{grasp_result['approach_dir'][1]:.4f}, {grasp_result['approach_dir'][2]:.4f}]")
        print(f"  Grip axis:  [{grasp_result['grip_axis'][0]:.4f}, "
              f"{grasp_result['grip_axis'][1]:.4f}, {grasp_result['grip_axis'][2]:.4f}]")
        print(f"  Score:      {grasp_result['score']:.4f}")
        print(f"  Source:     {grasp_result['source']}")

        elevation = np.degrees(np.arcsin(np.clip(-grasp_result['approach_dir'][2], -1, 1)))
        print(f"  Elevation:  {elevation:.1f} deg (target: {TILT_ANGLE_DEG} deg)")

        ee_pos = grasp_result['grasp_center'] - grasp_result['approach_dir'] * EE_OFFSET_Z
        print(f"  EE:     [{ee_pos[0]:.5f}, {ee_pos[1]:.5f}, {ee_pos[2]:.5f}]")
        print(f"{'='*60}")

        if visualize:
            grasp_geometries = create_grasp_visualization(grasp_result)
            # all_geometries = [pcd, coord_frame] + grasp_geometries
            all_geometries = [pcd] + grasp_geometries
            return attach_visualization(
                {"grasp_result": grasp_result, "ee_pos": ee_pos},
                all_geometries,
                window_name=f"Affordance +  ({TILT_ANGLE_DEG} deg)",
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
                 score=np.array(grasp_result['score']),
                 ee_position=ee_pos,
                 tilt_angle=np.array(TILT_ANGLE_DEG),
                 unit_threshold=np.array(threshold),
                 unit_converted=np.array(int(converted)))
        print(f"[INFO] Saved result: {save_path}")

    return run_threshold_debug_loop(raw_points, os.path.basename(type_path), preview_fn, accept_fn, default_threshold)


def find_pick_dirs(script_dir):
    return sorted(
        os.path.join(script_dir, entry)
        for entry in os.listdir(script_dir)
        if entry.endswith("_pick")
        and os.path.isdir(os.path.join(script_dir, entry))
        and os.path.isfile(os.path.join(script_dir, entry, "labeled_result_hang.pt"))
    )


# ============================================================================
# 
# ============================================================================
if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    type_paths = find_pick_dirs(script_dir)

    if not type_paths:
        raise FileNotFoundError(f"{script_dir}  labeled_result_hang.pt  *_pick ")

    if len(type_paths) > 1:
        print(f"[INFO]  {len(type_paths)}  labeled_result_hang.pt  *_pick ,  Open3D  Y/Enter U/D Q/Esc/X ")

    for index, type_path in enumerate(type_paths, start=1):
        folder_name = os.path.basename(type_path)
        print(f"\n[INFO] ({index}/{len(type_paths)}) : {folder_name}")
        if not process_type_path(type_path, visualize=True):
            print("[INFO] ")
            break
