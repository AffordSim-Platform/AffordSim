"""
5

 approach  XY  X  4 :
  Q0: 0-90 deg, Q1: 90-180 deg, Q2: 180-270 deg, Q3: 270-360 deg
 45 deg 
 5 :  Z 
"""

import numpy as np
import open3d as o3d
import os
from unit_threshold_helper import DEFAULT_UNIT_THRESHOLD, attach_visualization, run_threshold_debug_loop

try:
    import torch
except ImportError:
    torch = None

#  pick 
from check_pcdata_pick import (
    GRIPPER_MAX_OPENING, GRIPPER_FINGER_DEPTH,
    GRIPPER_FINGER_WIDTH, GRIPPER_FINGER_THICKNESS,
    EE_OFFSET_Z, TILT_ANGLE_DEG,
    cluster_affordance_points, sample_grasp_centers,
    generate_approach_directions, score_grasp,
    rotation_from_z_to_dir,
)

# 4  + 
QUADRANT_NAMES = ["Q0(0-90 deg)", "Q1(90-180 deg)", "Q2(180-270 deg)", "Q3(270-360 deg)", "(Z-)"]
QUADRANT_COLORS = [
    [1.0, 0.0, 0.0],   # Q0: 
    [0.0, 0.0, 1.0],   # Q1: 
    [1.0, 0.5, 0.0],   # Q2: 
    [0.5, 0.0, 1.0],   # Q3: 
    [0.0, 0.8, 0.8],   # : 
]


def get_approach_azimuth(approach_dir):
    """ approach  XY  X  (0-360 deg)"""
    angle = np.degrees(np.arctan2(approach_dir[1], approach_dir[0]))
    if angle < 0:
        angle += 360.0
    return angle


def get_quadrant(azimuth):
    """ 0-3"""
    return int(azimuth // 90) % 4


def compute_4dir_grasps(points, labels):
    """ 4 

    Returns:
        dict: {
            'grasps': list of 4 grasp results (None if no valid candidate),
            'aff_points': affordance points,
            'all_candidates': all scored candidates,
        }
    """
    aff_points, global_centroid = cluster_affordance_points(points, labels)
    centers = sample_grasp_centers(aff_points)

    # 
    all_results = []
    n_rotations = 12

    for center, local_pts, center_name in centers:
        approach_dirs = generate_approach_directions(local_pts, points, center)

        for approach_dir, approach_name in approach_dirs:
            arb = np.array([1, 0, 0]) if abs(approach_dir[0]) < 0.9 else np.array([0, 1, 0])
            base_grip = np.cross(approach_dir, arb)
            base_grip /= np.linalg.norm(base_grip)

            for i in range(n_rotations):
                angle = i * np.pi / n_rotations
                grip = (base_grip * np.cos(angle) +
                        np.cross(approach_dir, base_grip) * np.sin(angle) +
                        approach_dir * np.dot(approach_dir, base_grip) * (1 - np.cos(angle)))
                grip /= np.linalg.norm(grip)

                info = score_grasp(approach_dir, grip, center, points, labels)
                if info is None:
                    continue

                info['approach_dir'] = approach_dir
                info['grasp_center'] = center
                info['source'] = f"{center_name}/{approach_name}@{int(np.degrees(angle))}"

                # 
                azimuth = get_approach_azimuth(approach_dir)
                info['azimuth'] = azimuth
                info['quadrant'] = get_quadrant(azimuth)

                all_results.append(info)

    print(f"[INFO] Total candidates: {len(all_results)}")

    # 
    quadrant_results = [[] for _ in range(4)]
    for r in all_results:
        quadrant_results[r['quadrant']].append(r)

    for q in range(4):
        # 
        q_center = q * 90.0 + 45.0
        # :  -  ()
        def sort_key(r, qc=q_center):
            az = r['azimuth']
            #  ( 0/360 )
            diff = abs(az - qc)
            if diff > 180:
                diff = 360 - diff
            #  45 deg  ( 0.3)
            azimuth_penalty = (diff / 45.0) * 0.3
            return r['total_score'] - azimuth_penalty
        quadrant_results[q].sort(key=sort_key, reverse=True)
        print(f"  {QUADRANT_NAMES[q]}: {len(quadrant_results[q])} ")

    # 
    grasps = []
    for q in range(4):
        q_center_azimuth = q * 90.0 + 45.0

        if not quadrant_results[q]:
            # fallback: 
            print(f"[WARN] {QUADRANT_NAMES[q]} ,  fallback")
            az_rad = np.radians(q_center_azimuth)
            tilt_rad = np.radians(TILT_ANGLE_DEG)
            approach = np.array([
                np.cos(az_rad) * np.cos(tilt_rad),
                np.sin(az_rad) * np.cos(tilt_rad),
                -np.sin(tilt_rad)
            ])
            approach /= np.linalg.norm(approach)
        else:
            best = quadrant_results[q][0]
            approach = best['approach_dir'].copy()
            approach /= np.linalg.norm(approach)

        #   grip_axis:  Z  approach 
        # 
        z_up = np.array([0.0, 0.0, 1.0])
        grip_axis = z_up - np.dot(z_up, approach) * approach
        grip_norm = np.linalg.norm(grip_axis)
        if grip_norm < 1e-6:
            # approach  (, )
            grip_axis = np.array([0.0, 1.0, 0.0])
            grip_axis = grip_axis - np.dot(grip_axis, approach) * approach
            grip_axis /= np.linalg.norm(grip_axis)
        else:
            grip_axis /= grip_norm

        binormal = np.cross(grip_axis, approach)  # : X = Y x Z
        binormal /= np.linalg.norm(binormal)

        if not quadrant_results[q]:
            grasps.append({
                'grasp_center': global_centroid,
                'approach_dir': approach,
                'grip_axis': grip_axis,
                'binormal': binormal,
                'rotation_matrix': np.column_stack([binormal, grip_axis, approach]),
                'score': 0.0,
                'source': f'fallback({QUADRANT_NAMES[q]})',
                'azimuth': q_center_azimuth,
                'quadrant': q,
            })
        else:
            elevation = np.degrees(np.arcsin(np.clip(-approach[2], -1, 1)))
            grasps.append({
                'grasp_center': best['grasp_center'],
                'approach_dir': approach,
                'grip_axis': grip_axis,
                'binormal': binormal,
                'rotation_matrix': np.column_stack([binormal, grip_axis, approach]),
                'score': best['total_score'],
                'source': best['source'],
                'azimuth': best['azimuth'],
                'quadrant': q,
                'elevation': elevation,
            })

    #  5:  approach=[0,0,-1]
    # grip  affordance  PCA  (, )
    vertical_approach = np.array([0.0, 0.0, -1.0])

    #  affordance  XY  2D PCA
    aff_xy = aff_points[:, :2]
    aff_xy_centered = aff_xy - aff_xy.mean(axis=0)
    cov_2d = np.cov(aff_xy_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov_2d)
    # eigvecs[:,1]  =  XY 
    # eigvecs[:,0]  =  = 
    handle_dir_2d = eigvecs[:, 1]  # 
    perp_dir_2d = eigvecs[:, 0]    # 
    print(f"  [DEBUG] 2D PCA : {eigvals}")
    print(f"  [DEBUG] (XY): [{handle_dir_2d[0]:.4f}, {handle_dir_2d[1]:.4f}]")
    print(f"  [DEBUG] (XY): [{perp_dir_2d[0]:.4f}, {perp_dir_2d[1]:.4f}]")
    # grip_axis =  ()
    #  PCA snap 
    grip_axis = np.array([perp_dir_2d[0], perp_dir_2d[1], 0.0])
    grip_axis /= np.linalg.norm(grip_axis)
    #  X  Y  < 20 degsnap 
    SNAP_THRESHOLD = np.cos(np.radians(20.0))  # ~0.94
    axis_candidates = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
    for ax in axis_candidates:
        if abs(np.dot(grip_axis, ax)) > SNAP_THRESHOLD:
            grip_axis = ax * np.sign(np.dot(grip_axis, ax))
            print(f"  [DEBUG] grip_axis snapped to [{grip_axis[0]:.1f}, {grip_axis[1]:.1f}, 0.0]")
            break
    binormal = np.cross(grip_axis, vertical_approach)  # 
    binormal /= np.linalg.norm(binormal)

    best_score = max((r['total_score'] for r in all_results), default=0.0)

    print(f"  {QUADRANT_NAMES[4]}:  "
          f"(gripPCA [{grip_axis[0]:.3f}, {grip_axis[1]:.3f}])")

    grasps.append({
        'grasp_center': global_centroid,
        'approach_dir': vertical_approach,
        'grip_axis': grip_axis,
        'binormal': binormal,
        'rotation_matrix': np.column_stack([binormal, grip_axis, vertical_approach]),
        'score': best_score * 0.8,
        'source': 'vertical(PCA)',
        'azimuth': 0.0,
        'quadrant': 4,
        'elevation': 90.0,
    })

    #  5 
    print(f"\n{'='*90}")
    print(f"  {'':<20} {'':>8} {'':>8} {'':>8} {'':<40}")
    print(f"  {'-'*84}")
    for g in grasps:
        elev = np.degrees(np.arcsin(np.clip(-g['approach_dir'][2], -1, 1)))
        print(f"  {QUADRANT_NAMES[g['quadrant']]:<20} "
              f"{g['azimuth']:>7.1f} deg "
              f"{elev:>7.1f} deg "
              f"{g['score']:>8.4f} "
              f"{g['source']:<40}")
    print(f"{'='*90}")

    return {
        'grasps': grasps,
        'aff_points': aff_points,
        'all_candidates': all_results,
    }


def create_4dir_visualization(grasps_result, show_gripper=True):
    """ 4 

    Args:
        grasps_result: compute_4dir_grasps 
        show_gripper:  ( True)
    """
    geometries = []

    for i, grasp in enumerate(grasps_result['grasps']):
        #  (5, index=4)
        # if grasp['quadrant'] != 4:
        #     continue
        center = grasp['grasp_center']
        approach = grasp['approach_dir']
        grip = grasp['grip_axis']
        binormal = grasp['binormal']
        color = QUADRANT_COLORS[i] if i < len(QUADRANT_COLORS) else [0.5, 0.5, 0.5]

        R_local = np.column_stack([binormal, grip, approach])

        # 
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
        sphere.translate(center)
        sphere.paint_uniform_color([1.0, 1.0, 0.0])
        sphere.compute_vertex_normals()
        geometries.append(sphere)

        # Approach 
        arrow_len = 0.07
        arrow_start = center - approach * arrow_len
        shaft = o3d.geometry.TriangleMesh.create_cylinder(radius=0.0015, height=arrow_len * 0.75)
        shaft.translate([0, 0, arrow_len * 0.75 / 2])
        R_arrow = rotation_from_z_to_dir(approach)
        shaft.rotate(R_arrow, center=[0, 0, 0])
        shaft.translate(arrow_start)
        shaft.paint_uniform_color(color)
        shaft.compute_vertex_normals()
        geometries.append(shaft)

        cone = o3d.geometry.TriangleMesh.create_cone(radius=0.004, height=arrow_len * 0.25)
        cone.translate([0, 0, arrow_len * 0.75])
        cone.rotate(R_arrow, center=[0, 0, 0])
        cone.translate(arrow_start)
        cone.paint_uniform_color(color)
        cone.compute_vertex_normals()
        geometries.append(cone)

        #  ()
        if show_gripper:
            half_open = GRIPPER_MAX_OPENING / 2.0
            fd = GRIPPER_FINGER_DEPTH
            fw = GRIPPER_FINGER_WIDTH
            ft = GRIPPER_FINGER_THICKNESS

            for sign in [+1, -1]:
                finger = o3d.geometry.TriangleMesh.create_box(width=fw, height=ft, depth=fd)
                finger.translate([-fw / 2, -ft / 2 + sign * half_open, -fd])
                finger.rotate(R_local, center=[0, 0, 0])
                finger.translate(center)
                finger.paint_uniform_color(color)
                finger.compute_vertex_normals()
                geometries.append(finger)

        #  ()
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.015)
        frame.rotate(R_local, center=[0, 0, 0])
        frame.translate(center)
        geometries.append(frame)

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


def find_pick_dirs(script_dir):
    return sorted(
        os.path.join(script_dir, entry)
        for entry in os.listdir(script_dir)
        if entry.endswith("_pick") and os.path.isdir(os.path.join(script_dir, entry))
    )


def process_type_path(type_path, show_gripper=True, default_threshold=DEFAULT_UNIT_THRESHOLD):
    pt_path = find_pt_path(type_path)
    raw_points, labels = load_points_and_labels(pt_path)
    save_path = os.path.join(type_path, "grasp_result_4dir.npz")

    def preview_fn(points, threshold, converted, coord_range):
        print(f"[INFO] Points: {len(points)}, label=0: {(labels==0).sum()}, label=1: {(labels==1).sum()}")
        print(f"[INFO] Tilt angle: {TILT_ANGLE_DEG} deg")

        colors = np.zeros((len(labels), 3))
        colors[labels == 0] = [0.0, 1.0, 0.0]
        colors[labels == 1] = [1.0, 0.0, 0.0]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
        result_4dir = compute_4dir_grasps(points, labels)

        for i, g in enumerate(result_4dir['grasps']):
            elev = np.degrees(np.arcsin(np.clip(-g['approach_dir'][2], -1, 1)))
            ee_pos = g['grasp_center'] - g['approach_dir'] * EE_OFFSET_Z
            print(f"\n  [{QUADRANT_NAMES[g['quadrant']]}]")
            print(f"    Center:   [{g['grasp_center'][0]:.5f}, {g['grasp_center'][1]:.5f}, {g['grasp_center'][2]:.5f}]")
            print(f"    Approach: [{g['approach_dir'][0]:.4f}, {g['approach_dir'][1]:.4f}, {g['approach_dir'][2]:.4f}]")
            print(f"    Azimuth: {g['azimuth']:.1f} deg  elevation: {elev:.1f} deg")
            print(f"    Score:    {g['score']:.4f}")
            print(f"    EE:   [{ee_pos[0]:.5f}, {ee_pos[1]:.5f}, {ee_pos[2]:.5f}]")

        grasp_geometries = create_4dir_visualization(result_4dir, show_gripper=show_gripper)
        all_geometries = [pcd, coord_frame] + grasp_geometries

        print("\n[VIS] :")
        color_names = ['', '', '', '', '']
        for i in range(len(result_4dir['grasps'])):
            print(f"  {QUADRANT_NAMES[i]}: {color_names[i]}")

        return attach_visualization(
            {"result_4dir": result_4dir},
            all_geometries,
            window_name="5 (4+)",
            width=1280,
            height=720,
        )

    def accept_fn(payload, threshold, converted, coord_range):
        result_4dir = payload["result_4dir"]
        save_dict = {}
        for i, g in enumerate(result_4dir['grasps']):
            prefix = f"q{i}_"
            ee_pos = g['grasp_center'] - g['approach_dir'] * EE_OFFSET_Z
            save_dict[prefix + 'grasp_center'] = g['grasp_center']
            save_dict[prefix + 'approach_dir'] = g['approach_dir']
            save_dict[prefix + 'grip_axis'] = g['grip_axis']
            save_dict[prefix + 'binormal'] = g['binormal']
            save_dict[prefix + 'rotation_matrix'] = g['rotation_matrix']
            save_dict[prefix + 'score'] = np.array(g['score'])
            save_dict[prefix + 'azimuth'] = np.array(g['azimuth'])
            save_dict[prefix + 'ee_position'] = ee_pos
        save_dict['tilt_angle'] = np.array(TILT_ANGLE_DEG)
        save_dict['n_directions'] = np.array(len(result_4dir['grasps']))
        save_dict['unit_threshold'] = np.array(threshold)
        save_dict['unit_converted'] = np.array(int(converted))
        np.savez(save_path, **save_dict)
        print(f"\n[INFO] {len(result_4dir['grasps'])}: {save_path}")

    return run_threshold_debug_loop(raw_points, os.path.basename(type_path), preview_fn, accept_fn, default_threshold)


# ============================================================================
# 
# ============================================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="4")
    parser.add_argument("--no-gripper", action="store_true", help="")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    type_paths = find_pick_dirs(script_dir)

    if not type_paths:
        raise FileNotFoundError(f"{script_dir}  *_pick ")

    if len(type_paths) > 1:
        print(f"[INFO]  {len(type_paths)}  *_pick ,  Open3D  Y/Enter U/D Q/Esc/X ")

    for index, type_path in enumerate(type_paths, start=1):
        folder_name = os.path.basename(type_path)
        print(f"\n[INFO] ({index}/{len(type_paths)}) : {folder_name}")
        if not process_type_path(type_path, show_gripper=not args.no_gripper):
            print("[INFO] ")
            break
