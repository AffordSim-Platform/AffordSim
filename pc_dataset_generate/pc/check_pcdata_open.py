"""
微波炉把手抓取位姿生成 —— 简化版

核心思路:
  1. 聚类 affordance 点 → 找到把手中心
  2. 固定抓取方向: approach 投影到 XY 平面必须正对物体(从把手指向物体中心)
  3. approach 从斜上方 45° 接近, grip axis 保持水平且垂直于 approach 的水平投影
  4. 不打分, 直接输出
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
# 夹爪参数 (米)
# ============================================================================
GRIPPER_MAX_OPENING = 0.08        # 80mm
GRIPPER_FINGER_DEPTH = 0.10       # 100mm
GRIPPER_FINGER_WIDTH = 0.02       # 20mm
GRIPPER_FINGER_THICKNESS = 0.012  # 12mm
EE_OFFSET_Z = 0.220              # fr3_hand → ee_frame 偏移

TILT_ANGLE_DEG = 45.0            # approach 与水平面的夹角 (度)
                                   # 45° = 从斜上方 45° 接近把手


# ============================================================================
# Step 1: Affordance 点聚类 → 找把手中心
# ============================================================================
def cluster_affordance_points(points, labels, affordance_label=1,
                              dbscan_eps=0.008, dbscan_min_samples=5):
    """提取 affordance 点并聚类, 返回最大簇的点和质心."""
    aff_mask = (labels == affordance_label)
    aff_points = points[aff_mask]

    if len(aff_points) < 3:
        print(f"[WARN] affordance 点太少 ({len(aff_points)}), 使用全部")
        centroid = aff_points.mean(axis=0) if len(aff_points) > 0 else points.mean(axis=0)
        return aff_points, centroid

    if len(aff_points) < 20:
        print(f"[INFO] affordance 点较少 ({len(aff_points)}), 跳过聚类")
        return aff_points, aff_points.mean(axis=0)

    aff_pcd = o3d.geometry.PointCloud()
    aff_pcd.points = o3d.utility.Vector3dVector(aff_points)
    cluster_labels = np.array(aff_pcd.cluster_dbscan(
        eps=dbscan_eps, min_points=dbscan_min_samples, print_progress=False))

    if cluster_labels.max() < 0:
        print("[WARN] DBSCAN 未发现聚类, 使用全部 affordance 点")
        return aff_points, aff_points.mean(axis=0)

    unique_labels, counts = np.unique(cluster_labels[cluster_labels >= 0], return_counts=True)
    largest_label = unique_labels[np.argmax(counts)]
    cluster_mask = (cluster_labels == largest_label)
    cluster_points = aff_points[cluster_mask]

    print(f"[INFO] DBSCAN: {len(unique_labels)} 个簇, 最大簇 {np.max(counts)} 个点 "
          f"(共 {len(aff_points)} 个 affordance 点)")

    return cluster_points, cluster_points.mean(axis=0)


# ============================================================================
# Step 2: 计算固定抓取位姿
# ============================================================================
def compute_handle_grasp(points, labels):
    """
    计算微波炉把手的抓取位姿.

    策略:
      - 聚类 affordance 点得到把手中心 (grasp_center)
      - approach 方向: 从把手中心指向物体整体中心, 只取 XY 投影 (保证正对)
      - 根据 TILT_ANGLE_DEG 施加倾斜
      - grip axis: 水平面内垂直于 approach 的方向 (保证正着抓)
    """
    aff_points, handle_center = cluster_affordance_points(points, labels)
    obj_center = points.mean(axis=0)

    # --- approach 方向: 对齐到最近的坐标轴 (保证正着抓, 无夹角) ---
    diff_xy = obj_center[:2] - handle_center[:2]

    if abs(diff_xy[0]) < 1e-6 and abs(diff_xy[1]) < 1e-6:
        print("[WARN] 把手与物体中心 XY 重合, 使用默认方向 +X")
        approach_horiz = np.array([1.0, 0.0])
    elif abs(diff_xy[0]) >= abs(diff_xy[1]):
        # X 方向差值更大 → 对齐到 X 轴
        approach_horiz = np.array([np.sign(diff_xy[0]), 0.0])
    else:
        # Y 方向差值更大 → 对齐到 Y 轴
        approach_horiz = np.array([0.0, np.sign(diff_xy[1])])

    print(f"[INFO] approach 对齐到坐标轴: [{approach_horiz[0]:.0f}, {approach_horiz[1]:.0f}]")

    # 施加倾斜角
    tilt_rad = np.radians(TILT_ANGLE_DEG)
    approach_dir = np.array([
        approach_horiz[0] * np.cos(tilt_rad),
        approach_horiz[1] * np.cos(tilt_rad),
        -np.sin(tilt_rad)
    ])
    approach_dir /= np.linalg.norm(approach_dir)

    # --- grip axis: 水平面内垂直于 approach 的方向 ---
    # 绕 Z 轴旋转 90°: (ax, ay) → (-ay, ax)
    grip_axis = np.array([-approach_horiz[1], approach_horiz[0], 0.0])
    grip_axis /= np.linalg.norm(grip_axis)

    # --- binormal ---
    binormal = np.cross(approach_dir, grip_axis)
    binormal /= np.linalg.norm(binormal)

    # 重新正交化
    grip_axis = np.cross(binormal, approach_dir)
    grip_axis /= np.linalg.norm(grip_axis)

    rotation_matrix = np.column_stack([binormal, grip_axis, approach_dir])

    print(f"[INFO] 把手中心: [{handle_center[0]:.5f}, {handle_center[1]:.5f}, {handle_center[2]:.5f}]")
    print(f"[INFO] 物体中心: [{obj_center[0]:.5f}, {obj_center[1]:.5f}, {obj_center[2]:.5f}]")
    print(f"[INFO] Approach (XY对齐): [{approach_dir[0]:.4f}, {approach_dir[1]:.4f}, {approach_dir[2]:.4f}]")
    print(f"[INFO] Grip axis (水平): [{grip_axis[0]:.4f}, {grip_axis[1]:.4f}, {grip_axis[2]:.4f}]")
    print(f"[INFO] 倾斜角: {TILT_ANGLE_DEG}°")

    return {
        'grasp_center': handle_center,
        'approach_dir': approach_dir,
        'grip_axis': grip_axis,
        'binormal': binormal,
        'rotation_matrix': rotation_matrix,
        'aff_points': aff_points,
    }


# ============================================================================
# 可视化
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
    binormal = grasp_result['binormal']
    R_local = grasp_result['rotation_matrix']
    geometries = []

    half_open = GRIPPER_MAX_OPENING / 2.0
    fd = GRIPPER_FINGER_DEPTH
    fw = GRIPPER_FINGER_WIDTH
    ft = GRIPPER_FINGER_THICKNESS

    # 抓取中心: 黄色小球
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
    sphere.translate(center)
    sphere.paint_uniform_color([1.0, 1.0, 0.0])
    sphere.compute_vertex_normals()
    geometries.append(sphere)

    # Approach 箭头 (蓝色)
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

    # 两个手指 (青色)
    for sign in [+1, -1]:
        finger = o3d.geometry.TriangleMesh.create_box(width=fw, height=ft, depth=fd)
        finger.translate([-fw / 2, -ft / 2 + sign * half_open, -fd])
        finger.rotate(R_local, center=[0, 0, 0])
        finger.translate(center)
        finger.paint_uniform_color([0.0, 0.8, 0.8])
        finger.compute_vertex_normals()
        geometries.append(finger)

    # Grip axis 线 (品红)
    grip_line_len = GRIPPER_MAX_OPENING + 2 * ft + 0.01
    grip_cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=0.0008, height=grip_line_len)
    grip_cyl.translate([0, 0, -grip_line_len / 2])
    R_grip = rotation_from_z_to_dir(grip)
    grip_cyl.rotate(R_grip, center=[0, 0, 0])
    grip_cyl.translate(center)
    grip_cyl.paint_uniform_color([1.0, 0.0, 1.0])
    grip_cyl.compute_vertex_normals()
    geometries.append(grip_cyl)

    # 抓取坐标系
    grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.025)
    grasp_frame.rotate(R_local, center=[0, 0, 0])
    grasp_frame.translate(center)
    geometries.append(grasp_frame)

    return geometries


# ============================================================================
# 数据加载与批处理
# ============================================================================
def find_pt_path(type_path):
    pt_files = sorted(
        file_name for file_name in os.listdir(type_path)
        if file_name.endswith(".pt") and os.path.isfile(os.path.join(type_path, file_name))
    )

    if not pt_files:
        raise FileNotFoundError(f"{type_path} 中未找到 .pt 文件")

    if len(pt_files) > 1:
        print(f"[WARN] {type_path} 中找到多个 .pt 文件, 使用第一个: {pt_files[0]}")

    return os.path.join(type_path, pt_files[0])


def load_points_and_labels(pt_path):
    if torch is None:
        raise ImportError("未安装 torch, 无法加载 .pt 文件")

    data = torch.load(pt_path, map_location="cpu")
    points = data['points'].numpy()
    labels = data['labels'].numpy()
    print(f"[INFO] 加载: {pt_path}")
    return points, labels


def process_type_path(type_path, visualize=True, default_threshold=DEFAULT_UNIT_THRESHOLD):
    pt_path = find_pt_path(type_path)
    raw_points, labels = load_points_and_labels(pt_path)
    save_path = os.path.join(type_path, "grasp_result.npz")

    def preview_fn(points, threshold, converted, coord_range):
        print(f"[INFO] 总点数: {len(points)}, label=0: {(labels==0).sum()}, label=1: {(labels==1).sum()}")

        colors = np.zeros((len(labels), 3))
        colors[labels == 0] = [0.0, 1.0, 0.0]
        colors[labels == 1] = [1.0, 0.0, 0.0]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
        grasp_result = compute_handle_grasp(points, labels)

        print(f"\n{'='*60}")
        print(f"  抓取结果 (微波炉把手 - 固定方向)")
        print(f"  中心点:     [{grasp_result['grasp_center'][0]:.5f}, "
              f"{grasp_result['grasp_center'][1]:.5f}, {grasp_result['grasp_center'][2]:.5f}]")
        print(f"  Approach:   [{grasp_result['approach_dir'][0]:.4f}, "
              f"{grasp_result['approach_dir'][1]:.4f}, {grasp_result['approach_dir'][2]:.4f}]")
        print(f"  Grip axis:  [{grasp_result['grip_axis'][0]:.4f}, "
              f"{grasp_result['grip_axis'][1]:.4f}, {grasp_result['grip_axis'][2]:.4f}]")

        ee_pos = grasp_result['grasp_center'] - grasp_result['approach_dir'] * EE_OFFSET_Z
        print(f"  EE位置:     [{ee_pos[0]:.5f}, {ee_pos[1]:.5f}, {ee_pos[2]:.5f}]")
        print(f"{'='*60}")

        if visualize:
            grasp_geometries = create_grasp_visualization(grasp_result)
            all_geometries = [pcd, coord_frame] + grasp_geometries
            return attach_visualization(
                {"grasp_result": grasp_result, "ee_pos": ee_pos},
                all_geometries,
                window_name="微波炉把手抓取 (正面固定方向)",
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
        print(f"[INFO] 抓取结果已保存: {save_path}")

    return run_threshold_debug_loop(raw_points, os.path.basename(type_path), preview_fn, accept_fn, default_threshold)


def find_open_dirs(script_dir):
    return sorted(
        os.path.join(script_dir, entry)
        for entry in os.listdir(script_dir)
        if entry.endswith("_open") and os.path.isdir(os.path.join(script_dir, entry))
    )


# ============================================================================
# 主程序
# ============================================================================
if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    type_paths = find_open_dirs(script_dir)

    if not type_paths:
        raise FileNotFoundError(f"{script_dir} 下未找到任何 *_open 文件夹")

    if len(type_paths) > 1:
        print(f"[INFO] 检测到 {len(type_paths)} 个 *_open 文件夹, 将逐个可视化。请在 Open3D 窗口内按 Y/Enter 保存，U/D 调阈值，Q/Esc/X 退出。")

    for index, type_path in enumerate(type_paths, start=1):
        folder_name = os.path.basename(type_path)
        print(f"\n[INFO] ({index}/{len(type_paths)}) 处理文件夹: {folder_name}")
        if not process_type_path(type_path, visualize=True):
            print("[INFO] 用户中止批处理。")
            break
