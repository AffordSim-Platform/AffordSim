"""
5方向抓取位姿生成工具

将 approach 方向投影到 XY 平面，按与 X 轴夹角分为 4 个象限:
  Q0: 0-90°, Q1: 90-180°, Q2: 180-270°, Q3: 270-360°
每个象限各选出一个最优抓取位姿（保持 45° 倾斜约束）。
另外增加第 5 个方向: 垂直于 Z 轴（从正上方竖直下压抓取）。
"""

import numpy as np
import open3d as o3d
import os
from unit_threshold_helper import DEFAULT_UNIT_THRESHOLD, attach_visualization, run_threshold_debug_loop

try:
    import torch
except ImportError:
    torch = None

# 复用 check_pcdata 中的核心函数
from check_pcdata import (
    GRIPPER_MAX_OPENING, GRIPPER_FINGER_DEPTH,
    GRIPPER_FINGER_WIDTH, GRIPPER_FINGER_THICKNESS,
    EE_OFFSET_Z, TILT_ANGLE_DEG,
    cluster_affordance_points, sample_grasp_centers,
    generate_approach_directions, score_grasp,
    rotation_from_z_to_dir,
)

# 4 个象限 + 竖直方向的定义
QUADRANT_NAMES = ["Q0(0-90°)", "Q1(90-180°)", "Q2(180-270°)", "Q3(270-360°)", "垂直(Z-)"]
QUADRANT_COLORS = [
    [1.0, 0.0, 0.0],   # Q0: 红
    [0.0, 0.0, 1.0],   # Q1: 蓝
    [1.0, 0.5, 0.0],   # Q2: 橙
    [0.5, 0.0, 1.0],   # Q3: 紫
    [0.0, 0.8, 0.8],   # 垂直: 青
]


def get_approach_azimuth(approach_dir):
    """计算 approach 方向在 XY 平面投影与 X 轴的夹角 (0-360°)"""
    angle = np.degrees(np.arctan2(approach_dir[1], approach_dir[0]))
    if angle < 0:
        angle += 360.0
    return angle


def get_quadrant(azimuth):
    """根据方位角返回象限索引 0-3"""
    return int(azimuth // 90) % 4


def compute_4dir_grasps(points, labels):
    """计算 4 个象限各自的最优抓取位姿。

    Returns:
        dict: {
            'grasps': list of 4 grasp results (None if no valid candidate),
            'aff_points': affordance points,
            'all_candidates': all scored candidates,
        }
    """
    aff_points, global_centroid = cluster_affordance_points(points, labels)
    centers = sample_grasp_centers(aff_points)

    # 收集所有候选并打分
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

                # 计算方位角和象限
                azimuth = get_approach_azimuth(approach_dir)
                info['azimuth'] = azimuth
                info['quadrant'] = get_quadrant(azimuth)

                all_results.append(info)

    print(f"[INFO] 总计有效候选: {len(all_results)}")

    # 按象限分组，每组选最优
    quadrant_results = [[] for _ in range(4)]
    for r in all_results:
        quadrant_results[r['quadrant']].append(r)

    for q in range(4):
        # 象限中心角度
        q_center = q * 90.0 + 45.0
        # 综合排序: 原始得分 - 偏离象限中心的惩罚 (鼓励方位角靠近象限中心)
        def sort_key(r, qc=q_center):
            az = r['azimuth']
            # 角度差 (考虑 0/360 边界)
            diff = abs(az - qc)
            if diff > 180:
                diff = 360 - diff
            # 偏离 45° 中心越远，惩罚越大 (最大惩罚 0.3)
            azimuth_penalty = (diff / 45.0) * 0.3
            return r['total_score'] - azimuth_penalty
        quadrant_results[q].sort(key=sort_key, reverse=True)
        print(f"  {QUADRANT_NAMES[q]}: {len(quadrant_results[q])} 个候选")

    # 每个象限取最优
    grasps = []
    for q in range(4):
        q_center_azimuth = q * 90.0 + 45.0

        if not quadrant_results[q]:
            # fallback: 用该象限中心方向生成
            print(f"[WARN] {QUADRANT_NAMES[q]} 无有效候选, 使用 fallback")
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

        # ★ 统一规范 grip_axis: 用 Z 轴投影到与 approach 正交的平面
        # 这样夹爪手指尽量保持竖直方向开合
        z_up = np.array([0.0, 0.0, 1.0])
        grip_axis = z_up - np.dot(z_up, approach) * approach
        grip_norm = np.linalg.norm(grip_axis)
        if grip_norm < 1e-6:
            # approach 接近竖直时 (不应该走到这里, 但做保护)
            grip_axis = np.array([0.0, 1.0, 0.0])
            grip_axis = grip_axis - np.dot(grip_axis, approach) * approach
            grip_axis /= np.linalg.norm(grip_axis)
        else:
            grip_axis /= grip_norm

        binormal = np.cross(grip_axis, approach)  # 右手系: X = Y × Z
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

    # ★ 第5个方向: 纯竖直下压 approach=[0,0,-1]
    # grip 方向由 affordance 点的 PCA 决定 (取水平面内最窄方向, 方便夹爪夹住)
    vertical_approach = np.array([0.0, 0.0, -1.0])

    # 用 affordance 点的 XY 坐标做 2D PCA，找到把手在水平面内的延伸方向
    aff_xy = aff_points[:, :2]
    aff_xy_centered = aff_xy - aff_xy.mean(axis=0)
    cov_2d = np.cov(aff_xy_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov_2d)
    # eigvecs[:,1] 是最大特征值方向 = 把手在 XY 平面的延伸方向
    # eigvecs[:,0] 是最小特征值方向 = 垂直于把手 = 夹爪开合方向
    handle_dir_2d = eigvecs[:, 1]  # 把手方向
    perp_dir_2d = eigvecs[:, 0]    # 垂直于把手
    print(f"  [DEBUG] 2D PCA 特征值: {eigvals}")
    print(f"  [DEBUG] 把手方向(XY): [{handle_dir_2d[0]:.4f}, {handle_dir_2d[1]:.4f}]")
    print(f"  [DEBUG] 垂直方向(XY): [{perp_dir_2d[0]:.4f}, {perp_dir_2d[1]:.4f}]")
    # grip_axis = 垂直于把手方向 (夹爪沿此方向开合以夹住把手)
    # 当 PCA 方向接近坐标轴时，snap 到坐标轴以消除点云噪声引起的偏转
    grip_axis = np.array([perp_dir_2d[0], perp_dir_2d[1], 0.0])
    grip_axis /= np.linalg.norm(grip_axis)
    # 如果与 X 或 Y 轴夹角 < 20°，snap 到该轴
    SNAP_THRESHOLD = np.cos(np.radians(20.0))  # ~0.94
    axis_candidates = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
    for ax in axis_candidates:
        if abs(np.dot(grip_axis, ax)) > SNAP_THRESHOLD:
            grip_axis = ax * np.sign(np.dot(grip_axis, ax))
            print(f"  [DEBUG] grip_axis snapped to [{grip_axis[0]:.1f}, {grip_axis[1]:.1f}, 0.0]")
            break
    binormal = np.cross(grip_axis, vertical_approach)  # 右手系
    binormal /= np.linalg.norm(binormal)

    best_score = max((r['total_score'] for r in all_results), default=0.0)

    print(f"  {QUADRANT_NAMES[4]}: 纯竖直下压 "
          f"(grip沿PCA窄方向 [{grip_axis[0]:.3f}, {grip_axis[1]:.3f}])")

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

    # 打印 5 个方向的结果
    print(f"\n{'='*90}")
    print(f"  {'方向':<20} {'方位角':>8} {'仰角':>8} {'得分':>8} {'来源':<40}")
    print(f"  {'-'*84}")
    for g in grasps:
        elev = np.degrees(np.arcsin(np.clip(-g['approach_dir'][2], -1, 1)))
        print(f"  {QUADRANT_NAMES[g['quadrant']]:<20} "
              f"{g['azimuth']:>7.1f}° "
              f"{elev:>7.1f}° "
              f"{g['score']:>8.4f} "
              f"{g['source']:<40}")
    print(f"{'='*90}")

    return {
        'grasps': grasps,
        'aff_points': aff_points,
        'all_candidates': all_results,
    }


def create_4dir_visualization(grasps_result, show_gripper=True):
    """为 4 个方向的抓取位姿创建可视化几何体。

    Args:
        grasps_result: compute_4dir_grasps 的返回值
        show_gripper: 是否显示夹爪模型 (默认 True)
    """
    geometries = []

    for i, grasp in enumerate(grasps_result['grasps']):
        # 只可视化竖直方向 (第5个, index=4)
        # if grasp['quadrant'] != 4:
        #     continue
        center = grasp['grasp_center']
        approach = grasp['approach_dir']
        grip = grasp['grip_axis']
        binormal = grasp['binormal']
        color = QUADRANT_COLORS[i] if i < len(QUADRANT_COLORS) else [0.5, 0.5, 0.5]

        R_local = np.column_stack([binormal, grip, approach])

        # 抓取中心球
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
        sphere.translate(center)
        sphere.paint_uniform_color([1.0, 1.0, 0.0])
        sphere.compute_vertex_normals()
        geometries.append(sphere)

        # Approach 箭头
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

        # 夹爪模型 (可选)
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

        # 象限标签 (用小坐标系表示)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.015)
        frame.rotate(R_local, center=[0, 0, 0])
        frame.translate(center)
        geometries.append(frame)

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
        print(f"[INFO] 总点数: {len(points)}, label=0: {(labels==0).sum()}, label=1: {(labels==1).sum()}")
        print(f"[INFO] 目标倾斜角: {TILT_ANGLE_DEG}° (从斜上方接近)")

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
            print(f"    中心点:   [{g['grasp_center'][0]:.5f}, {g['grasp_center'][1]:.5f}, {g['grasp_center'][2]:.5f}]")
            print(f"    Approach: [{g['approach_dir'][0]:.4f}, {g['approach_dir'][1]:.4f}, {g['approach_dir'][2]:.4f}]")
            print(f"    方位角:   {g['azimuth']:.1f}°  仰角: {elev:.1f}°")
            print(f"    得分:     {g['score']:.4f}")
            print(f"    EE位置:   [{ee_pos[0]:.5f}, {ee_pos[1]:.5f}, {ee_pos[2]:.5f}]")

        grasp_geometries = create_4dir_visualization(result_4dir, show_gripper=show_gripper)
        all_geometries = [pcd, coord_frame] + grasp_geometries

        print("\n[VIS] 颜色说明:")
        color_names = ['红', '蓝', '橙', '紫', '青']
        for i in range(len(result_4dir['grasps'])):
            print(f"  {QUADRANT_NAMES[i]}: {color_names[i]}")

        return attach_visualization(
            {"result_4dir": result_4dir},
            all_geometries,
            window_name="5方向抓取位姿 (4象限+竖直)",
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
        print(f"\n[INFO] {len(result_4dir['grasps'])}方向抓取结果已保存: {save_path}")

    return run_threshold_debug_loop(raw_points, os.path.basename(type_path), preview_fn, accept_fn, default_threshold)


# ============================================================================
# 主程序
# ============================================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="4方向抓取位姿生成")
    parser.add_argument("--no-gripper", action="store_true", help="可视化时不显示夹爪模型")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    type_paths = find_pick_dirs(script_dir)

    if not type_paths:
        raise FileNotFoundError(f"{script_dir} 下未找到任何 *_pick 文件夹")

    if len(type_paths) > 1:
        print(f"[INFO] 检测到 {len(type_paths)} 个 *_pick 文件夹, 将逐个可视化。请在 Open3D 窗口内按 Y/Enter 保存，U/D 调阈值，Q/Esc/X 退出。")

    for index, type_path in enumerate(type_paths, start=1):
        folder_name = os.path.basename(type_path)
        print(f"\n[INFO] ({index}/{len(type_paths)}) 处理文件夹: {folder_name}")
        if not process_type_path(type_path, show_gripper=not args.no_gripper):
            print("[INFO] 用户中止批处理。")
            break
