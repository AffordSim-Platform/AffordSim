
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
GRIPPER_MAX_OPENING = 0.08        # 80mm 最大张开距离(Y/开合方向)
GRIPPER_FINGER_DEPTH = 0.10       # 100mm 手指深度/长度(Z/approach方向)
GRIPPER_FINGER_WIDTH = 0.02       # 20mm 手指宽度(X方向)
GRIPPER_FINGER_THICKNESS = 0.012  # 12mm 手指厚度(Y/开合方向)
EE_OFFSET_Z = 0.220              # fr3_hand → ee_frame 偏移
LOCAL_X_ROTATION_DEG = 0.0       # 输出抓取姿态绕局部 X(binormal) 旋转角度
LOCAL_Y_ROTATION_DEG = 0.0       # 输出抓取姿态绕局部 Y(grip_axis) 旋转角度
LOCAL_Z_ROTATION_DEG = 0.0      # 输出抓取姿态绕局部 Z(approach) 旋转角度

# ============================================================================
# ★ 倾斜抓取参数
# ============================================================================
TILT_ANGLE_DEG = 90.0             # approach 方向与水平面的夹角 (度)
                                   # 45° = 从斜上方45度接近物体
                                   # 0°  = 水平接近 (原始行为)
                                   # 90° = 从正上方下压

# ============================================================================
# 输入输出配置
# ============================================================================
INPUT_DIR = "red_mugcup_pick"
OUTPUT_FILE = "grasp_result_hang.npz"
PT_CANDIDATE_NAMES = ("labeled_result_hang.pt",)
NPZ_CANDIDATE_NAMES = ("labeled_result_hang.npz",)
DEFAULT_UNIT_THRESHOLD_VALUE = DEFAULT_UNIT_THRESHOLD
POINT_CLOUD_GRAY = np.array([0.58, 0.58, 0.58])
AFFORDANCE_RED = np.array([1.0, 0.22, 0.22])


# ============================================================================
# ★ 辅助函数: 将水平方向倾斜指定角度向上
# ============================================================================
def tilt_direction_upward(direction, tilt_deg=TILT_ANGLE_DEG):
    """将一个方向向量倾斜 tilt_deg 度向上 (Z+方向).

    思路: 将 direction 分解为水平分量和竖直分量,
    然后重新组合, 使其与水平面夹角为 tilt_deg.
    如果原方向已经是纯竖直 (如 [0,0,-1]), 则保持不变.

    Args:
        direction: 3D方向向量 (会被归一化)
        tilt_deg: 与水平面的夹角 (度), 正值=向上倾斜

    Returns:
        倾斜后的单位方向向量
    """
    d = np.array(direction, dtype=np.float64)
    d /= np.linalg.norm(d)

    # 水平分量
    d_horiz = np.array([d[0], d[1], 0.0])
    horiz_norm = np.linalg.norm(d_horiz)

    # 如果方向几乎是纯竖直的, 不做倾斜处理
    if horiz_norm < 1e-6:
        return d

    d_horiz /= horiz_norm

    tilt_rad = np.radians(tilt_deg)
    # 新方向 = 水平分量 * cos(tilt) + 竖直向上 * sin(tilt)
    # 注意: approach 是"从哪里来"的方向, 倾斜向上意味着从斜上方接近
    # 所以 Z 分量应该是负的 (从上往下的分量), 但 approach 指向物体,
    # 如果原方向是水平指向物体, 倾斜后应该从斜上方指向物体
    # approach 方向 = 夹爪前进方向, 所以加一个向下的Z分量表示"从上方来"
    new_dir = d_horiz * np.cos(tilt_rad) + np.array([0.0, 0.0, -np.sin(tilt_rad)])

    norm = np.linalg.norm(new_dir)
    if norm < 1e-8:
        return d
    return new_dir / norm


# ============================================================================
# Step 1: Affordance 点聚类
# ============================================================================
def cluster_affordance_points(points, labels, affordance_label=1,
                              dbscan_eps=0.008, dbscan_min_samples=5):
    """提取 affordance 点并聚类，返回最大簇的点和质心。"""
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
# Step 2: 采样抓取中心点
# ============================================================================
def sample_grasp_centers(aff_points, n_samples=12, local_radius=0.003):
    """在 affordance 点上采样多个局部抓取中心。"""
    aff_spread = aff_points.ptp(axis=0).max()
    centroid = aff_points.mean(axis=0)

    if aff_spread < GRIPPER_MAX_OPENING * 1.2 or len(aff_points) < 30:
        return [(centroid, aff_points, "质心")]

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
        centers.append((local_center, local_pts, f"局部{i}"))

    print(f"[INFO] 采样 {len(centers)} 个抓取中心 (aff_spread={aff_spread*1000:.1f}mm, "
          f"local_r={local_radius*1000:.0f}mm)")
    return centers


# ============================================================================
# Step 3: 生成 approach 方向 (★ 修改: 全部倾斜45°向上)
# ============================================================================
def compute_handle_direction(aff_points, all_points):
    """计算杯把方向: affordance 区域质心相对于整体质心的水平方向。

    返回单位向量 (仅水平分量, Z=0), 指向杯把方向。
    如果无法计算返回 None。
    """
    if len(aff_points) < 3:
        return None
    aff_centroid = aff_points.mean(axis=0)
    obj_centroid = all_points.mean(axis=0)
    diff = aff_centroid - obj_centroid
    # 只取水平分量
    handle_dir = np.array([diff[0], diff[1], 0.0])
    norm = np.linalg.norm(handle_dir)
    if norm < 1e-6:
        return None
    return handle_dir / norm


def generate_approach_directions(local_aff_points, all_points, grasp_center):
    """为一个 grasp_center 生成候选 approach 方向列表。

    ★ 优先生成正对杯把方向的 approach (从杯把正面接近),
       所有水平方向都会被倾斜 TILT_ANGLE_DEG 度向上。
    """
    directions = []

    def normalize(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-8 else None

    def add(d, src, apply_tilt=True):
        """添加候选方向。apply_tilt=True 时对水平方向施加倾斜。"""
        v = normalize(np.array(d, dtype=np.float64))
        if v is None:
            return

        if apply_tilt and TILT_ANGLE_DEG > 0:
            v = tilt_direction_upward(v, TILT_ANGLE_DEG)
            src += f"(倾斜{TILT_ANGLE_DEG:.0f}°)"

        directions.append((v, src))

    # ★★★ 杯把方向优先: 从杯把正面接近 ★★★
    handle_dir = compute_handle_direction(local_aff_points, all_points)
    if handle_dir is not None:
        # approach 应该正对杯把 (从杯把外侧朝杯体方向)
        add(-handle_dir, "杯把正面")
        add(handle_dir, "杯把背面")
        # 杯把方向旋转 ±30°, ±60°, ±90° 的候选
        for angle_deg in [30, -30, 60, -60, 90, -90]:
            angle_rad = np.radians(angle_deg)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            rotated = np.array([
                handle_dir[0] * cos_a - handle_dir[1] * sin_a,
                handle_dir[0] * sin_a + handle_dir[1] * cos_a,
                0.0
            ])
            add(-rotated, f"杯把旋转{angle_deg}°")

    # 表面法线
    if len(local_aff_points) >= 3:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(local_aff_points)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
        normals = np.asarray(pcd.normals)
        obj_centroid = all_points.mean(axis=0)
        for i in range(len(normals)):
            if np.dot(normals[i], local_aff_points[i] - obj_centroid) < 0:
                normals[i] *= -1
        avg_normal = normals.mean(axis=0)
        n = normalize(avg_normal)
        if n is not None:
            add(-n, "法线朝内")
            add(n, "法线朝外")

    # PCA
    if len(local_aff_points) >= 3:
        centered = local_aff_points - grasp_center
        cov = np.cov(centered.T)
        _, eigvecs = np.linalg.eigh(cov)
        for sign in [1, -1]:
            add(sign * eigvecs[:, 0], f"PCA{'+' if sign > 0 else '-'}")

    # 物体中心 -> grasp_center
    obj_centroid = all_points.mean(axis=0)
    diff = grasp_center - obj_centroid
    add(diff, "离心")
    add(-diff, "向心")

    # 固定方向 (★ 水平方向倾斜, 竖直方向保持)
    add([1, 0, 0], "+X")
    add([-1, 0, 0], "-X")
    add([0, 1, 0], "+Y")
    add([0, -1, 0], "-Y")
    # 竖直方向不倾斜
    add([0, 0, -1], "下压", apply_tilt=False)
    add([0, 0, 1], "上托", apply_tilt=False)

    # ★ 额外添加: 斜45°的8个方向 (均匀覆盖水平面各角度)
    tilt_rad = np.radians(TILT_ANGLE_DEG)
    cos_t = np.cos(tilt_rad)
    sin_t = np.sin(tilt_rad)
    for angle_deg in range(0, 360, 45):
        angle_rad = np.radians(angle_deg)
        d = np.array([
            np.cos(angle_rad) * cos_t,
            np.sin(angle_rad) * cos_t,
            -sin_t
        ])
        add(d, f"斜{angle_deg}°", apply_tilt=False)

    return directions


# ============================================================================
# Step 4: 打分 (★ 修改: upright_score 鼓励45°倾斜)
# ============================================================================
def score_grasp(approach_dir, grip_axis, grasp_center,
                all_points, labels,
                gripper_opening=GRIPPER_MAX_OPENING,
                finger_depth=GRIPPER_FINGER_DEPTH,
                finger_width=GRIPPER_FINGER_WIDTH,
                finger_thickness=GRIPPER_FINGER_THICKNESS,
                handle_dir=None):
    """对单个 (grasp_center, approach, grip_axis) 打分。返回 None 表示淘汰。

    局部坐标系: X=binormal(宽度方向), Y=grip_axis(开合方向), Z=approach(深度方向)
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

    # 两指之间 (非 aff 点) -> 硬约束
    n_closing = int(in_box(non_aff_local,
                           [-half_width, half_width],
                           [-half_open, half_open],
                           [-finger_depth, 0]).sum())
    if n_closing > 3:
        return None

    # 手指碰撞
    n_left = int(in_box(non_aff_local,
                        [-half_width, half_width],
                        [half_open, half_open + finger_thickness],
                        [-finger_depth, 0]).sum())
    n_right = int(in_box(non_aff_local,
                         [-half_width, half_width],
                         [-half_open - finger_thickness, -half_open],
                         [-finger_depth, 0]).sum())

    # 手掌碰撞
    n_palm = int(in_box(non_aff_local,
                        [-half_width, half_width],
                        [-half_open - finger_thickness, half_open + finger_thickness],
                        [-finger_depth - 0.02, -finger_depth]).sum())

    # 进近通道
    n_approach = int(in_box(all_local,
                            [-half_width, half_width],
                            [-half_open - finger_thickness, half_open + finger_thickness],
                            [-finger_depth - 0.06, -finger_depth - 0.02]).sum())

    n_finger_col = n_left + n_right
    collision_score = max(0.0, 1.0 - n_finger_col * 0.2 - n_closing * 0.3 - n_palm * 0.15)
    clearance_score = max(0.0, 1.0 - n_approach * 0.05)

    # 覆盖
    in_grip = in_box(aff_local,
                     [-half_width, half_width],
                     [-half_open, half_open],
                     [-finger_depth, 0])
    n_aff_in_grip = int(in_grip.sum())
    coverage_score = n_aff_in_grip / max(len(aff_local), 1)

    # 对心性
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

    # ★★★ 姿态正则化: 鼓励 approach 方向与水平面夹角接近目标倾斜角 ★★★
    # approach_dir 的 Z 分量 = -sin(倾斜角), 倾斜45°时 Z ≈ -0.707
    target_tilt_rad = np.radians(TILT_ANGLE_DEG)
    target_z = -np.sin(target_tilt_rad)  # 目标 approach 的 Z 分量 (负=从上方来)

    # 实际 approach 的倾斜角
    actual_z = approach_dir[2]
    # 与目标的偏差 (越小越好)
    tilt_error = abs(actual_z - target_z)
    # 归一化到 [0, 1], 误差为0时得分为1
    tilt_score = max(0.0, 1.0 - tilt_error * 2.0)

    # grip_axis 和 binormal 不应指向正下方 (避免手指戳桌面)
    grip_z = abs(grip_axis[2])
    binormal_z = abs(binormal[2])
    orientation_penalty = 0.5 * (grip_z + binormal_z)
    upright_score = tilt_score * (1.0 - 0.3 * orientation_penalty)

    # 手指内壁接触
    if n_aff_in_grip > 0:
        aff_in = aff_local[in_grip]
        dist_to_wall = half_open - np.abs(aff_in[:, 1])
        contact_score = np.mean(np.clip(1.0 - dist_to_wall / half_open, 0.0, 1.0))
    else:
        contact_score = 0.0

    # ★ 额外: 桌面碰撞惩罚
    # 如果 approach 太水平 (Z分量接近0), 严重惩罚
    horiz_penalty = 0.0
    approach_elevation = np.degrees(np.arcsin(np.clip(-approach_dir[2], -1, 1)))
    if approach_elevation < 15.0:  # 倾斜角小于15°, 几乎水平
        horiz_penalty = 0.5 * (1.0 - approach_elevation / 15.0)

    # ★★★ 杯把方向对齐分 ★★★
    # 1) approach 的水平分量应正对杯把方向
    # 2) grip axis (开合方向) 应与杯把方向对齐, 这样夹爪才能夹住杯把
    handle_align_score = 0.5  # 默认中性分 (无杯把信息时)
    grip_handle_score = 0.5   # grip axis 与杯把方向的对齐分
    if handle_dir is not None:
        # approach 对齐
        approach_horiz = np.array([approach_dir[0], approach_dir[1], 0.0])
        ah_norm = np.linalg.norm(approach_horiz)
        if ah_norm > 1e-6:
            approach_horiz /= ah_norm
            alignment = np.dot(approach_horiz, -handle_dir)
            handle_align_score = (alignment + 1.0) / 2.0
        else:
            handle_align_score = 0.3

        # ★ grip axis 对齐: grip axis 的水平投影应与杯把方向平行
        # 夹爪沿 grip axis 开合, 要夹住杯把, grip axis 应垂直于杯把延伸方向
        # 即 grip axis 水平投影 与 handle_dir 垂直 (dot ≈ 0) 才能正面夹住
        # 或者 grip axis 与 handle_dir 平行 (dot ≈ ±1) 表示从侧面夹
        # 对于杯把: 夹爪应该从杯把两侧夹住, 即 grip axis ⊥ handle_dir
        grip_horiz = np.array([grip_axis[0], grip_axis[1], 0.0])
        gh_norm = np.linalg.norm(grip_horiz)
        if gh_norm > 1e-6:
            grip_horiz /= gh_norm
            # grip 与 handle_dir 的点积, 越接近 0 表示越垂直 (夹爪正面夹杯把)
            grip_dot = abs(np.dot(grip_horiz, handle_dir))
            # grip_dot ≈ 0 → 垂直 → 最好 (得分 1.0)
            # grip_dot ≈ 1 → 平行 → 最差 (得分 0.0)
            grip_handle_score = 1.0 - grip_dot
        else:
            grip_handle_score = 0.3

    total = (collision_score * 0.12 +
             clearance_score * 0.05 +
             coverage_score * 0.14 +
             antipodal_score * 0.08 +
             upright_score * 0.16 +
             contact_score * 0.08 +
             handle_align_score * 0.15 +    # approach 正对杯把
             grip_handle_score * 0.17 -      # ★ grip axis 垂直于杯把 (最高权重)
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


def rotation_matrix_from_local_euler(x_deg=0.0, y_deg=0.0, z_deg=0.0):
    """局部坐标系下的旋转矩阵，按 X -> Y -> Z 顺序依次施加。"""
    x_rad = np.radians(x_deg)
    y_rad = np.radians(y_deg)
    z_rad = np.radians(z_deg)

    rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(x_rad), -np.sin(x_rad)],
        [0.0, np.sin(x_rad), np.cos(x_rad)],
    ])
    ry = np.array([
        [np.cos(y_rad), 0.0, np.sin(y_rad)],
        [0.0, 1.0, 0.0],
        [-np.sin(y_rad), 0.0, np.cos(y_rad)],
    ])
    rz = np.array([
        [np.cos(z_rad), -np.sin(z_rad), 0.0],
        [np.sin(z_rad), np.cos(z_rad), 0.0],
        [0.0, 0.0, 1.0],
    ])
    return rx @ ry @ rz


def rotate_grasp_result_about_local_axes(
    grasp_result,
    x_deg=LOCAL_X_ROTATION_DEG,
    y_deg=LOCAL_Y_ROTATION_DEG,
    z_deg=LOCAL_Z_ROTATION_DEG,
):
    """将最终抓取姿态绕局部 X/Y/Z 轴旋转指定角度。"""
    r_local = np.array(grasp_result['rotation_matrix'], dtype=np.float64)
    r_delta = rotation_matrix_from_local_euler(x_deg=x_deg, y_deg=y_deg, z_deg=z_deg)
    r_rotated = r_local @ r_delta

    rotated_binormal = r_rotated[:, 0]
    rotated_grip = r_rotated[:, 1]
    rotated_approach = r_rotated[:, 2]

    rotated_binormal /= np.linalg.norm(rotated_binormal)
    rotated_grip /= np.linalg.norm(rotated_grip)
    rotated_approach /= np.linalg.norm(rotated_approach)

    rotated = dict(grasp_result)
    rotated['binormal'] = rotated_binormal
    rotated['grip_axis'] = rotated_grip
    rotated['approach_dir'] = rotated_approach
    rotated['rotation_matrix'] = np.column_stack([rotated_binormal, rotated_grip, rotated_approach])
    rotated['source'] = (
        f"{grasp_result['source']}/"
        f"rx{int(x_deg)}_ry{int(y_deg)}_rz{int(z_deg)}"
    )
    rotated['local_x_rotation_deg'] = x_deg
    rotated['local_y_rotation_deg'] = y_deg
    rotated['local_z_rotation_deg'] = z_deg
    return rotated


# ============================================================================
# Step 5: 计算最优抓取
# ============================================================================
def compute_best_grasp(points, labels):
    """主函数：从标注点云计算最优抓取位姿。"""
    aff_points, global_centroid = cluster_affordance_points(points, labels)
    centers = sample_grasp_centers(aff_points)

    # ★ 计算杯把方向 (用于打分时对齐奖励)
    handle_dir = compute_handle_direction(aff_points, points)
    if handle_dir is not None:
        print(f"[INFO] 杯把方向: [{handle_dir[0]:.3f}, {handle_dir[1]:.3f}, {handle_dir[2]:.3f}]")

    results = []
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

                info = score_grasp(approach_dir, grip, center, points, labels,
                                   handle_dir=handle_dir)
                if info is None:
                    continue

                info['approach_dir'] = approach_dir
                info['grasp_center'] = center
                info['source'] = f"{center_name}/{approach_name}@{int(np.degrees(angle))}"
                results.append(info)

    if not results:
        # ★ Fallback 也使用倾斜方向
        print("[WARN] 所有候选都被淘汰! 使用 fallback (斜45°下压)")
        tilt_rad = np.radians(TILT_ANGLE_DEG)
        approach = np.array([np.cos(tilt_rad), 0.0, -np.sin(tilt_rad)])
        approach /= np.linalg.norm(approach)
        grip = np.array([0.0, 1.0, 0.0])
        binormal = np.cross(approach, grip)
        binormal /= np.linalg.norm(binormal)
        return rotate_grasp_result_about_local_axes({
            'grasp_center': global_centroid,
            'approach_dir': approach,
            'grip_axis': grip,
            'binormal': binormal,
            'rotation_matrix': np.column_stack([binormal, grip, approach]),
            'score': 0.0,
            'source': 'fallback(tilted)',
            'all_candidates': [],
            'aff_points': aff_points,
        })

    results.sort(key=lambda x: x['total_score'], reverse=True)

    # 打印 Top 20
    print(f"\n{'='*140}")
    print(f"  {'候选':<40} {'总分':>6} {'碰撞':>6} {'通道':>6} {'覆盖':>6} "
          f"{'对心':>6} {'倾斜':>6} {'贴壁':>6} {'朝向':>6} {'夹向':>6} {'碰数':>5} {'闭碰':>5} {'夹aff':>5} {'宽mm':>6} {'仰角':>6}")
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
              f"{r.get('approach_elevation', 0):>5.1f}°")
    if len(results) > 20:
        print(f"  ... 还有 {len(results)-20} 个有效候选")
    print(f"{'='*120}")
    print(f"[INFO] 总计有效候选: {len(results)}")

    best = results[0]
    approach_dir = best['approach_dir']
    grip_axis = best['grip_axis']
    binormal = np.cross(approach_dir, grip_axis)
    binormal /= np.linalg.norm(binormal)
    grip_axis = np.cross(binormal, approach_dir)
    grip_axis /= np.linalg.norm(grip_axis)

    # ★ 打印 approach 倾斜角度
    elevation = np.degrees(np.arcsin(np.clip(-approach_dir[2], -1, 1)))
    print(f"[INFO] 最优 approach 与水平面夹角: {elevation:.1f}° (目标: {TILT_ANGLE_DEG}°)")

    return rotate_grasp_result_about_local_axes({
        'grasp_center': best['grasp_center'],
        'approach_dir': approach_dir,
        'grip_axis': grip_axis,
        'binormal': binormal,
        'rotation_matrix': np.column_stack([binormal, grip_axis, approach_dir]),
        'score': best['total_score'],
        'source': best['source'],
        'all_candidates': results,
        'aff_points': aff_points,
    })


# ============================================================================
# Step 6: 可视化
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

    # ★ 新增: 画一条虚拟桌面参考线 (帮助观察倾斜角度)
    elevation = np.degrees(np.arcsin(np.clip(-approach[2], -1, 1)))
    print(f"[VIS] approach 与水平面夹角: {elevation:.1f}°")

    return geometries


# ============================================================================
# 主程序
# ============================================================================
def load_points_and_labels(type_path):
    for name in PT_CANDIDATE_NAMES:
        candidate = os.path.join(type_path, name)
        if os.path.exists(candidate):
            if torch is None:
                raise ImportError("未安装 torch, 无法加载 .pt 文件")
            data = torch.load(candidate, map_location="cpu")
            points = data['points'].numpy()
            labels = data['labels'].numpy()
            print(f"[INFO] 加载: {candidate}")
            return points, labels

    for name in NPZ_CANDIDATE_NAMES:
        candidate = os.path.join(type_path, name)
        if os.path.exists(candidate):
            data = np.load(candidate)
            points = data['points']
            labels = data['labels']
            print(f"[INFO] 加载: {candidate}")
            return points, labels

    raise FileNotFoundError(f"未找到标注文件: {type_path}")


def find_pick_dirs(script_dir):
    return sorted(
        os.path.join(script_dir, entry)
        for entry in os.listdir(script_dir)
        if entry.endswith("_pick") and os.path.isdir(os.path.join(script_dir, entry))
    )


def process_type_path(type_path, default_threshold=DEFAULT_UNIT_THRESHOLD_VALUE):
    raw_points, labels = load_points_and_labels(type_path)
    save_path = os.path.join(type_path, OUTPUT_FILE)

    def preview_fn(points, threshold, converted, coord_range):
        print(f"[INFO] 总点数: {len(points)}, label=0: {(labels==0).sum()}, label=1: {(labels==1).sum()}")
        print(f"[INFO] 目标倾斜角: {TILT_ANGLE_DEG}° (从斜上方接近)")
        colors = np.tile(POINT_CLOUD_GRAY, (len(labels), 1))
        colors[labels == 1] = AFFORDANCE_RED

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        grasp_result = compute_best_grasp(points, labels)
        print(f"\n{'='*60}")
        print(f"  抓取结果")
        print(f"  中心点:     [{grasp_result['grasp_center'][0]:.5f}, "
              f"{grasp_result['grasp_center'][1]:.5f}, {grasp_result['grasp_center'][2]:.5f}]")
        print(f"  Approach:   [{grasp_result['approach_dir'][0]:.4f}, "
              f"{grasp_result['approach_dir'][1]:.4f}, {grasp_result['approach_dir'][2]:.4f}]")
        print(f"  Grip axis:  [{grasp_result['grip_axis'][0]:.4f}, "
              f"{grasp_result['grip_axis'][1]:.4f}, {grasp_result['grip_axis'][2]:.4f}]")
        print(f"  得分:       {grasp_result['score']:.4f}")
        print(f"  来源:       {grasp_result['source']}")
        print(f"  局部X旋转:  {grasp_result['local_x_rotation_deg']:.1f}°")
        print(f"  局部Y旋转:  {grasp_result['local_y_rotation_deg']:.1f}°")
        print(f"  局部Z旋转:  {grasp_result['local_z_rotation_deg']:.1f}°")
        elevation = np.degrees(np.arcsin(np.clip(-grasp_result['approach_dir'][2], -1, 1)))
        print(f"  倾斜角:     {elevation:.1f}° (目标: {TILT_ANGLE_DEG}°)")
        ee_pos = grasp_result['grasp_center'] - grasp_result['approach_dir'] * EE_OFFSET_Z
        print(f"  EE位置:     [{ee_pos[0]:.5f}, {ee_pos[1]:.5f}, {ee_pos[2]:.5f}]")
        print(f"{'='*60}")
        grasp_geometries = create_grasp_visualization(grasp_result)
        all_geometries = [pcd] + grasp_geometries
        return attach_visualization(
            {"grasp_result": grasp_result, "ee_pos": ee_pos},
            all_geometries,
            window_name=(
                f"Affordance + 抓取位姿 "
                f"(倾斜{TILT_ANGLE_DEG}°, "
                f"Rx{LOCAL_X_ROTATION_DEG:.0f}°, "
                f"Ry{LOCAL_Y_ROTATION_DEG:.0f}°, "
                f"Rz{LOCAL_Z_ROTATION_DEG:.0f}°)"
            ),
            width=1280,
            height=720,
        )

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
                 local_x_rotation_deg=np.array(grasp_result['local_x_rotation_deg']),
                 local_y_rotation_deg=np.array(grasp_result['local_y_rotation_deg']),
                 local_z_rotation_deg=np.array(grasp_result['local_z_rotation_deg']),
                 unit_threshold=np.array(threshold),
                 unit_converted=np.array(int(converted)))
        print(f"[INFO] 抓取结果已保存: {save_path}")

    return run_threshold_debug_loop(
        raw_points, os.path.basename(type_path), preview_fn, accept_fn, default_threshold
    )


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    type_path = os.path.join(script_dir, INPUT_DIR)

    if not os.path.isdir(type_path):
        raise FileNotFoundError(f"目标文件夹不存在: {type_path}")

    print(f"[INFO] 处理文件夹: {os.path.basename(type_path)}")
    print("[INFO] 键位: Y/Enter=保存, U/↑=阈值×10, D/↓=阈值÷10, Q/Esc/X=退出")
    process_type_path(type_path)
