import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    print("[警告] 未安装 torch，将使用 numpy 替代")
    torch = None

try:
    import open3d as o3d
except ImportError:
    print("[错误] 请先安装 open3d:  pip install open3d")
    sys.exit(1)

try:
    import cv2
except ImportError:
    cv2 = None
    print("[警告] 未安装 opencv-python，图像读取功能受限")


# =============================================================================
# ★★★ 在这里配置你的相机内外参 ★★★
# =============================================================================

K = np.array([
    [796.44,    0,      640],
    [   0,    796.44,   640],
    [   0,       0,        1]
], dtype=np.float64)

DEPTH_SCALE = 1000

CAMERAS = {
    'front': np.array([
        [-0.032170, 0.698593, -0.714796, 1.168809],
        [0.998000, 0.061387, 0.015079, -0.027872],
        [0.054413, -0.712881, -0.699170, 0.686328],
        [0.000000, 0.000000, 0.000000, 1.000000],
    ]),

    'side' : np.array([
        [-1.000000, -0.000000, 0.000000, 0.614220],
        [0.000000, 0.500000, -0.866025, 0.983940],
        [0.000000, -0.866025, -0.500000, 0.672690],
        [0.000000, 0.000000, 0.000000, 1.000000],
    ]),

    'right': np.array([
        [1.000000, -0.000000, 0.000000, 0.614220],
        [0.000000, -0.500000, 0.866025, -0.983940],
        [-0.000000, -0.866025, -0.500000, 0.672690],
        [0.000000, 0.000000, 0.000000, 1.000000],
    ]),

    'top': np.array([
        [0.017695, 0.999271, -0.033840, 0.717000],
        [0.997591, -0.015374, 0.067638, -0.168000],
        [0.067069, -0.034955, -0.997136, 0.602000],
        [0.000000, 0.000000, 0.000000, 1.000000],
    ]),
}


# =============================================================================
# 第一部分: 多视角 RGB + 深度图 → 点云
# =============================================================================

def depth_to_pointcloud(rgb_img: np.ndarray,
                        depth_img: np.ndarray,
                        K_mat: np.ndarray,
                        T_c2w: np.ndarray = None,
                        depth_scale: float = 1000.0,
                        depth_trunc: float = 10.0) -> o3d.geometry.PointCloud:
    """单帧 RGB + 深度图反投影为点云。"""
    fx, fy = K_mat[0, 0], K_mat[1, 1]
    cx, cy = K_mat[0, 2], K_mat[1, 2]

    h, w = depth_img.shape[:2]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = u.astype(np.float64)
    v = v.astype(np.float64)

    z = depth_img.astype(np.float64) / depth_scale
    mask = (z > 0) & (z < depth_trunc)

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points_cam = np.stack([x[mask], y[mask], z[mask]], axis=-1)
    colors = rgb_img[mask].astype(np.float64) / 255.0

    if T_c2w is not None:
        R = T_c2w[:3, :3]
        t = T_c2w[:3, 3]
        points_world = (R @ points_cam.T).T + t
    else:
        points_world = points_cam

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def depth_to_pointcloud_with_mask(rgb_img: np.ndarray,
                                  depth_img: np.ndarray,
                                  K_mat: np.ndarray,
                                  T_c2w: np.ndarray = None,
                                  depth_scale: float = 1000.0,
                                  depth_trunc: float = 10.0,
                                  object_mask: np.ndarray = None) -> o3d.geometry.PointCloud:
    """
    单帧 RGB + 深度图反投影为点云，支持 object_mask 只保留物体区域。

    参数:
        object_mask: [H, W] bool, True 表示该像素属于物体
    """
    fx, fy = K_mat[0, 0], K_mat[1, 1]
    cx, cy = K_mat[0, 2], K_mat[1, 2]

    h, w = depth_img.shape[:2]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = u.astype(np.float64)
    v = v.astype(np.float64)

    z = depth_img.astype(np.float64) / depth_scale
    mask = (z > 0) & (z < depth_trunc)

    # ★ 额外用 object_mask 过滤
    if object_mask is not None:
        mask = mask & object_mask

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points_cam = np.stack([x[mask], y[mask], z[mask]], axis=-1)
    colors = rgb_img[mask].astype(np.float64) / 255.0

    if T_c2w is not None:
        R = T_c2w[:3, :3]
        t = T_c2w[:3, 3]
        points_world = (R @ points_cam.T).T + t
    else:
        points_world = points_cam

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


# =============================================================================
# ★★★ 新增: 图像级物体分割 (在反投影之前过滤背景) ★★★
# =============================================================================

def create_object_mask_by_color(rgb_img: np.ndarray,
                                bg_color: tuple = (150, 145, 180),
                                threshold: float = 60.0,
                                morph_kernel: int = 5) -> np.ndarray:
    """
    通过排除背景色来生成物体 mask。

    原理: 计算每个像素与背景色的欧氏距离，距离大于 threshold 的认为是物体。
    适用于背景色比较均匀的场景 (如你的紫色桌面)。

    参数:
        rgb_img:     [H, W, 3] uint8, RGB 格式
        bg_color:    背景色 RGB (默认紫色桌面 ~(150,145,180))
        threshold:   容差, 越小越严格
        morph_kernel: 形态学操作核大小, 去除噪点

    返回:
        mask: [H, W] bool, True = 物体
    """
    bg = np.array(bg_color, dtype=np.float64)
    diff = np.linalg.norm(rgb_img.astype(np.float64) - bg, axis=-1)
    mask = diff > threshold

    # 形态学操作: 去除小噪点，填充小孔洞
    if morph_kernel > 0:
        kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
        mask_u8 = mask.astype(np.uint8) * 255
        # 先开操作去噪点
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
        # 再闭操作填孔洞
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
        mask = mask_u8 > 0

    n_obj = np.sum(mask)
    n_total = mask.size
    print(f"    颜色过滤: {n_obj}/{n_total} 像素属于物体 ({n_obj/n_total*100:.1f}%)")
    return mask


def create_object_mask_by_depth_diff(depth_img: np.ndarray,
                                     depth_scale: float = 1000.0,
                                     bg_depth_range: tuple = (0.5, 10.0),
                                     obj_depth_range: tuple = (0.1, 0.5)) -> np.ndarray:
    """
    通过深度范围来区分物体和背景。

    如果物体比背景更近 (桌上的物体 vs 桌面)，可以用深度差异过滤。
    """
    z = depth_img.astype(np.float64) / depth_scale
    mask = (z > obj_depth_range[0]) & (z < obj_depth_range[1])
    n_obj = np.sum(mask)
    print(f"    深度过滤: {n_obj}/{mask.size} 像素在深度范围 [{obj_depth_range[0]}, {obj_depth_range[1]}]m")
    return mask


def auto_detect_background_color(rgb_img: np.ndarray, sample_ratio: float = 0.1) -> tuple:
    """
    自动检测背景色。

    通过对边缘区域采样来估计背景色 (假设边缘大多是背景)。
    """
    h, w = rgb_img.shape[:2]
    border = int(min(h, w) * sample_ratio)

    # 采集四个边缘区域
    samples = np.concatenate([
        rgb_img[:border, :, :].reshape(-1, 3),       # 上
        rgb_img[-border:, :, :].reshape(-1, 3),      # 下
        rgb_img[:, :border, :].reshape(-1, 3),       # 左
        rgb_img[:, -border:, :].reshape(-1, 3),      # 右
    ], axis=0)

    # 中位数作为背景色 (比均值更鲁棒)
    bg_color = np.median(samples, axis=0).astype(int)
    print(f"    自动检测背景色: RGB=({bg_color[0]}, {bg_color[1]}, {bg_color[2]})")
    return tuple(bg_color)


# =============================================================================
# ★★★ 新增: 点云级物体提取 (在反投影之后过滤) ★★★
# =============================================================================

def filter_pointcloud_by_color(pcd: o3d.geometry.PointCloud,
                               bg_color_01: tuple = (0.588, 0.569, 0.706),
                               threshold: float = 0.2) -> o3d.geometry.PointCloud:
    """
    在点云层面按颜色过滤背景。

    参数:
        bg_color_01: 归一化的背景色 [0,1]
        threshold:   容差
    """
    colors = np.asarray(pcd.colors)
    bg = np.array(bg_color_01)
    diff = np.linalg.norm(colors - bg, axis=-1)
    mask = diff > threshold

    indices = np.where(mask)[0]
    filtered = pcd.select_by_index(indices.tolist())
    print(f"[INFO] 颜色过滤: {len(pcd.points)} → {len(filtered.points)} 个点")
    return filtered


def filter_pointcloud_by_bbox(pcd: o3d.geometry.PointCloud,
                              bbox: tuple) -> o3d.geometry.PointCloud:
    """
    用包围盒裁剪点云。

    参数:
        bbox: (x_min, x_max, y_min, y_max, z_min, z_max)
    """
    x0, x1, y0, y1, z0, z1 = bbox
    points = np.asarray(pcd.points)
    mask = ((points[:, 0] >= x0) & (points[:, 0] <= x1) &
            (points[:, 1] >= y0) & (points[:, 1] <= y1) &
            (points[:, 2] >= z0) & (points[:, 2] <= z1))
    indices = np.where(mask)[0]
    filtered = pcd.select_by_index(indices.tolist())
    print(f"[INFO] 包围盒裁剪: {len(pcd.points)} → {len(filtered.points)} 个点")
    return filtered


def filter_pointcloud_by_cluster(pcd: o3d.geometry.PointCloud,
                                 eps: float = 0.01,
                                 min_points: int = 100,
                                 keep_largest: bool = True,
                                 keep_n: int = 1) -> o3d.geometry.PointCloud:
    """
    DBSCAN 聚类，保留最大的 N 个簇 (去除桌面等大面积平面或零散噪点)。

    策略:
      - 默认保留最大簇 (通常是桌面)
      - 但如果你的物体比桌面小很多，需要反转: 去掉最大簇
      - 通过 keep_largest=False 实现

    参数:
        eps:          DBSCAN 邻域半径
        min_points:   最小簇点数
        keep_largest: True=保留最大簇, False=去掉最大簇(保留物体)
        keep_n:       保留前 N 大的簇 (keep_largest=True 时生效)
    """
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))

    if labels.max() < 0:
        print("[WARN] 未发现任何聚类")
        return pcd

    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    sorted_idx = np.argsort(-counts)  # 从大到小

    print(f"[INFO] DBSCAN 发现 {len(unique_labels)} 个簇:")
    for i, idx in enumerate(sorted_idx[:10]):
        lbl = unique_labels[idx]
        cnt = counts[idx]
        print(f"    簇 {lbl}: {cnt} 个点 ({cnt/len(labels)*100:.1f}%)")

    if keep_largest:
        # 保留前 N 大的簇
        keep_labels = set(unique_labels[sorted_idx[:keep_n]])
    else:
        # 去掉最大簇，保留其余所有
        largest_label = unique_labels[sorted_idx[0]]
        keep_labels = set(unique_labels) - {largest_label}
        print(f"[INFO] 去掉最大簇 (label={largest_label}, {counts[sorted_idx[0]]} 个点)")

    mask = np.array([l in keep_labels for l in labels])
    indices = np.where(mask)[0]
    filtered = pcd.select_by_index(indices.tolist())
    print(f"[INFO] 聚类过滤: {len(pcd.points)} → {len(filtered.points)} 个点")
    return filtered


def filter_pointcloud_remove_plane(pcd: o3d.geometry.PointCloud,
                                   distance_threshold: float = 0.005,
                                   ransac_n: int = 3,
                                   num_iterations: int = 1000) -> o3d.geometry.PointCloud:
    """
    RANSAC 平面检测 + 去除，适合去除桌面。

    反复检测并去除最大平面，直到剩余点不够。
    """
    remaining = pcd
    removed_total = 0

    for i in range(3):  # 最多去除 3 个平面
        if len(remaining.points) < 100:
            break

        plane_model, inliers = remaining.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations)

        if len(inliers) < 0.05 * len(remaining.points):
            # 平面太小，停止
            break

        [a, b, c, d] = plane_model
        print(f"[INFO] 检测到平面 {i+1}: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0 "
              f"({len(inliers)} 个内点)")

        # 去除平面内点
        remaining = remaining.select_by_index(inliers, invert=True)
        removed_total += len(inliers)

    print(f"[INFO] 平面去除: 共去除 {removed_total} 个点, 剩余 {len(remaining.points)} 个点")
    return remaining


# =============================================================================
# 综合物体提取流水线
# =============================================================================

def extract_object_from_pointcloud(pcd: o3d.geometry.PointCloud,
                                   method: str = "plane+cluster",
                                   bg_color: tuple = None,
                                   bg_threshold: float = 0.2,
                                   bbox: tuple = None,
                                   cluster_eps: float = 0.01,
                                   cluster_min: int = 100,
                                   plane_dist: float = 0.005) -> o3d.geometry.PointCloud:
    """
    综合物体提取流水线。

    method 可以是单个方法或组合 (用 + 分隔):
      "color"           - 颜色过滤
      "cluster"         - DBSCAN 聚类 (去掉最大簇)
      "plane"           - RANSAC 去平面
      "bbox"            - 包围盒裁剪
      "color+cluster"   - 先颜色过滤再聚类
      "color+plane"     - 先颜色过滤再去平面
      "plane+cluster"   - 先去平面再聚类
    """
    methods = method.lower().split('+')
    result = pcd

    for m in methods:
        m = m.strip()
        before = len(result.points)

        if m == "color":
            if bg_color is None:
                # 自动检测: 从点云颜色的众数估计背景色
                colors = np.asarray(result.colors)
                # 用中位数近似背景色
                bg_color_01 = tuple(np.median(colors, axis=0))
                print(f"[INFO] 自动检测背景色: RGB_01=({bg_color_01[0]:.3f}, {bg_color_01[1]:.3f}, {bg_color_01[2]:.3f})")
            else:
                bg_color_01 = tuple(c / 255.0 for c in bg_color)
            result = filter_pointcloud_by_color(result, bg_color_01, bg_threshold)

        elif m == "cluster":
            result = filter_pointcloud_by_cluster(
                result, eps=cluster_eps, min_points=cluster_min,
                keep_largest=False)

        elif m == "plane":
            result = filter_pointcloud_remove_plane(result, distance_threshold=plane_dist)

        elif m == "bbox":
            if bbox is None:
                print("[WARN] bbox 未指定，跳过")
                continue
            result = filter_pointcloud_by_bbox(result, bbox)

        else:
            print(f"[WARN] 未知过滤方法: {m}")

        after = len(result.points)
        print(f"[INFO] 方法 '{m}': {before} → {after} 个点")

    # 最后再做一次离群点去除
    if len(result.points) > 50:
        cl, ind = result.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        removed = len(result.points) - len(ind)
        result = result.select_by_index(ind)
        if removed > 0:
            print(f"[INFO] 去除离群点: {removed} 个")

    print(f"[INFO] ★ 物体提取完成: {len(result.points)} 个点")
    return result


# =============================================================================
# 图像级过滤版本的多视角点云生成 (在反投影前就过滤)
# =============================================================================

def generate_pointcloud_object_only(rgb_dir: str,
                                    depth_dir: str,
                                    bg_color: tuple = None,
                                    bg_threshold: float = 60.0,
                                    depth_trunc: float = 10.0,
                                    voxel_size: float = 0.002,
                                    post_filter: str = "cluster",
                                    cluster_eps: float = 0.01,
                                    cluster_min: int = 100) -> o3d.geometry.PointCloud:
    """
    ★ 推荐方法: 在图像级就做前景/背景分割，然后再反投影。

    这样比先生成全场景点云再过滤更干净，因为不会引入背景深度的噪点。
    """
    if cv2 is None:
        raise ImportError("需要 opencv-python: pip install opencv-python")

    rgb_files = sorted(
        list(Path(rgb_dir).glob('*.png')) +
        list(Path(rgb_dir).glob('*.jpg')) +
        list(Path(rgb_dir).glob('*.jpeg'))
    )
    if not rgb_files:
        raise FileNotFoundError(f"在 {rgb_dir} 中未找到图像")

    print(f"[INFO] 找到 {len(rgb_files)} 个图像文件")
    print(f"[INFO] ★ 物体提取模式: 图像级颜色过滤 + 点云级后处理")

    combined_pcd = o3d.geometry.PointCloud()

    for rgb_path in rgb_files:
        stem = rgb_path.stem

        rgb = cv2.imread(str(rgb_path))
        if rgb is None:
            print(f"  [WARN] 无法读取: {rgb_path}")
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # 读深度图
        depth_path = None
        for ext in ['.png', '.tiff', '.exr', '.npy']:
            candidate = Path(depth_dir) / f"{stem}{ext}"
            if candidate.exists():
                depth_path = candidate
                break
        if depth_path is None:
            print(f"  [WARN] 未找到 {stem} 的深度图，跳过")
            continue

        if str(depth_path).endswith('.npy'):
            depth = np.load(str(depth_path))
        else:
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            print(f"  [WARN] 无法读取深度图: {depth_path}")
            continue

        # ★ 图像级物体 mask
        if bg_color is None:
            detected_bg = auto_detect_background_color(rgb)
        else:
            detected_bg = bg_color
        obj_mask = create_object_mask_by_color(rgb, detected_bg, bg_threshold)

        # 查找外参
        T_c2w = CAMERAS.get(stem, None)
        if T_c2w is None:
            print(f"  [WARN] CAMERAS 中未配置 '{stem}' 的外参，使用单位矩阵")
            T_c2w = np.eye(4)

        # ★ 带 mask 的反投影
        pcd = depth_to_pointcloud_with_mask(rgb, depth, K, T_c2w, DEPTH_SCALE, depth_trunc,
                                            object_mask=obj_mask)
        n = len(pcd.points)
        print(f"  视角 '{stem}': {n} 个物体点")
        combined_pcd += pcd

    # 体素下采样
    if voxel_size > 0 and len(combined_pcd.points) > 0:
        before = len(combined_pcd.points)
        combined_pcd = combined_pcd.voxel_down_sample(voxel_size)
        print(f"[INFO] 体素下采样: {before} → {len(combined_pcd.points)} 个点")

    # 点云级后处理
    if post_filter and len(combined_pcd.points) > 100:
        print(f"[INFO] 点云级后处理: {post_filter}")
        combined_pcd = extract_object_from_pointcloud(
            combined_pcd,
            method=post_filter,
            cluster_eps=cluster_eps,
            cluster_min=cluster_min)

    print(f"[INFO] ★ 最终物体点云: {len(combined_pcd.points)} 个点")
    return combined_pcd


# =============================================================================
# 原始多视角点云生成 (不过滤)
# =============================================================================

def generate_pointcloud_from_multiview(rgb_dir: str,
                                       depth_dir: str,
                                       depth_trunc: float = 10.0,
                                       voxel_size: float = 0.002) -> o3d.geometry.PointCloud:
    """从多视角图像生成融合点云 (原版，不过滤背景)。"""
    if cv2 is None:
        raise ImportError("需要 opencv-python: pip install opencv-python")

    rgb_files = sorted(
        list(Path(rgb_dir).glob('*.png')) +
        list(Path(rgb_dir).glob('*.jpg')) +
        list(Path(rgb_dir).glob('*.jpeg'))
    )
    if not rgb_files:
        raise FileNotFoundError(f"在 {rgb_dir} 中未找到图像")

    print(f"[INFO] 找到 {len(rgb_files)} 个图像文件")
    print(f"[INFO] 已配置 {len(CAMERAS)} 个视角外参: {list(CAMERAS.keys())}")

    combined_pcd = o3d.geometry.PointCloud()

    for rgb_path in rgb_files:
        stem = rgb_path.stem

        rgb = cv2.imread(str(rgb_path))
        if rgb is None:
            print(f"  [WARN] 无法读取: {rgb_path}")
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        depth_path = None
        for ext in ['.png', '.tiff', '.exr', '.npy']:
            candidate = Path(depth_dir) / f"{stem}{ext}"
            if candidate.exists():
                depth_path = candidate
                break
        if depth_path is None:
            print(f"  [WARN] 未找到 {stem} 的深度图，跳过")
            continue

        if str(depth_path).endswith('.npy'):
            depth = np.load(str(depth_path))
        else:
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            print(f"  [WARN] 无法读取深度图: {depth_path}")
            continue

        T_c2w = CAMERAS.get(stem, None)
        if T_c2w is None:
            print(f"  [WARN] CAMERAS 中未配置 '{stem}' 的外参，使用单位矩阵")
            T_c2w = np.eye(4)

        pcd = depth_to_pointcloud(rgb, depth, K, T_c2w, DEPTH_SCALE, depth_trunc)
        n = len(pcd.points)
        print(f"  视角 '{stem}': {n} 个点")
        combined_pcd += pcd

    if voxel_size > 0 and len(combined_pcd.points) > 0:
        before = len(combined_pcd.points)
        combined_pcd = combined_pcd.voxel_down_sample(voxel_size)
        print(f"[INFO] 体素下采样: {before} → {len(combined_pcd.points)} 个点")

    if len(combined_pcd.points) > 100:
        cl, ind = combined_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        removed = len(combined_pcd.points) - len(ind)
        combined_pcd = combined_pcd.select_by_index(ind)
        print(f"[INFO] 去除离群点: {removed} 个")

    print(f"[INFO] 最终点云: {len(combined_pcd.points)} 个点")
    return combined_pcd


# =============================================================================
# Demo 点云
# =============================================================================

def create_demo_pointcloud(n_points: int = 60000) -> o3d.geometry.PointCloud:
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
    pcd1 = mesh_sphere.sample_points_uniformly(n_points // 3)
    pcd1.paint_uniform_color([0.7, 0.7, 0.7])

    mesh_cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=0.15, height=0.6)
    mesh_cyl.translate([0, 0, -0.6])
    pcd2 = mesh_cyl.sample_points_uniformly(n_points // 3)
    pcd2.paint_uniform_color([0.7, 0.7, 0.7])

    mesh_cone = o3d.geometry.TriangleMesh.create_cone(radius=0.2, height=0.4)
    mesh_cone.translate([0, 0, 0.4])
    pcd3 = mesh_cone.sample_points_uniformly(n_points // 3)
    pcd3.paint_uniform_color([0.7, 0.7, 0.7])

    combined = o3d.geometry.PointCloud()
    for p in [pcd1, pcd2, pcd3]:
        combined += p
    return combined


# =============================================================================
# 交互式标注 (保持原版)
# =============================================================================

LABEL_COLORS = np.array([
    [0.7, 0.7, 0.7],   # 0: 未标注 (灰)
    [1.0, 0.0, 0.0],   # 1: 红
    [0.0, 1.0, 0.0],   # 2: 绿
    [0.0, 0.0, 1.0],   # 3: 蓝
    [1.0, 1.0, 0.0],   # 4: 黄
    [1.0, 0.0, 1.0],   # 5: 品红
    [0.0, 1.0, 1.0],   # 6: 青
    [1.0, 0.5, 0.0],   # 7: 橙
    [0.5, 0.0, 1.0],   # 8: 紫
    [0.0, 0.5, 0.0],   # 9: 深绿
], dtype=np.float64)

LABEL_NAMES = ['未标注', '红', '绿', '蓝', '黄', '品红', '青', '橙', '紫', '深绿']


class PointCloudLabeler:
    def __init__(self, pcd: o3d.geometry.PointCloud):
        self.pcd = pcd
        self.points = np.asarray(pcd.points).copy()
        self.n_points = len(self.points)

        if pcd.has_colors():
            self.original_colors = np.asarray(pcd.colors).copy()
        else:
            self.original_colors = np.ones((self.n_points, 3)) * 0.7

        self.labels = np.zeros(self.n_points, dtype=np.int64)
        self.current_label = 1
        self.history = []
        self.kdtree = o3d.geometry.KDTreeFlann(pcd)

    def _print_help(self):
        print("""
╔══════════════════════════════════════════════════════╗
║               点云交互式标注工具                       ║
╠══════════════════════════════════════════════════════╣
║  E → 框选标注    N → 点选+邻域    V → 预览           ║
║  1-9 → 切换标签  U → 撤销  R → 重置  S → 保存       ║
║  Q → 退出                                            ║
╚══════════════════════════════════════════════════════╝
""")

    def _update_colors(self):
        colors = self.original_colors.copy()
        for lbl in range(1, int(self.labels.max()) + 1):
            mask = self.labels == lbl
            if mask.any():
                colors[mask] = LABEL_COLORS[lbl % len(LABEL_COLORS)]
        self.pcd.colors = o3d.utility.Vector3dVector(colors)

    def _label_indices(self, indices, label):
        if len(indices) == 0:
            return
        self.history.append((indices.copy(), self.labels[indices].copy()))
        self.labels[indices] = label
        self._update_colors()
        labeled = np.sum(self.labels > 0)
        name = LABEL_NAMES[label] if label < len(LABEL_NAMES) else str(label)
        print(f"  ✓ 标注 {len(indices)} 个点 → label {label} ({name})  "
              f"[已标注 {labeled}/{self.n_points}]")

    def _undo(self):
        if not self.history:
            print("  无可撤销操作")
            return
        indices, old = self.history.pop()
        self.labels[indices] = old
        self._update_colors()
        print(f"  ✓ 撤销 {len(indices)} 个点")

    def _reset(self):
        self.history.clear()
        self.labels[:] = 0
        self._update_colors()
        print("  ✓ 所有标注已清除")

    def _print_status(self):
        labeled = np.sum(self.labels > 0)
        name = LABEL_NAMES[self.current_label] if self.current_label < len(LABEL_NAMES) else str(self.current_label)
        print(f"\n── label={self.current_label}({name}) | 已标注 {labeled}/{self.n_points} ──")

    def _edit_crop_mode(self):
        print(f"\n  [框选模式] label = {self.current_label}")
        print("  → K → 画框 → C 确认 → 关闭窗口")

        self._update_colors()
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name=f"框选 | label={self.current_label}", width=1280, height=720)
        vis.add_geometry(self.pcd)
        vis.run()
        vis.destroy_window()

        cropped = vis.get_cropped_geometry()
        cropped_pts = np.asarray(cropped.points)

        if len(cropped_pts) == 0 or len(cropped_pts) == self.n_points:
            print("  未框选，跳过")
            return

        indices = []
        for pt in cropped_pts:
            [_, idx, _] = self.kdtree.search_knn_vector_3d(pt, 1)
            indices.append(idx[0])
        indices = np.unique(np.array(indices))

        mode = input(f"  框内 {len(indices)} 个点. 标注 [I]框内/[O]框外? (默认 I): ").strip().upper()
        if mode == 'O':
            indices = np.setdiff1d(np.arange(self.n_points), indices)
        self._label_indices(indices, self.current_label)

    def _vertex_pick_mode(self):
        radius_s = input(f"  邻域半径 (默认 0.03): ").strip()
        radius = float(radius_s) if radius_s else 0.03

        self._update_colors()
        vis = o3d.visualization.VisualizerWithVertexSelection()
        vis.create_window(window_name=f"点选 | Shift+点击", width=1280, height=720)
        vis.add_geometry(self.pcd)
        vis.run()
        vis.destroy_window()

        picked = vis.get_picked_points()
        if not picked:
            print("  未选点，跳过")
            return

        all_indices = set()
        for p in picked:
            [k, nbr_idx, _] = self.kdtree.search_radius_vector_3d(self.points[p.index], radius)
            all_indices.update(nbr_idx)
        self._label_indices(np.array(list(all_indices), dtype=np.int64), self.current_label)

    def _preview(self):
        self._update_colors()
        unique, counts = np.unique(self.labels, return_counts=True)
        print("  当前标注分布:")
        for u, c in zip(unique, counts):
            name = LABEL_NAMES[u] if u < len(LABEL_NAMES) else str(u)
            print(f"    label {u} ({name}): {c} ({c/self.n_points*100:.1f}%)")
        o3d.visualization.draw_geometries([self.pcd], window_name="预览", width=1280, height=720)

    def get_result(self):
        result = {
            'points': self.points.copy(),
            'colors': self.original_colors.copy(),
            'labels': self.labels.copy(),
        }
        if torch is not None:
            result['points_tensor'] = torch.from_numpy(result['points']).float()
            result['colors_tensor'] = torch.from_numpy(result['colors']).float()
            result['labels_tensor'] = torch.from_numpy(result['labels']).long()
        return result

    def save_result(self, output_path):
        result = self.get_result()
        if output_path.endswith('.pt') and torch is not None:
            torch.save({
                'points': result['points_tensor'],
                'colors': result['colors_tensor'],
                'labels': result['labels_tensor'],
            }, output_path)
        else:
            if not output_path.endswith('.npz'):
                output_path += '.npz'
            np.savez(output_path, points=result['points'], colors=result['colors'], labels=result['labels'])

        unique, counts = np.unique(result['labels'], return_counts=True)
        print(f"\n  ✓ 已保存: {output_path}")
        for u, c in zip(unique, counts):
            name = LABEL_NAMES[u] if u < len(LABEL_NAMES) else str(u)
            print(f"    label {u} ({name}): {c} ({c/self.n_points*100:.1f}%)")
        return output_path

    def run_interactive(self, output_path="labeled_result.pt"):
        self._update_colors()
        print(f"\n[INFO] 点云: {self.n_points} 个点")
        self._print_help()

        while True:
            self._print_status()
            cmd = input("  命令 [E/N/V/1-9/U/R/S/H/Q]: ").strip().upper()

            if cmd == 'Q':
                if np.sum(self.labels > 0) > 0:
                    if input("  保存后退出? (y/n): ").strip().lower() == 'y':
                        self.save_result(output_path)
                break
            elif cmd == 'S':  self.save_result(output_path)
            elif cmd == 'U':  self._undo()
            elif cmd == 'V':  self._preview()
            elif cmd == 'E':  self._edit_crop_mode()
            elif cmd == 'N':  self._vertex_pick_mode()
            elif cmd == 'H':  self._print_help()
            elif cmd == 'R':
                if input("  确认重置? (y/n): ").strip().lower() == 'y':
                    self._reset()
            elif cmd in [str(i) for i in range(10)]:
                self.current_label = int(cmd)
                name = LABEL_NAMES[self.current_label] if self.current_label < len(LABEL_NAMES) else ''
                print(f"  ✓ 切换到 label {self.current_label} ({name})")
            else:
                print("  未知命令，按 H 查看帮助")


class SimpleLabeler:
    def __init__(self, pcd):
        self.pcd = pcd
        self.points = np.asarray(pcd.points).copy()
        self.n_points = len(self.points)
        self.original_colors = (np.asarray(pcd.colors).copy() if pcd.has_colors()
                                else np.ones((self.n_points, 3)) * 0.7)
        self.labels = np.zeros(self.n_points, dtype=np.int64)
        self.history = []
        self.current_label = 1

    def _update_colors(self):
        colors = self.original_colors.copy()
        for lbl in range(1, max(1, int(self.labels.max()) + 1)):
            mask = self.labels == lbl
            if mask.any():
                colors[mask] = LABEL_COLORS[lbl % len(LABEL_COLORS)]
        self.pcd.colors = o3d.utility.Vector3dVector(colors)

    def get_result(self):
        r = {'points': self.points.copy(), 'colors': self.original_colors.copy(),
             'labels': self.labels.copy()}
        if torch:
            r['points_tensor'] = torch.from_numpy(r['points']).float()
            r['colors_tensor'] = torch.from_numpy(r['colors']).float()
            r['labels_tensor'] = torch.from_numpy(r['labels']).long()
        return r

    def run(self, output_path="labeled_result.pt"):
        mins, maxs = self.points.min(0), self.points.max(0)
        print(f"\n  点云范围: X[{mins[0]:.4f},{maxs[0]:.4f}] "
              f"Y[{mins[1]:.4f},{maxs[1]:.4f}] Z[{mins[2]:.4f},{maxs[2]:.4f}]")

        while True:
            print(f"\n  label={self.current_label} | [V]预览 [B]框选 [1-9]标签 [S]保存 [Q]退出")
            cmd = input("  命令: ").strip().upper()
            if cmd == 'Q': break
            elif cmd == 'V':
                self._update_colors()
                o3d.visualization.draw_geometries([self.pcd], width=1280, height=720)
            elif cmd == 'S':
                r = self.get_result()
                if output_path.endswith('.pt') and torch:
                    torch.save({k: v for k, v in r.items() if 'tensor' in k}, output_path)
                else:
                    np.savez(output_path.replace('.pt', '.npz'),
                             **{k: v for k, v in r.items() if 'tensor' not in k})
                print(f"  ✓ 已保存 {output_path}")
            elif cmd in [str(i) for i in range(10)]:
                self.current_label = int(cmd)
            elif cmd == 'B':
                try:
                    print("  输入范围 (回车=不限):")
                    x0 = float(input(f"    x_min[{mins[0]:.4f}]: ") or mins[0])
                    x1 = float(input(f"    x_max[{maxs[0]:.4f}]: ") or maxs[0])
                    y0 = float(input(f"    y_min[{mins[1]:.4f}]: ") or mins[1])
                    y1 = float(input(f"    y_max[{maxs[1]:.4f}]: ") or maxs[1])
                    z0 = float(input(f"    z_min[{mins[2]:.4f}]: ") or mins[2])
                    z1 = float(input(f"    z_max[{maxs[2]:.4f}]: ") or maxs[2])
                    m = ((self.points[:, 0] >= x0) & (self.points[:, 0] <= x1) &
                         (self.points[:, 1] >= y0) & (self.points[:, 1] <= y1) &
                         (self.points[:, 2] >= z0) & (self.points[:, 2] <= z1))
                    idx = np.where(m)[0]
                    if len(idx) > 0:
                        self.history.append((idx.copy(), self.labels[idx].copy()))
                        self.labels[idx] = self.current_label
                        self._update_colors()
                        print(f"  ✓ 标注 {len(idx)} 个点")
                    else:
                        print("  未选中")
                except ValueError:
                    print("  格式错误")


# =============================================================================
# 主程序
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="多视角点云生成 + 物体提取 + 交互式标注")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--demo', action='store_true', help='Demo 模式')
    group.add_argument('--generate', action='store_true', help='从图像生成点云')
    group.add_argument('--load_pcd', type=str, default="/home/lmy/code/pc_dataset/pc/013_bowl.ply", help='加载点云 (.ply/.pcd)')

    parser.add_argument('--rgb_dir', type=str, default="/home/lmy/code/pointcloud_reconstruct/banana/rgb")
    parser.add_argument('--depth_dir', type=str, default="/home/lmy/code/pointcloud_reconstruct/banana/depth")
    parser.add_argument('--depth_trunc', type=float, default=10.0)
    parser.add_argument('--voxel_size', type=float, default=0.002)
    parser.add_argument('--output', type=str, default='labeled_result.pt')
    parser.add_argument('--simple', action='store_true')

    # ★ 新增: 物体提取参数
    parser.add_argument('--object_only', action='store_true',
                        help='只保留物体，去除背景')
    parser.add_argument('--filter_method', type=str, default='color',
                        help='过滤方法: color / cluster / plane / bbox / color+cluster / color+plane (默认 color)')
    parser.add_argument('--bg_color', type=str, default=None,
                        help='背景色 RGB, 如 "150,145,180" (不指定则自动检测)')
    parser.add_argument('--bg_threshold', type=float, default=60.0,
                        help='背景色容差, 图像级 [0-255] (默认 60)')
    parser.add_argument('--bg_threshold_pcd', type=float, default=0.2,
                        help='背景色容差, 点云级 [0-1] (默认 0.2)')
    parser.add_argument('--bbox', type=str, default=None,
                        help='包围盒 "x0,x1,y0,y1,z0,z1"')
    parser.add_argument('--cluster_eps', type=float, default=0.01,
                        help='DBSCAN eps (默认 0.01)')
    parser.add_argument('--cluster_min', type=int, default=100,
                        help='DBSCAN min_points (默认 100)')
    parser.add_argument('--save_ply', type=str, default="/home/lmy/code/isaaclab5/scene_reconstruction/pc/013_bowl.ply",
                        help='额外保存为 PLY 文件 (方便预览)')

    args = parser.parse_args()

    # 解析 bg_color
    bg_color = None
    if args.bg_color:
        bg_color = tuple(int(x) for x in args.bg_color.split(','))

    # 解析 bbox
    bbox = None
    if args.bbox:
        bbox = tuple(float(x) for x in args.bbox.split(','))

    if args.demo:
        print("[INFO] 生成 Demo 点云...")
        pcd = create_demo_pointcloud()

    elif args.generate:
        if not args.rgb_dir or not args.depth_dir:
            parser.error("--generate 需要 --rgb_dir 和 --depth_dir")

        if args.object_only:
            # ★ 新流程: 图像级过滤 + 点云级后处理
            print("=" * 60)
            print("  ★ 物体提取模式 ★")
            print(f"  过滤方法: {args.filter_method}")
            print(f"  背景色: {'自动检测' if bg_color is None else bg_color}")
            print(f"  容差: {args.bg_threshold}")
            print("=" * 60)

            # 判断是否需要图像级颜色过滤
            methods = args.filter_method.lower().split('+')
            if 'color' in methods:
                # 图像级颜色过滤 + 后续处理
                post_methods = [m for m in methods if m != 'color']
                post_filter = '+'.join(post_methods) if post_methods else None
                pcd = generate_pointcloud_object_only(
                    args.rgb_dir, args.depth_dir,
                    bg_color=bg_color,
                    bg_threshold=args.bg_threshold,
                    depth_trunc=args.depth_trunc,
                    voxel_size=args.voxel_size,
                    post_filter=post_filter,
                    cluster_eps=args.cluster_eps,
                    cluster_min=args.cluster_min)
            else:
                # 先生成完整点云，再做点云级过滤
                pcd = generate_pointcloud_from_multiview(
                    args.rgb_dir, args.depth_dir, args.depth_trunc, args.voxel_size)
                pcd = extract_object_from_pointcloud(
                    pcd,
                    method=args.filter_method,
                    bg_color=bg_color,
                    bg_threshold=args.bg_threshold_pcd,
                    bbox=bbox,
                    cluster_eps=args.cluster_eps,
                    cluster_min=args.cluster_min)
        else:
            # 原始流程
            pcd = generate_pointcloud_from_multiview(
                args.rgb_dir, args.depth_dir, args.depth_trunc, args.voxel_size)

    else:
        print(f"[INFO] 加载: {args.load_pcd}")
        pcd = o3d.io.read_point_cloud(args.load_pcd)
        if not pcd.has_colors():
            pcd.paint_uniform_color([0.7, 0.7, 0.7])

        # 对已加载的点云也可以做物体提取
        if args.object_only:
            pcd = extract_object_from_pointcloud(
                pcd,
                method=args.filter_method,
                bg_color=bg_color,
                bg_threshold=args.bg_threshold_pcd,
                bbox=bbox,
                cluster_eps=args.cluster_eps,
                cluster_min=args.cluster_min)

    if pcd is None or len(pcd.points) == 0:
        print("[错误] 无有效点云")
        sys.exit(1)

    print(f"[INFO] 点云: {len(pcd.points)} 个点")

    # 额外保存 PLY
    if args.save_ply:
        o3d.io.write_point_cloud(args.save_ply, pcd)
        print(f"[INFO] 已保存 PLY: {args.save_ply}")

    if args.simple:
        labeler = SimpleLabeler(pcd)
        labeler.run(args.output)
    else:
        labeler = PointCloudLabeler(pcd)
        labeler.run_interactive(args.output)

    result = labeler.get_result()
    print("\n" + "=" * 50)
    print("  标注结果:")
    print(f"    points : {result['points'].shape}")
    print(f"    labels : {result['labels'].shape}")
    if torch:
        print(f"    points_tensor: {result['points_tensor'].shape} {result['points_tensor'].dtype}")
        print(f"    labels_tensor: {result['labels_tensor'].shape} {result['labels_tensor'].dtype}")
    unique, counts = np.unique(result['labels'], return_counts=True)
    for u, c in zip(unique, counts):
        name = LABEL_NAMES[u] if u < len(LABEL_NAMES) else str(u)
        print(f"    label {u} ({name}): {c} ({c/len(result['labels'])*100:.1f}%)")
    print("=" * 50)


if __name__ == '__main__':
    main()