"""点云处理流水线工具

整合 mesh → 点云转换、降采样、交互式标注 三个功能。
支持单独使用或串联执行。

Usage:
    # 完整流水线：mesh → 点云 → 降采样 → 标注
    python pc_pipeline.py full --method fps --target_num 2048 --input model.obj --output labeled_result.pt

    # 仅 mesh → 点云
    python pc_pipeline.py mesh2pc --input model.obj --output cloud.ply

    # 仅降采样
    python pc_pipeline.py downsample --input cloud.ply --output cloud_ds.ply

    # 仅标注
    python pc_pipeline.py label --input cloud.ply --output labeled_result.pt
"""

import argparse
import os
import sys
import numpy as np
import open3d as o3d

try:
    import trimesh
except ImportError:
    trimesh = None

try:
    import torch
except ImportError:
    torch = None


# =============================================================================
# 点云 IO
# =============================================================================

def load_pointcloud(path: str) -> o3d.geometry.PointCloud:
    """加载点云，支持 .ply / .pcd / .npz"""
    if path.endswith(".npz"):
        data = np.load(path)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data["points"])
        if "colors" in data:
            pcd.colors = o3d.utility.Vector3dVector(data["colors"])
    else:
        pcd = o3d.io.read_point_cloud(path)
    if pcd is None or len(pcd.points) == 0:
        raise ValueError(f"无法加载点云: {path}")
    return pcd


def ensure_parent_dir(path: str):
    """保存前确保父目录存在。"""
    parent_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent_dir, exist_ok=True)


def save_pointcloud(pcd: o3d.geometry.PointCloud, path: str):
    """保存点云，支持 .ply / .pcd / .npz / .pt"""
    ensure_parent_dir(path)
    if path.endswith(".npz"):
        save_dict = {"points": np.asarray(pcd.points)}
        if pcd.has_colors():
            save_dict["colors"] = np.asarray(pcd.colors)
        if pcd.has_normals():
            save_dict["normals"] = np.asarray(pcd.normals)
        np.savez(path, **save_dict)
    elif path.endswith(".pt"):
        if torch is None:
            raise ImportError("保存 .pt 需要安装 torch")
        save_dict = {
            'points': torch.from_numpy(np.asarray(pcd.points)).float(),
        }
        if pcd.has_colors():
            save_dict['colors'] = torch.from_numpy(np.asarray(pcd.colors)).float()
        if pcd.has_normals():
            save_dict['normals'] = torch.from_numpy(np.asarray(pcd.normals)).float()
        torch.save(save_dict, path)
    else:
        o3d.io.write_point_cloud(path, pcd)
    print(f"[INFO] 已保存: {path} ({len(pcd.points)} 个点)")


# =============================================================================
# Mesh → 点云
# =============================================================================

def mesh_to_pointcloud(input_path: str, num_points: int = 2048) -> o3d.geometry.PointCloud:
    """将 OBJ/STL 等 mesh 文件转为点云"""
    if trimesh is None:
        raise ImportError("请安装 trimesh: pip install trimesh")

    tm = trimesh.load(input_path, force="mesh")
    if tm is None or len(tm.vertices) == 0:
        raise ValueError(f"无法加载 mesh: {input_path}")

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(tm.vertices)
    mesh.triangles = o3d.utility.Vector3iVector(tm.faces)
    mesh.compute_vertex_normals()
    print(f"[INFO] Mesh 加载: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 面片")

    # 过采样后用 FPS 降采样
    pcd = mesh.sample_points_uniformly(number_of_points=num_points * 3)
    try:
        pcd = pcd.farthest_point_down_sample(num_points)
    except AttributeError:
        bbox = pcd.get_axis_aligned_bounding_box()
        diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
        voxel_size = diag / 100.0
        while len(pcd.points) > num_points:
            pcd = pcd.voxel_down_sample(voxel_size)
            voxel_size *= 1.1
        if len(pcd.points) < num_points:
            pcd = mesh.sample_points_uniformly(number_of_points=num_points)

    print(f"[INFO] 采样得到 {len(pcd.points)} 个点")
    return pcd


# =============================================================================
# 降采样方法
# =============================================================================

def voxel_downsample(pcd: o3d.geometry.PointCloud, voxel_size: float) -> o3d.geometry.PointCloud:
    """体素降采样"""
    return pcd.voxel_down_sample(voxel_size)


def uniform_downsample(pcd: o3d.geometry.PointCloud, every_k: int) -> o3d.geometry.PointCloud:
    """均匀降采样，每隔 every_k 个点取一个"""
    return pcd.uniform_down_sample(every_k)


def random_downsample(pcd: o3d.geometry.PointCloud, target_num: int) -> o3d.geometry.PointCloud:
    """随机降采样到指定点数"""
    n = len(pcd.points)
    if target_num >= n:
        print(f"[WARN] 目标点数 {target_num} >= 当前点数 {n}，不做降采样")
        return pcd
    indices = np.random.choice(n, target_num, replace=False)
    return pcd.select_by_index(indices)


def fps_downsample(pcd: o3d.geometry.PointCloud, target_num: int) -> o3d.geometry.PointCloud:
    """最远点采样 (Farthest Point Sampling)"""
    points = np.asarray(pcd.points)
    n = len(points)
    if target_num >= n:
        print(f"[WARN] 目标点数 {target_num} >= 当前点数 {n}，不做降采样")
        return pcd

    selected = np.zeros(target_num, dtype=np.int64)
    distances = np.full(n, np.inf)
    selected[0] = np.random.randint(n)

    for i in range(1, target_num):
        dist = np.linalg.norm(points - points[selected[i - 1]], axis=1)
        distances = np.minimum(distances, dist)
        selected[i] = np.argmax(distances)

    return pcd.select_by_index(selected)


def downsample(pcd: o3d.geometry.PointCloud, method: str = "voxel",
               voxel_size: float = 0.005, every_k: int = 5,
               target_num: int = 4096) -> o3d.geometry.PointCloud:
    """统一降采样入口"""
    n_before = len(pcd.points)
    if method == "voxel":
        pcd_down = voxel_downsample(pcd, voxel_size)
    elif method == "uniform":
        pcd_down = uniform_downsample(pcd, every_k)
    elif method == "random":
        pcd_down = random_downsample(pcd, target_num)
    elif method == "fps":
        pcd_down = fps_downsample(pcd, target_num)
    else:
        raise ValueError(f"未知降采样方法: {method}")

    n_after = len(pcd_down.points)
    print(f"[INFO] 降采样: {n_before} → {n_after} 个点 ({n_after / n_before * 100:.1f}%)")
    return pcd_down


# =============================================================================
# 交互式标注
# =============================================================================

LABEL_COLORS = np.array([
    [0.7, 0.7, 0.7],  # 0: 未标注
    [1.0, 0.0, 0.0],  # 1: 红
    [0.0, 1.0, 0.0],  # 2: 绿
    [0.0, 0.0, 1.0],  # 3: 蓝
    [1.0, 1.0, 0.0],  # 4: 黄
    [1.0, 0.0, 1.0],  # 5: 品红
    [0.0, 1.0, 1.0],  # 6: 青
    [1.0, 0.5, 0.0],  # 7: 橙
    [0.5, 0.0, 1.0],  # 8: 紫
    [0.0, 0.5, 0.0],  # 9: 深绿
], dtype=np.float64)

LABEL_NAMES = ['未标注', '红', '绿', '蓝', '黄', '品红', '青', '橙', '紫', '深绿']


class PointCloudLabeler:
    def __init__(self, pcd: o3d.geometry.PointCloud):
        self.pcd = pcd
        self.points = np.asarray(pcd.points).copy()
        self.n_points = len(self.points)
        self.original_colors = (np.asarray(pcd.colors).copy() if pcd.has_colors()
                                else np.ones((self.n_points, 3)) * 0.7)
        self.labels = np.zeros(self.n_points, dtype=np.int64)
        self.current_label = 1
        self.history = []
        self.kdtree = o3d.geometry.KDTreeFlann(pcd)

    def _update_colors(self):
        colors = self.original_colors.copy()
        for lbl in range(1, max(1, int(self.labels.max()) + 1)):
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
        vis.create_window(window_name="点选 | Shift+点击", width=1280, height=720)
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

    def _bbox_mode(self):
        """坐标范围框选标注"""
        mins, maxs = self.points.min(0), self.points.max(0)
        try:
            print("  输入范围 (回车=不限):")
            x0 = float(input(f"    x_min[{mins[0]:.4f}]: ") or mins[0])
            x1 = float(input(f"    x_max[{maxs[0]:.4f}]: ") or maxs[0])
            y0 = float(input(f"    y_min[{mins[1]:.4f}]: ") or mins[1])
            y1 = float(input(f"    y_max[{maxs[1]:.4f}]: ") or maxs[1])
            z0 = float(input(f"    z_min[{mins[2]:.4f}]: ") or mins[2])
            z1 = float(input(f"    z_max[{maxs[2]:.4f}]: ") or maxs[2])
            mask = ((self.points[:, 0] >= x0) & (self.points[:, 0] <= x1) &
                    (self.points[:, 1] >= y0) & (self.points[:, 1] <= y1) &
                    (self.points[:, 2] >= z0) & (self.points[:, 2] <= z1))
            idx = np.where(mask)[0]
            if len(idx) > 0:
                self._label_indices(idx, self.current_label)
            else:
                print("  未选中")
        except ValueError:
            print("  格式错误")

    def _preview(self):
        self._update_colors()
        unique, counts = np.unique(self.labels, return_counts=True)
        print("  当前标注分布:")
        for u, c in zip(unique, counts):
            name = LABEL_NAMES[u] if u < len(LABEL_NAMES) else str(u)
            print(f"    label {u} ({name}): {c} ({c / self.n_points * 100:.1f}%)")
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
            ensure_parent_dir(output_path)
            torch.save({
                'points': result['points_tensor'],
                'colors': result['colors_tensor'],
                'labels': result['labels_tensor'],
            }, output_path)
        else:
            if not output_path.endswith('.npz'):
                output_path = os.path.splitext(output_path)[0] + '.npz'
            ensure_parent_dir(output_path)
            np.savez(output_path, points=result['points'],
                     colors=result['colors'], labels=result['labels'])

        unique, counts = np.unique(result['labels'], return_counts=True)
        print(f"\n  ✓ 已保存: {output_path}")
        for u, c in zip(unique, counts):
            name = LABEL_NAMES[u] if u < len(LABEL_NAMES) else str(u)
            print(f"    label {u} ({name}): {c} ({c / self.n_points * 100:.1f}%)")
        return output_path

    def run_interactive(self, output_path="labeled_result.pt"):
        self._update_colors()
        print(f"\n[INFO] 点云: {self.n_points} 个点")
        print("""
╔══════════════════════════════════════════════════════╗
║               点云交互式标注工具                       ║
╠══════════════════════════════════════════════════════╣
║  E → 框选标注    N → 点选+邻域    B → 坐标框选       ║
║  V → 预览  1-9 → 切换标签  U → 撤销  R → 重置       ║
║  S → 保存  Q → 退出                                  ║
╚══════════════════════════════════════════════════════╝
""")
        while True:
            self._print_status()
            cmd = input("  命令 [E/N/B/V/1-9/U/R/S/Q]: ").strip().upper()

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
            elif cmd == 'B':  self._bbox_mode()
            elif cmd == 'R':
                if input("  确认重置? (y/n): ").strip().lower() == 'y':
                    self._reset()
            elif cmd in [str(i) for i in range(10)]:
                self.current_label = int(cmd)
                name = LABEL_NAMES[self.current_label] if self.current_label < len(LABEL_NAMES) else ''
                print(f"  ✓ 切换到 label {self.current_label} ({name})")
            else:
                print("  未知命令")


def run_labeling(pcd: o3d.geometry.PointCloud, output_path: str, save_ply: str = None):
    """运行标注流程"""
    if not pcd.has_colors():
        pcd.paint_uniform_color([0.7, 0.7, 0.7])

    if save_ply:
        save_pointcloud(pcd, save_ply)

    labeler = PointCloudLabeler(pcd)
    labeler.run_interactive(output_path)

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
        print(f"    label {u} ({name}): {c} ({c / len(result['labels']) * 100:.1f}%)")
    print("=" * 50)
    return result


# =============================================================================
# CLI 子命令
# =============================================================================

def cmd_mesh2pc(args):
    pcd = mesh_to_pointcloud(args.input, args.num_points)
    save_pointcloud(pcd, args.output)
    if args.visualize:
        o3d.visualization.draw_geometries([pcd], window_name="Mesh → 点云", width=1280, height=720)


def cmd_downsample(args):
    pcd = load_pointcloud(args.input)
    print(f"[INFO] 输入点云: {len(pcd.points)} 个点")
    pcd_down = downsample(pcd, args.method, args.voxel_size, args.every_k, args.target_num)
    save_pointcloud(pcd_down, args.output)
    if args.visualize:
        o3d.visualization.draw_geometries([pcd_down], window_name="降采样结果", width=1280, height=720)


def cmd_label(args):
    pcd = load_pointcloud(args.input)
    print(f"[INFO] 点云: {len(pcd.points)} 个点")
    run_labeling(pcd, args.output, args.save_ply)


def cmd_full(args):
    """完整流水线: mesh → 点云 → 降采样 → 标注"""
    # Step 1: mesh → 点云
    print("\n" + "=" * 50)
    print("  Step 1: Mesh → 点云")
    print("=" * 50)
    pcd = mesh_to_pointcloud(args.input, args.num_points)

    # Step 2: 降采样
    print("\n" + "=" * 50)
    print("  Step 2: 降采样")
    print("=" * 50)
    pcd_down = downsample(pcd, args.method, args.voxel_size, args.every_k, args.target_num)

    # 保存中间结果
    if args.save_intermediate:
        save_pointcloud(pcd_down, args.save_intermediate)

    # Step 3: 标注
    print("\n" + "=" * 50)
    print("  Step 3: 交互式标注")
    print("=" * 50)
    run_labeling(pcd_down, args.output, args.save_ply)


def main():
    parser = argparse.ArgumentParser(description="点云处理流水线: mesh转换 / 降采样 / 标注")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # ---- mesh2pc ----
    p_m2p = subparsers.add_parser("mesh2pc", help="Mesh → 点云转换")
    p_m2p.add_argument("--input", type=str, default="/home/lmy/code/Real2Sim2Real-Pipeline-auto_task/models/obj/pink_cup.obj", help="输入 mesh 路径 (.obj/.stl 等)")
    p_m2p.add_argument("--output", type=str, default="/home/lmy/code/Real2Sim2Real-Pipeline-auto_task/pc_dataset_generate/pc/pinkcup_pick/pink_cup.ply", help="输出点云路径 (.ply/.pcd/.npz)")
    p_m2p.add_argument("--num_points", type=int, default=2048, help="目标点数 (默认: 2048)")
    p_m2p.add_argument("--visualize", action="store_true", help="可视化结果")

    # ---- downsample ----
    p_ds = subparsers.add_parser("downsample", help="点云降采样")
    p_ds.add_argument("--input", type=str, required=True, help="输入点云路径")
    p_ds.add_argument("--output", type=str, required=True, help="输出点云路径")
    p_ds.add_argument("--method", type=str, default="voxel",
                      choices=["voxel", "uniform", "random", "fps"], help="降采样方法 (默认: voxel)")
    p_ds.add_argument("--voxel_size", type=float, default=0.005, help="体素大小 (默认: 0.005)")
    p_ds.add_argument("--every_k", type=int, default=5, help="均匀采样间隔 (默认: 5)")
    p_ds.add_argument("--target_num", type=int, default=4096, help="目标点数 (默认: 4096)")
    p_ds.add_argument("--visualize", action="store_true", help="可视化结果")

    # ---- label ----
    p_lb = subparsers.add_parser("label", help="点云交互式标注")
    p_lb.add_argument("--input", type=str, required=True, help="输入点云路径")
    p_lb.add_argument("--output", type=str, default="labeled_result.pt", help="输出标注结果路径 (.pt/.npz)")
    p_lb.add_argument("--save_ply", type=str, default=None, help="额外保存 PLY 文件")

    # ---- full ----
    p_full = subparsers.add_parser("full", help="完整流水线: mesh → 降采样 → 标注")
    p_full.add_argument("--input", type=str, required=True, help="输入 mesh 路径")
    p_full.add_argument("--output", type=str, default="labeled_result.pt", help="输出标注结果路径")
    p_full.add_argument("--num_points", type=int, default=8000, help="mesh 采样点数 (默认: 8000)")
    p_full.add_argument("--method", type=str, default="voxel",
                        choices=["voxel", "uniform", "random", "fps"], help="降采样方法")
    p_full.add_argument("--voxel_size", type=float, default=0.005, help="体素大小")
    p_full.add_argument("--every_k", type=int, default=5, help="均匀采样间隔")
    p_full.add_argument("--target_num", type=int, default=4096, help="降采样目标点数")
    p_full.add_argument("--save_intermediate", type=str, default=None, help="保存降采样中间结果")
    p_full.add_argument("--save_ply", type=str, default=None, help="额外保存 PLY")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    {"mesh2pc": cmd_mesh2pc, "downsample": cmd_downsample,
     "label": cmd_label, "full": cmd_full}[args.command](args)


if __name__ == "__main__":
    main()
